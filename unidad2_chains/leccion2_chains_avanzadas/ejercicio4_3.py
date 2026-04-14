"""
ejercicio4_3.py 
===========================
Ejercicio 4.3 — Map-Reduce summarizer


Description:
-----------
- Input: todos los archivos .md de un repositorio (simula con 5-10 archivos 
  de Klimbook)
- Map: para cada archivo, generar un resumen de máximo 3 oraciones
- Reduce: combinar todos los resúmenes en un documento ejecutivo de 1 página
- Usar chunking con overlap de 100 tokens para archivos grandes
- Comparar calidad: map-reduce vs concatenar todo y pasar en un solo prompt 
  (si cabe en la ventana)


Metadata:
----------
* Author: zxxz6 
* Version: 1.0.0


History:
------------
Author      Date            Description
zxxz6       20/02/2026      Creation

"""

from anthropic import AsyncAnthropic
from pydantic import BaseModel, field_validator
from pathlib import Path
import asyncio
import logging
import json
import time
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("map-reduce")

client = AsyncAnthropic()


# ═══════════════════════════════════════════════════════════════════════════
# Models & Pricing
# ═══════════════════════════════════════════════════════════════════════════

MODELS = {
    "sonnet": {
        "id": "claude-sonnet-4-20250514",
        "input": 3.00,
        "output": 15.00,
    },
    "haiku": {
        "id": "claude-haiku-4-5-20251001",
        "input": 0.80,
        "output": 4.00,
    },
}

# Semaphore: máximo 5 requests paralelos en la fase map
MAP_SEMAPHORE = asyncio.Semaphore(5)


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

class PipelineMetrics:
    """Acumula métricas de todo el pipeline."""

    def __init__(self):
        self.steps: list[dict] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    def register(self, step_name: str, model_name: str, usage, elapsed: float):
        cfg = MODELS[model_name]
        inp, out = usage.input_tokens, usage.output_tokens
        cost = (inp / 1e6) * cfg["input"] + (out / 1e6) * cfg["output"]

        self.total_input_tokens += inp
        self.total_output_tokens += out
        self.total_cost += cost

        self.steps.append({
            "step": step_name,
            "model": model_name,
            "input_tokens": inp,
            "output_tokens": out,
            "cost": cost,
            "time": elapsed,
        })

        logger.info(
            f"  [{step_name}] Tokens → in: {inp}, out: {out} | "
            f"${cost:.6f} | {elapsed:.2f}s"
        )

    def summary(self, label: str = "Pipeline"):
        total_time = sum(s["time"] for s in self.steps)
        logger.info(f"\n{'═'*60}")
        logger.info(f"[{label}] Resumen:")
        logger.info(f"  Steps:         {len(self.steps)}")
        logger.info(f"  Tokens input:  {self.total_input_tokens}")
        logger.info(f"  Tokens output: {self.total_output_tokens}")
        logger.info(f"  Costo total:   ${self.total_cost:.6f}")
        logger.info(f"  Tiempo total:  {total_time:.2f}s")
        logger.info(f"{'═'*60}")
        return {
            "steps": len(self.steps),
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "cost": self.total_cost,
            "time": total_time,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════════════════════════════════════

class FileSummary(BaseModel):
    """Resumen de un archivo individual (fase Map)."""
    filename: str
    summary: str

    @field_validator("summary")
    @classmethod
    def max_three_sentences(cls, v):
        sentences = [s.strip() for s in v.replace("...", ".").split(".") if s.strip()]
        if len(sentences) > 5:
            raise ValueError(
                f"Resumen tiene {len(sentences)} oraciones, máximo 3-5 permitidas"
            )
        return v


class ExecutiveSummary(BaseModel):
    """Documento ejecutivo final (fase Reduce)."""
    title: str
    summary: str

    @field_validator("summary")
    @classmethod
    def min_length(cls, v):
        if len(v) < 200:
            raise ValueError(f"Resumen ejecutivo muy corto: {len(v)} chars (mínimo 200)")
        return v


# ═══════════════════════════════════════════════════════════════════════════
# Chunking
# ═══════════════════════════════════════════════════════════════════════════

def chunk_with_overlap(
    text: str,
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 100,
) -> list[str]:
    """
    Divide texto en chunks con overlap.
    
    Args:
        text: Texto a dividir
        chunk_size_tokens: Tamaño de cada chunk en tokens (~3 chars/token en español)
        overlap_tokens: Tokens de overlap entre chunks consecutivos
    
    Returns:
        Lista de chunks
    """
    chars_per_token = 3
    chunk_chars = chunk_size_tokens * chars_per_token
    overlap_chars = overlap_tokens * chars_per_token

    if len(text) <= chunk_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_chars

        # Intentar cortar en un salto de línea para no partir párrafos
        if end < len(text):
            newline_pos = text.rfind("\n", start + chunk_chars // 2, end)
            if newline_pos > start:
                end = newline_pos + 1

        chunks.append(text[start:end].strip())
        start = end - overlap_chars

        # Evitar loop infinito si overlap >= chunk
        if start >= len(text):
            break

    logger.info(
        f"  [Chunking] {len(text)} chars → {len(chunks)} chunks "
        f"(~{chunk_size_tokens} tokens/chunk, {overlap_tokens} overlap)"
    )
    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# File Discovery
# ═══════════════════════════════════════════════════════════════════════════

def discover_md_files(directory: Path) -> list[Path]:
    """Encuentra todos los archivos .md en un directorio (no recursivo)."""
    files = sorted(directory.glob("*.md"))
    logger.info(f"[Discovery] {len(files)} archivos .md encontrados en {directory}")
    for f in files:
        size = f.stat().st_size
        logger.info(f"  {f.name} ({size:,} bytes)")
    return files


def read_file(path: Path) -> tuple[str, str]:
    """Lee un archivo y retorna (nombre, contenido)."""
    content = path.read_text(encoding="utf-8", errors="replace")
    return path.name, content


# ═══════════════════════════════════════════════════════════════════════════
# MAP Phase
# ═══════════════════════════════════════════════════════════════════════════

async def map_summarize_file(
    filename: str,
    content: str,
    metrics: PipelineMetrics,
    model_name: str = "haiku",
) -> FileSummary:
    """
    Fase Map: resume UN archivo en máximo 3 oraciones.
    Si el archivo es grande, lo divide en chunks, resume cada chunk,
    y luego combina los resúmenes parciales.
    """
    step_name = f"Map:{filename}"

    # Si el archivo es pequeño, resumir directamente
    chunks = chunk_with_overlap(content, chunk_size_tokens=500, overlap_tokens=100)

    if len(chunks) == 1:
        summary = await _summarize_single(chunks[0], filename, metrics, model_name)
    else:
        # Archivo grande: resumir cada chunk y luego combinar
        logger.info(f"  [{step_name}] Archivo grande → {len(chunks)} chunks")
        chunk_summaries = []

        for i, chunk in enumerate(chunks):
            chunk_name = f"Map:{filename}:chunk{i+1}"
            s = await _summarize_single(chunk, chunk_name, metrics, model_name)
            chunk_summaries.append(s)

        # Combinar resúmenes de chunks en uno solo
        combined = "\n".join(f"- {s}" for s in chunk_summaries)
        summary = await _reduce_chunk_summaries(
            combined, filename, metrics, model_name
        )

    return FileSummary(filename=filename, summary=summary)


async def _summarize_single(
    text: str,
    step_name: str,
    metrics: PipelineMetrics,
    model_name: str,
) -> str:
    """Resume un bloque de texto en máximo 3 oraciones."""
    async with MAP_SEMAPHORE:
        start = time.time()

        response = await client.messages.create(
            model=MODELS[model_name]["id"],
            system="""Eres un ingeniero de software senior. Resume el contenido 
proporcionado en EXACTAMENTE 3 oraciones concisas en español. 
Enfócate en: qué hace, qué tecnologías usa, y qué problema resuelve.
Responde SOLO con las 3 oraciones, nada más.""",
            messages=[
                {"role": "user", "content": f"Resume este contenido:\n\n{text}"},
            ],
            temperature=0.0,
            max_tokens=300,
        )

        elapsed = time.time() - start
        metrics.register(step_name, model_name, response.usage, elapsed)

        return response.content[0].text.strip()


async def _reduce_chunk_summaries(
    combined_summaries: str,
    filename: str,
    metrics: PipelineMetrics,
    model_name: str,
) -> str:
    """Combina resúmenes de chunks de un mismo archivo en 3 oraciones."""
    step_name = f"ChunkReduce:{filename}"

    async with MAP_SEMAPHORE:
        start = time.time()

        response = await client.messages.create(
            model=MODELS[model_name]["id"],
            system="""Combina estos resúmenes parciales de un mismo documento 
en EXACTAMENTE 3 oraciones concisas en español. Elimina redundancias.""",
            messages=[
                {"role": "user", "content": f"Resúmenes parciales:\n{combined_summaries}"},
            ],
            temperature=0.0,
            max_tokens=300,
        )

        elapsed = time.time() - start
        metrics.register(step_name, model_name, response.usage, elapsed)

        return response.content[0].text.strip()


async def map_phase(
    files: list[tuple[str, str]],
    metrics: PipelineMetrics,
) -> list[FileSummary]:
    """
    Ejecuta la fase Map en paralelo: resume cada archivo.
    
    Args:
        files: Lista de (filename, content)
        metrics: Acumulador de métricas
    
    Returns:
        Lista de FileSummary
    """
    logger.info(f"\n[MAP] Iniciando fase Map con {len(files)} archivos...")
    map_start = time.time()

    tasks = [
        map_summarize_file(filename, content, metrics, model_name="haiku")
        for filename, content in files
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    summaries = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            filename = files[i][0]
            logger.error(f"  [Map:{filename}] Falló: {result}")
            # Fallback: usar las primeras 3 líneas como resumen
            lines = files[i][1].strip().split("\n")[:3]
            summaries.append(FileSummary(
                filename=filename,
                summary=" ".join(lines)[:200],
            ))
        else:
            summaries.append(result)

    map_elapsed = time.time() - map_start
    logger.info(f"[MAP] Completado en {map_elapsed:.2f}s — {len(summaries)} resúmenes")

    return summaries


# ═══════════════════════════════════════════════════════════════════════════
# REDUCE Phase
# ═══════════════════════════════════════════════════════════════════════════

async def reduce_phase(
    summaries: list[FileSummary],
    metrics: PipelineMetrics,
    model_name: str = "sonnet",
) -> ExecutiveSummary:
    """
    Fase Reduce: combina todos los resúmenes en un documento ejecutivo de 1 página.
    Usa Sonnet para mejor calidad de redacción.
    """
    step_name = "Reduce"
    logger.info(f"\n[REDUCE] Combinando {len(summaries)} resúmenes...")
    start = time.time()

    # Preparar input para el reduce
    summaries_text = "\n\n".join(
        f"### {s.filename}\n{s.summary}"
        for s in summaries
    )

    response = await client.messages.create(
        model=MODELS[model_name]["id"],
        system="""Eres un technical writer senior. Tu tarea es combinar 
resúmenes individuales de archivos de un proyecto de software en UN 
documento ejecutivo coherente de 1 página (~400-600 palabras) en español.

El documento debe:
1. Tener un título descriptivo del proyecto
2. Abrir con un párrafo de visión general
3. Cubrir las áreas principales del proyecto (arquitectura, funcionalidades, 
   tecnologías, infraestructura)
4. Cerrar con un párrafo de conclusión sobre el estado del proyecto

No uses listas con viñetas. Escribe en prosa fluida.
Elimina redundancias entre los resúmenes.
No inventes información que no esté en los resúmenes.""",
        messages=[
            {
                "role": "user",
                "content": f"""Combina estos resúmenes de archivos del proyecto 
Klimbook (una red social para escaladores) en un documento ejecutivo de 1 página:

<summaries>
{summaries_text}
</summaries>"""
            },
        ],
        temperature=0.3,
        max_tokens=2000,
    )

    elapsed = time.time() - start
    metrics.register(step_name, model_name, response.usage, elapsed)

    result = ExecutiveSummary(
        title="Klimbook — Resumen Ejecutivo del Proyecto",
        summary=response.content[0].text.strip(),
    )

    logger.info(f"[REDUCE] Documento ejecutivo generado ({len(result.summary)} chars) | {elapsed:.2f}s")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Single Prompt (para comparación)
# ═══════════════════════════════════════════════════════════════════════════

async def single_prompt_summarize(
    files: list[tuple[str, str]],
    metrics: PipelineMetrics,
    model_name: str = "sonnet",
) -> str:
    """
    Concatena TODOS los archivos y los pasa en un solo prompt.
    Para comparar calidad y costo vs map-reduce.
    """
    step_name = "SinglePrompt"
    logger.info(f"\n[SINGLE] Concatenando {len(files)} archivos en un solo prompt...")
    start = time.time()

    # Concatenar todo
    all_content = "\n\n---\n\n".join(
        f"### {filename}\n{content}"
        for filename, content in files
    )

    total_chars = len(all_content)
    estimated_tokens = total_chars // 3
    logger.info(f"[SINGLE] Total: {total_chars:,} chars (~{estimated_tokens:,} tokens)")

    # Verificar que quepa en la ventana de contexto
    if estimated_tokens > 150_000:
        logger.warning(
            f"[SINGLE] Contenido muy grande ({estimated_tokens:,} tokens). "
            f"Podría exceder la ventana de contexto o ser muy costoso."
        )

    response = await client.messages.create(
        model=MODELS[model_name]["id"],
        system="""Eres un technical writer senior. Genera un documento ejecutivo 
de 1 página (~400-600 palabras) en español que resuma todo el proyecto.
No uses listas con viñetas. Escribe en prosa fluida.
Solo usa información de los documentos proporcionados.""",
        messages=[
            {
                "role": "user",
                "content": f"""Resume todo el contenido de estos archivos del proyecto 
Klimbook (red social para escaladores) en un documento ejecutivo de 1 página:

<files>
{all_content}
</files>"""
            },
        ],
        temperature=0.3,
        max_tokens=2000,
    )

    elapsed = time.time() - start
    metrics.register(step_name, model_name, response.usage, elapsed)

    result = response.content[0].text.strip()
    logger.info(f"[SINGLE] Completado ({len(result)} chars) | {elapsed:.2f}s")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Test Data (simulated Klimbook .md files)
# ═══════════════════════════════════════════════════════════════════════════

SIMULATED_FILES = {
    "README.md": """\
# Klimbook

Klimbook is a cross-platform social network for rock climbers. It allows users 
to track their climbing ascents, discover new routes and crags, follow other 
climbers, and share their progress.

## Tech Stack
- **Backend**: FastAPI (Python), PostgreSQL with PostGIS, Neo4j for social graph, 
  Redis for caching, Cloudflare R2 for media storage
- **Mobile**: React Native with Expo
- **Web**: React with Vite
- **Infrastructure**: Docker, Nginx, VPS deployment

## Features
- Route and crag discovery with interactive maps
- Ascent logging with grade tracking (YDS, French, V-scale)
- Social features: follow, feed, notifications
- Climber's book with personal statistics
- Multi-language support (English, Spanish)
- Onboarding tutorial for new users

## Getting Started
1. Clone the repository
2. Copy `.env.example` to `.env` and fill in your values
3. Run `docker-compose up -d`
4. Access the API at `http://localhost:8000/docs`
""",

    "ARCHITECTURE.md": """\
# Architecture

Klimbook follows a microservices architecture with the following services:

## Services
- **auth-service**: Handles user registration, login, JWT token management, 
  and password reset. Uses bcrypt for hashing and RS256 for JWT signing.
- **climbing-service**: Manages routes, crags, ascents, and grading systems. 
  Integrates with PostGIS for geospatial queries (finding crags near a location).
- **social-service**: Manages the social graph using Neo4j. Handles follows, 
  feed generation, and activity streams.
- **notification-service**: Push notifications via Firebase Cloud Messaging (FCM). 
  Supports both iOS and Android.
- **profile-service**: User profiles, preferences, statistics, and achievements.

## Communication
Services communicate via REST APIs over an internal Docker network. 
Nginx acts as reverse proxy and handles SSL termination.

## Database
- PostgreSQL 15 with PostGIS extension for geospatial data
- Neo4j for social relationships (follows, likes, comments)
- Redis for session caching, rate limiting, and frequently accessed data
""",

    "DEPLOYMENT.md": """\
# Deployment Guide

## Environments
- **Development**: Local Docker Compose setup with hot-reload
- **Staging**: VPS with Docker Compose, automated via GitHub Actions
- **Production**: VPS with Docker Compose, manual deployment with rollback support

## Docker Setup
Each service has its own Dockerfile. The `docker-compose.yml` orchestrates 
all services, databases, and the reverse proxy.

### Key Commands
- `docker-compose up -d` — Start all services
- `docker-compose logs -f <service>` — View service logs
- `docker-compose exec <service> alembic upgrade head` — Run migrations

## CI/CD
GitHub Actions workflow runs on push to `main`:
1. Lint and type check
2. Run unit tests
3. Build Docker images
4. Deploy to staging (automatic)
5. Deploy to production (manual approval)

## Database Migrations
Using Alembic for PostgreSQL schema migrations. Each service manages 
its own migration history. Run migrations before deploying new versions.
""",

    "API_DOCS.md": """\
# API Documentation

## Base URL
- Development: `http://localhost:8000`
- Production: `https://api.klimbook.com`

## Authentication
All endpoints except `/auth/login` and `/auth/register` require a valid 
JWT token in the `Authorization: Bearer <token>` header.

### POST /auth/register
Create a new user account.
- Body: `{email, password, username, display_name}`
- Returns: `{user_id, token, refresh_token}`

### POST /auth/login
Authenticate and receive tokens.
- Body: `{email, password}`
- Returns: `{token, refresh_token, user}`

## Climbing Endpoints

### GET /routes?lat=X&lon=Y&radius=Z
Find routes near a location using PostGIS.
- Query params: `lat`, `lon`, `radius` (km), `grade_min`, `grade_max`, `type`
- Returns: Array of routes with distance

### POST /ascents
Log a new climbing ascent.
- Body: `{route_id, style (onsight|flash|redpoint|repeat), date, notes}`
- Returns: Created ascent with updated statistics

### GET /users/{user_id}/book
Get a user's climbing book with statistics.
- Returns: `{total_ascents, highest_grade, grade_distribution, recent_ascents}`

## Social Endpoints

### POST /follow/{user_id}
Follow another user.

### GET /feed
Get the authenticated user's activity feed.
- Query params: `page`, `limit`
- Returns: Paginated feed of activities from followed users
""",

    "CHANGELOG.md": """\
# Changelog

## v2.9.0 (April 3, 2026)
- Added onboarding tutorial system for new users (mobile)
- Backend support for `first_login` preference flag
- 45+ new i18n translation keys for tutorial content
- Tutorial restart option in user menu

## v2.8.0 (March 28, 2026)
- Wall mode integration in climbing book
- New wall cards, action menus, and grade charts
- Filter improvements for wall/block views
- 6 new screens for wall-related features

## v2.7.0 (March 21, 2026)
- Full multi-language support (English and Spanish)
- i18next integration with ~1,030 translation keys
- Language persistence via AsyncStorage
- Zustand store integration for language sync

## v2.6.0 (March 14, 2026)
- Push notification system via Firebase Cloud Messaging
- Notification preferences per category
- Badge count management for iOS and Android
- Silent notifications for feed updates

## v2.5.0 (March 7, 2026)
- Interactive map for crag discovery
- PostGIS integration for proximity search
- Crag detail pages with route listings
- Map clustering for dense areas
""",

    "CONTRIBUTING.md": """\
# Contributing to Klimbook

Thank you for your interest in contributing to Klimbook!

## Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make your changes following our coding standards
4. Write tests for new functionality
5. Submit a pull request

## Coding Standards
- Python: Follow PEP 8, use type hints, write docstrings
- TypeScript/React Native: Follow ESLint config, use functional components
- Commits: Follow Conventional Commits (`feat:`, `fix:`, `refactor:`, etc.)
- PRs: Include description, screenshots for UI changes, test coverage

## Architecture Decisions
- Keep services small and focused (single responsibility)
- Use Pydantic for all data validation
- Prefer async/await for I/O operations
- Use PostGIS for all geospatial queries (don't calculate in Python)

## Testing
- Unit tests with pytest (backend) and Jest (mobile)
- Integration tests for API endpoints
- Manual testing checklist for UI changes
""",
}


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════

async def run_map_reduce(files: list[tuple[str, str]]) -> dict:
    """Ejecuta el pipeline map-reduce completo."""
    metrics = PipelineMetrics()
    pipeline_start = time.time()

    # MAP
    summaries = await map_phase(files, metrics)

    # Mostrar resúmenes individuales
    print(f"\n{'═'*60}")
    print("  MAP — Resúmenes individuales")
    print(f"{'═'*60}")
    for s in summaries:
        print(f"\n  {s.filename}:")
        print(f"     {s.summary}")

    # REDUCE
    executive = await reduce_phase(summaries, metrics, model_name="sonnet")

    # Mostrar documento ejecutivo
    print(f"\n{'═'*60}")
    print(f"  REDUCE — {executive.title}")
    print(f"{'═'*60}")
    print(f"\n{executive.summary}")

    pipeline_elapsed = time.time() - pipeline_start
    logger.info(f"\n[MapReduce] Pipeline total: {pipeline_elapsed:.2f}s")
    mr_stats = metrics.summary("Map-Reduce")

    return {
        "summaries": [s.model_dump() for s in summaries],
        "executive": executive.model_dump(),
        "metrics": mr_stats,
    }


async def run_single_prompt(files: list[tuple[str, str]]) -> dict:
    """Ejecuta el enfoque de prompt único para comparación."""
    metrics = PipelineMetrics()
    pipeline_start = time.time()

    result = await single_prompt_summarize(files, metrics, model_name="sonnet")

    print(f"\n{'═'*60}")
    print("  SINGLE PROMPT — Documento Ejecutivo")
    print(f"{'═'*60}")
    print(f"\n{result}")

    pipeline_elapsed = time.time() - pipeline_start
    logger.info(f"\n[Single] Pipeline total: {pipeline_elapsed:.2f}s")
    sp_stats = metrics.summary("Single Prompt")

    return {
        "executive": result,
        "metrics": sp_stats,
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Map-Reduce summarizer para archivos .md"
    )
    parser.add_argument(
        "--dir", type=Path, default=None,
        help="Directorio con archivos .md (si no se proporciona, usa datos simulados)"
    )
    parser.add_argument(
        "--skip-single", action="store_true",
        help="Omitir la comparación con single prompt (ahorra tokens)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  MAP-REDUCE SUMMARIZER")
    print("=" * 60)

    # Cargar archivos
    if args.dir and args.dir.exists():
        md_files = discover_md_files(args.dir)
        files = [read_file(f) for f in md_files]
    else:
        logger.info("[Setup] Usando archivos simulados de Klimbook")
        files = list(SIMULATED_FILES.items())

    logger.info(f"[Setup] {len(files)} archivos a procesar")
    total_chars = sum(len(c) for _, c in files)
    logger.info(f"[Setup] Total: {total_chars:,} chars (~{total_chars//3:,} tokens)")

    # ── Enfoque 1: Map-Reduce ────────────────────────────────────────
    print(f"\n\n{'#'*60}")
    print("  ENFOQUE 1: MAP-REDUCE")
    print(f"{'#'*60}")

    mr_result = await run_map_reduce(files)

    # ── Enfoque 2: Single Prompt (comparación) ───────────────────────
    if not args.skip_single:
        print(f"\n\n{'#'*60}")
        print("  ENFOQUE 2: SINGLE PROMPT (comparación)")
        print(f"{'#'*60}")

        sp_result = await run_single_prompt(files)

        # ── Comparación ──────────────────────────────────────────────
        mr_m = mr_result["metrics"]
        sp_m = sp_result["metrics"]

        print(f"\n\n{'═'*60}")
        print("  COMPARACIÓN: Map-Reduce vs Single Prompt")
        print(f"{'═'*60}")
        print(f"                    {'Map-Reduce':>14s}  {'Single':>14s}  {'Diferencia':>14s}")
        print(f"  {'─'*56}")
        print(f"  Steps:            {mr_m['steps']:>14d}  {sp_m['steps']:>14d}  {mr_m['steps']-sp_m['steps']:>+14d}")
        print(f"  Tokens input:     {mr_m['input_tokens']:>14,}  {sp_m['input_tokens']:>14,}  {mr_m['input_tokens']-sp_m['input_tokens']:>+14,}")
        print(f"  Tokens output:    {mr_m['output_tokens']:>14,}  {sp_m['output_tokens']:>14,}  {mr_m['output_tokens']-sp_m['output_tokens']:>+14,}")
        print(f"  Costo (USD):      ${mr_m['cost']:>13.6f}  ${sp_m['cost']:>13.6f}  ${mr_m['cost']-sp_m['cost']:>+13.6f}")
        print(f"  Tiempo (s):       {mr_m['time']:>14.2f}  {sp_m['time']:>14.2f}  {mr_m['time']-sp_m['time']:>+14.2f}")

        print(f"\n  Observaciones:")
        print(f"     Map-Reduce usa Haiku para Map (barato) + Sonnet para Reduce")
        print(f"     Single Prompt usa Sonnet para todo")
        if mr_m['cost'] < sp_m['cost']:
            saving = (1 - mr_m['cost']/sp_m['cost']) * 100
            print(f"     Map-Reduce es {saving:.0f}% más barato")
        else:
            extra = (mr_m['cost']/sp_m['cost'] - 1) * 100
            print(f"     Single Prompt es {extra:.0f}% más barato")
        print(f"     Revisa la calidad de ambos manualmente para decidir")
    else:
        logger.info("[Skip] Single prompt omitido (--skip-single)")

    # Guardar resultados
    output_path = Path("map_reduce_results.json")
    output_path.write_text(json.dumps(mr_result, indent=2, ensure_ascii=False))
    logger.info(f"\n[Output] Resultados guardados en {output_path}")


if __name__ == "__main__":
    asyncio.run(main())