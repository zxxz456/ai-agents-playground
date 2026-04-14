"""
ejercicio4_1.py 
===========================
Ejercicio 4.1 — Parallel chain multi-idioma


Description:
-----------
Input: changelog de Klimbook (output del ejercicio 2.1)
Ejecutar en paralelo con asyncio.gather: 
    (a) release notes en inglés, 
    (b) release notes en español, 
    (c) post corto para Ko-fi, 
    (d) resumen de 1 línea para commit message
Usar Semaphore para limitar a 3 requests concurrentes
Medir tiempo total vs tiempo secuencial estimado — debería ser ~3-4x más rápido
Manejar el caso donde una tarea falla pero las otras continúan (return_exceptions=True)


Metadata:
----------
* Author: zxxz6 
* Version: 1.1.0


History:
------------
Author      Date            Description
zxxz6       20/02/2026      Creation

"""

from anthropic import AsyncAnthropic
from pydantic import BaseModel, ValidationError, field_validator
from pathlib import Path
import logging
import sys
import subprocess
import argparse
import asyncio
import json
import time
import hashlib

# ─── Import ejercicio2_1 (changelog extractor) ───────────────────────────
# Los directorios tienen guiones (inválidos como nombres de paquete Python),
# así que agregamos la ruta al sys.path e importamos directamente el módulo.
_ejercicio2_1_dir = Path(__file__).resolve().parent.parent.parent / "unidad1-fundamentos" / "leccion2-prompting"
sys.path.insert(0, str(_ejercicio2_1_dir))
import ejercicio2_1
sys.path.pop(0)  # Limpiar sys.path después del import

# Re-exportar para uso directo
changelog_extractor = ejercicio2_1.changelog_extractor
entries_to_changelog_md = ejercicio2_1.entries_to_changelog_md
CommitEntry = ejercicio2_1.CommitEntry

CACHE_DIR = Path(".pipeline_cache")
CACHE_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

client = AsyncAnthropic()


# ── Modelos disponibles y costos (USD por millón de tokens) ──────────────
MODELS = {
    "sonnet": {
        "id": "claude-sonnet-4-20250514",
        "input": 3.00,
        "output": 15.00,
    },
    "haiku": {
        "id": "claude-haiku-3-5-20241022",
        "input": 0.80,
        "output": 4.00,
    },
}

# Semaphore: máximo 3 requests concurrentes a la API
API_SEMAPHORE = asyncio.Semaphore(3)


# ═══════════════════════════════════════════════════════════════════════════
# Token tracking y caching
# ═══════════════════════════════════════════════════════════════════════════

tokens_acumulados = {"input": 0, "output": 0}
costo_acumulado = 0.0
task_timings: dict[str, float] = {}


def registrar_tokens(usage, task_name: str, model_name: str):
    """Registra tokens de una respuesta y acumula el total."""
    global costo_acumulado
    cfg = MODELS[model_name]
    inp = usage.input_tokens
    out = usage.output_tokens
    tokens_acumulados["input"] += inp
    tokens_acumulados["output"] += out
    costo_step = (inp / 1_000_000) * cfg["input"] + (out / 1_000_000) * cfg["output"]
    costo_acumulado += costo_step
    logger.info(
        f"  [{task_name}] Tokens → in: {inp}, out: {out} ({model_name}) | "
        f"Costo: ${costo_step:.6f}"
    )


def resumen_costo_total():
    """Imprime el resumen final de tokens y costo acumulado real."""
    logger.info(
        f"[TOTAL] Tokens input: {tokens_acumulados['input']} | "
        f"Tokens output: {tokens_acumulados['output']} | "
        f"Costo total: ${costo_acumulado:.6f}"
    )


def cache_key(task_name: str, content: str) -> str:
    """Genera una key de cache basada en el nombre de tarea y contenido."""
    h = hashlib.md5(content.encode()).hexdigest()[:12]
    return f"{task_name}_{h}"


def save_cache(key: str, data: str):
    """Guarda resultado en disco."""
    path = CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps({"result": data}))
    logger.debug(f"  Cache guardado: {path.name}")


def load_cache(key: str) -> str | None:
    """Carga resultado de cache si existe."""
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        logger.info(f"  Cache hit: {path.name}")
        return json.loads(path.read_text())["result"]
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Git log y changelog (ejercicio 2.1)
# ═══════════════════════════════════════════════════════════════════════════

def get_git_log(version_anterior: str, version_actual: str) -> str:
    """Ejecuta git log --oneline entre dos versiones y devuelve el output."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", f"{version_anterior}..{version_actual}"],
            capture_output=True, text=True, check=True,
        )
        output = result.stdout.strip()
        if not output:
            raise ValueError(f"No se encontraron commits entre {version_anterior} y {version_actual}")
        return output
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error ejecutando git log: {e.stderr.strip()}") from e


def get_changelog(git_log: str, version: str = "latest") -> str:
    """Ejecuta el extractor del ejercicio 2.1 y devuelve el changelog en markdown."""
    entries = changelog_extractor(git_log, max_retries=3)
    if not entries:
        raise ValueError("No se pudo extraer el changelog del git log proporcionado.")
    return entries_to_changelog_md(entries, version=version)


# ═══════════════════════════════════════════════════════════════════════════
# Prompts por plataforma
# ═══════════════════════════════════════════════════════════════════════════

MD_FORMATTING = """\
# <product_name> — Release Notes <version>
**<release_date>**

<overall_summary_of_the_release>

---

## Release Notes

### <feature_or_fix_title>
<detailed_description>

* **<key_change_1>:** <description>
* **<key_change_2>:** <description>

---

## Installation
<installation_instructions>
"""

MD_EXAMPLE = """\
# Klimbook Web — Release Notes v2.9.0
**April 3, 2026**

We are pleased to announce the release of Klimbook Web version 2.9.0. \
This release adds backend support for the onboarding tutorial system on mobile.

---

## Release Notes

### Onboarding Tutorial Support

Backend support for triggering and tracking the onboarding tutorial on first login.

* **New `first_login` Preference Field:** Added `first_login: True` to user preferences defaults.
* **Schema Updates:** `PreferencesResponse` exposes `first_login` in GET responses.
* **No Migration Needed:** `first_login` is stored inside the existing `User.preferences` JSONB column.

### Test Coverage

* **2 New Tests:** `test_get_default_first_login` and `test_update_first_login`.
* **2 Updated Tests:** `test_get_default_preferences` and `test_empty_patch_preserves_defaults` updated.

---

## Installation

This version is available on the web platform. No installation required.
"""

KOFI_EXAMPLE = """\
Weekly Update

Hey everyone!
This week I focused on something I had been putting off — what happens when \
someone opens Klimbook for the first time and has no idea where anything is... \
In order to solve this I built a full onboarding tutorial. When you log in for \
the first time, a guided tour shows you around (search, notifications, your book, \
how to visit a venue, browse walls, report missing blocks, all of it).
This is just for the mobile version of Klimbook, and it is available now.
Cya.

— zxxz6.
"""


def build_platform_prompts(changelog_md: str) -> dict[str, dict]:
    """Construye los prompts para cada plataforma/tarea.

    Returns:
        Dict con key=nombre_tarea, value=dict con 'system' y 'user' y 'max_tokens'.
    """
    base_release_notes = f"""\
Con base en las siguientes notas de actualización:
<release_notes>
{changelog_md}
</release_notes>"""

    return {
        "github_en": {
            "model": "sonnet",
            "system": "Eres un Community Manager experto en redactar release notes técnicos para GitHub. Escribe TODO en inglés.",
            "user": f"""{base_release_notes}
Genera release notes para GitHub en markdown con este formato:
<format>
{MD_FORMATTING}
</format>
Ejemplo de referencia:
<example>
{MD_EXAMPLE}
</example>
Genera todo el contenido en inglés.""",
            "max_tokens": 1500,
        },
        "github_es": {
            "model": "sonnet",
            "system": "Eres un Community Manager experto en redactar release notes técnicos para GitHub. Escribe TODO en español.",
            "user": f"""{base_release_notes}
Genera release notes para GitHub en markdown con este formato:
<format>
{MD_FORMATTING}
</format>
Ejemplo de referencia (adapta a español):
<example>
{MD_EXAMPLE}
</example>
Genera todo el contenido en español.""",
            "max_tokens": 1500,
        },
        "kofi": {
            "model": "haiku",
            "system": "Eres un community manager indie que escribe posts personales y agradecidos para supporters en Ko-fi.",
            "user": f"""{base_release_notes}
Genera un post de actualización para Ko-fi. Tono personal y agradecido.
Formato: TÍTULO, CONTENIDO, FIRMA (— zxxz6.)
Ejemplo de referencia:
<example>
{KOFI_EXAMPLE}
</example>""",
            "max_tokens": 800,
        },
        "commit_msg": {
            "model": "haiku",
            "system": "Eres un desarrollador senior. Genera mensajes de commit concisos siguiendo Conventional Commits.",
            "user": f"""{base_release_notes}
Genera UN resumen de exactamente 1 línea para usar como mensaje de commit.
Formato: 'release: <resumen conciso de los cambios principales>'
Máximo 72 caracteres. Solo la línea, sin explicación.""",
            "max_tokens": 100,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline paralelo
# ═══════════════════════════════════════════════════════════════════════════

async def generate_for_task(
    task_name: str,
    system: str,
    user: str,
    max_tokens: int,
    model_name: str,
    changelog_md: str,
) -> tuple[str, str]:
    """Genera contenido para una tarea específica con semaphore, cache, y logging.

    Returns:
        Tupla (task_name, resultado).
    """
    # Check cache
    key = cache_key(task_name, changelog_md)
    cached = load_cache(key)
    if cached is not None:
        task_timings[task_name] = 0.0
        return task_name, cached

    # Ejecutar con semaphore (máx 3 concurrentes)
    async with API_SEMAPHORE:
        logger.info(f"  [{task_name}] Iniciando request...")
        start = time.time()
        response = await client.messages.create(
            model=MODELS[model_name]["id"],
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=0.5,
            max_tokens=max_tokens,
        )
        elapsed = time.time() - start

    result = response.content[0].text
    task_timings[task_name] = elapsed

    # Registrar tokens
    registrar_tokens(response.usage, task_name, model_name)
    logger.info(f"  [{task_name}]  Completado en {elapsed:.1f}s ({len(result)} chars)")

    # Guardar cache
    save_cache(key, result)

    return task_name, result


async def run_parallel_pipeline(changelog_md: str) -> dict[str, str | None]:
    """Ejecuta las 4 tareas en paralelo con asyncio.gather.

    - Semaphore limita a 3 concurrentes
    - return_exceptions=True para que un fallo no cancele el resto
    - Mide tiempo paralelo vs estimado secuencial
    """
    prompts = build_platform_prompts(changelog_md)

    logger.info(f"[Pipeline] Lanzando {len(prompts)} tareas en paralelo (máx 3 concurrentes)...")
    total_start = time.time()

    # Crear tareas
    tasks = [
        generate_for_task(
            task_name=name,
            system=cfg["system"],
            user=cfg["user"],
            max_tokens=cfg["max_tokens"],
            model_name=cfg["model"],
            changelog_md=changelog_md,
        )
        for name, cfg in prompts.items()
    ]

    # Ejecutar en paralelo — return_exceptions=True para resiliencia
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    total_elapsed = time.time() - total_start

    # Procesar resultados
    results: dict[str, str | None] = {}
    successes = 0
    failures = 0

    for raw in raw_results:
        if isinstance(raw, Exception):
            failures += 1
            logger.error(f"   Tarea falló: {type(raw).__name__}: {raw}")
        else:
            name, content = raw
            results[name] = content
            successes += 1

    # ─── Resumen de tiempos ───────────────────────────────────────────
    sequential_estimate = sum(task_timings.values())
    speedup = sequential_estimate / total_elapsed if total_elapsed > 0 else 0

    logger.info(f"{'═' * 60}")
    logger.info(f"[Pipeline] Resumen de ejecución:")
    logger.info(f"  Tareas: {successes}  {failures} ")
    logger.info(f"  Tiempo paralelo:             {total_elapsed:.1f}s")
    logger.info(f"  Tiempo secuencial estimado:  {sequential_estimate:.1f}s")
    logger.info(f"  Speedup:                     {speedup:.1f}x")
    for name, t in task_timings.items():
        logger.info(f"    {name:15s} → {t:.1f}s")
    logger.info(f"{'═' * 60}")

    # Resumen de tokens y costo
    resumen_costo_total()

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(
        description="Pipeline paralelo: genera release notes multi-plataforma a partir de git log.",
    )
    parser.add_argument("version_anterior", help="Tag o commit de inicio (e.g. v2.7.0)")
    parser.add_argument("version_actual", help="Tag o commit de fin (e.g. v2.8.0)")
    parser.add_argument("--clear-cache", action="store_true", help="Limpiar cache antes de ejecutar")
    args = parser.parse_args()

    if args.clear_cache:
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()
        logger.info("[Cache] Cache limpiado")

    # Paso 1: Obtener git log
    logger.info(f"[Git] Obteniendo log: {args.version_anterior}..{args.version_actual}")
    git_log_output = get_git_log(args.version_anterior, args.version_actual)
    commit_count = len(git_log_output.strip().splitlines())
    logger.info(f"[Git] {commit_count} commits encontrados")

    # Paso 2: Extraer changelog estructurado (ejercicio 2.1)
    logger.info("[Changelog] Extrayendo changelog con ejercicio 2.1...")
    changelog_md = get_changelog(git_log_output, version=args.version_actual)
    logger.info(f"[Changelog] Changelog generado ({len(changelog_md)} chars)")

    # Paso 3: Pipeline paralelo
    results = await run_parallel_pipeline(changelog_md)

    # Paso 4: Mostrar resultados
    for name, content in results.items():
        print(f"\n{'═' * 60}")
        print(f" {name.upper()}")
        print(f"{'═' * 60}")
        print(content)

    # Guardar todos los resultados en un archivo consolidado
    output_path = CACHE_DIR / f"release_{args.version_actual}.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info(f"[Output] Resultados guardados en {output_path}")


if __name__ == "__main__":
    asyncio.run(main())