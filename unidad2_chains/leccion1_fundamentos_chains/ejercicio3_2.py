"""
ejercicio3_2.py 
===========================
Ejercicio 3.2 — Chain con validación


Description:
-----------
Extender el ejercicio 3.1 con:
- Validación Pydantic entre cada paso
- Si el paso 1 devuelve JSON inválido: retry hasta 3 veces con temperature 
  decreciente (0.7 → 0.3 → 0.0)
- Si el paso 3 genera tweets >280 caracteres: retry pidiendo explícitamente 
  que acorte
- Implementar un decorator @retry_on_validation_error(max_retries=3) reutilizable
- Logging estructurado de cada paso: timestamp, step_name, tokens_in, tokens_out, 
  success/fail, retry_count


Metadata:
----------
* Author: zxxz6 
* Version: 1.0.0


History:
------------
Author      Date            Description
zxxz6       09/02/2026      Creation

"""

from anthropic import AsyncAnthropic
from pydantic import BaseModel, ValidationError, field_validator
import json
from pathlib import Path
import argparse
import asyncio
import logging
import time
import re
import functools

CACHE_DIR = Path(".pipeline_cache")
CACHE_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

client = AsyncAnthropic()


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════════════════════════════════════

class ExtractedPoints(BaseModel):
    """Validación del output del Paso 1."""
    component: str
    version: str
    key_points: list[str]

    @field_validator("key_points")
    @classmethod
    def must_have_five_points(cls, v):
        if len(v) != 5:
            raise ValueError(f"Se esperaban 5 puntos clave, se recibieron {len(v)}")
        return v

    @field_validator("key_points")
    @classmethod
    def points_not_empty(cls, v):
        for i, point in enumerate(v):
            if not point.strip():
                raise ValueError(f"El punto clave {i+1} está vacío")
        return v


class Summary(BaseModel):
    """Validación del output del Paso 2."""
    text: str

    @field_validator("text")
    @classmethod
    def must_have_two_paragraphs(cls, v):
        paragraphs = [p.strip() for p in v.strip().split("\n\n") if p.strip()]
        if len(paragraphs) < 2:
            raise ValueError(
                f"Se esperaban al menos 2 párrafos, se encontraron {len(paragraphs)}"
            )
        return v

    @field_validator("text")
    @classmethod
    def min_length(cls, v):
        if len(v) < 100:
            raise ValueError(f"Resumen muy corto: {len(v)} chars (mínimo 100)")
        return v


class Tweet(BaseModel):
    """Validación de un tweet individual."""
    text: str

    @field_validator("text")
    @classmethod
    def max_280_chars(cls, v):
        if len(v) > 280:
            raise ValueError(
                f"Tweet excede 280 chars: {len(v)} chars — '{v[:50]}...'"
            )
        return v

    @field_validator("text")
    @classmethod
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("Tweet vacío")
        return v


class TweetThread(BaseModel):
    """Validación del output del Paso 3."""
    tweets: list[Tweet]

    @field_validator("tweets")
    @classmethod
    def must_have_3_to_5(cls, v):
        if not (3 <= len(v) <= 5):
            raise ValueError(
                f"Se esperaban 3-5 tweets, se recibieron {len(v)}"
            )
        return v


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

tokens_acumulados = {"input": 0, "output": 0}
costos_acumulados = []


# ═══════════════════════════════════════════════════════════════════════════
# Metrics & Checkpoints
# ═══════════════════════════════════════════════════════════════════════════

def registrar_tokens(usage, step_name: str, model_name: str, retry_count: int = 0):
    """Registra tokens y costo de un step."""
    cfg = MODELS[model_name]
    inp = usage.input_tokens
    out = usage.output_tokens
    tokens_acumulados["input"] += inp
    tokens_acumulados["output"] += out
    costo = (inp / 1e6) * cfg["input"] + (out / 1e6) * cfg["output"]
    costos_acumulados.append(costo)

    retry_info = f" | retry={retry_count}" if retry_count > 0 else ""
    logger.info(
        f"[{step_name}] Tokens → in: {inp}, out: {out} | "
        f"Costo: ${costo:.6f}{retry_info} | "
        f"Acumulado → in: {tokens_acumulados['input']}, out: {tokens_acumulados['output']}"
    )


def resumen_costo_total():
    """Imprime el resumen final de tokens y costo."""
    total = sum(costos_acumulados)
    logger.info(
        f"[TOTAL] Tokens in: {tokens_acumulados['input']} | "
        f"Tokens out: {tokens_acumulados['output']} | "
        f"Steps: {len(costos_acumulados)} | "
        f"Costo total: ${total:.6f}"
    )


def save_checkpoint(step_name: str, data, run_id: str):
    """Guarda el output de un step en disco."""
    path = CACHE_DIR / f"{run_id}_{step_name}.json"
    path.write_text(json.dumps(data, default=str))


def load_checkpoint(step_name: str, run_id: str):
    """Carga un checkpoint previo si existe."""
    path = CACHE_DIR / f"{run_id}_{step_name}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Retry Decorator
# ═══════════════════════════════════════════════════════════════════════════

def retry_on_validation_error(max_retries: int = 3, temperatures: list[float] = None):
    """
    Decorator que reintenta una función async si falla con ValidationError
    o JSONDecodeError, bajando la temperature en cada intento.

    Args:
        max_retries: Número máximo de intentos
        temperatures: Lista de temperatures por intento. 
                      Default: [0.7, 0.3, 0.0]

    Usage:
        @retry_on_validation_error(max_retries=3)
        async def step_extract(..., temperature: float = 0.7):
            ...
    """
    if temperatures is None:
        temperatures = [0.7, 0.3, 0.0]

    # Asegurar que haya suficientes temperatures para los retries
    while len(temperatures) < max_retries:
        temperatures.append(0.0)

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            step_name = func.__name__
            last_error = None

            for attempt in range(max_retries):
                temp = temperatures[attempt]
                try:
                    logger.info(
                        f"[{step_name}] Intento {attempt + 1}/{max_retries} | "
                        f"temperature={temp}"
                    )
                    # Inyectar temperature en kwargs
                    kwargs["temperature"] = temp
                    kwargs["retry_count"] = attempt

                    result = await func(*args, **kwargs)

                    logger.info(
                        f"[{step_name}] OK en intento {attempt + 1}"
                    )
                    return result

                except (ValidationError, json.JSONDecodeError) as e:
                    last_error = e
                    logger.warning(
                        f"[{step_name}] FAIL Intento {attempt + 1} falló | "
                        f"Error: {type(e).__name__}: {str(e)[:200]}"
                    )

                    if attempt == max_retries - 1:
                        logger.error(
                            f"[{step_name}] FATAL Todos los intentos fallaron"
                        )
                        raise

            raise last_error

        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════
# Text Extraction (no LLM)
# ═══════════════════════════════════════════════════════════════════════════

def extract_raw_text(text: str, component: str, version: str) -> str | None:
    """Extrae notas de un componente/versión con regex."""
    step_name = "Raw text"
    logger.info(
        f"[{step_name}] Extrayendo notas para {component} {version} | "
        f"Longitud: {len(text)} chars"
    )
    pattern = rf"(#### {component} `{re.escape(version)}` — .+?)(?=\n#### |\Z)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_formated_text(file_path: Path, component: str, version: str) -> str | None:
    """Lee archivo y extrae notas de un componente/versión."""
    with open(file_path, "r") as f:
        input_text = f.read()

    raw_notes = extract_raw_text(input_text, component, version)

    if raw_notes:
        logger.info(
            f"[Extracción] Notas encontradas para {component} {version} | "
            f"Longitud: {len(raw_notes)} chars"
        )
    else:
        logger.warning(
            f"[Extracción] No se encontraron notas para {component} {version}"
        )

    return raw_notes


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Extracción → JSON validado con Pydantic
# ═══════════════════════════════════════════════════════════════════════════

@retry_on_validation_error(max_retries=3, temperatures=[0.7, 0.3, 0.0])
async def step_extract(
    file_path: Path,
    component: str,
    version: str,
    model_name: str = "haiku",
    temperature: float = 0.7,  # inyectado por el decorator
    retry_count: int = 0,      # inyectado por el decorator
) -> ExtractedPoints:
    """
    Paso 1 — Extrae puntos clave y valida con Pydantic.
    Si el JSON es inválido o no pasa validación, el decorator reintenta
    con temperature decreciente.
    """
    step_name = "Extracción LLM"
    start = time.time()

    SYSTEM = """Eres un ingeniero de software experto en documentación técnica. 
Extrae exactamente 5 puntos clave de las notas de versión.
Responde SOLO con JSON válido, sin texto adicional."""

    raw_notes = extract_formated_text(file_path, component, version)
    if not raw_notes:
        raise ValueError(f"No se encontraron notas para {component} {version}")

    PROMPT = f"""Extrae los 5 puntos clave de estas notas de versión.

<raw_notes>
{raw_notes}
</raw_notes>

Responde con JSON válido con esta estructura exacta:
{{
    "component": "{component}",
    "version": "{version}",
    "key_points": [
        "Punto clave 1",
        "Punto clave 2",
        "Punto clave 3",
        "Punto clave 4",
        "Punto clave 5"
    ]
}}"""

    response = await client.messages.create(
        model=MODELS[model_name]["id"],
        system=SYSTEM,
        messages=[
            {"role": "user", "content": PROMPT},
            {"role": "assistant", "content": "{"},
        ],
        temperature=temperature,
        max_tokens=1500,
    )

    registrar_tokens(response.usage, step_name, model_name, retry_count)
    elapsed = time.time() - start

    # Parse JSON (puede lanzar JSONDecodeError → decorator reintenta)
    raw_json = "{" + response.content[0].text
    parsed = json.loads(raw_json)

    # Validar con Pydantic (puede lanzar ValidationError → decorator reintenta)
    result = ExtractedPoints(**parsed)

    logger.info(
        f"[{step_name}] OK Completado en {elapsed:.2f}s | "
        f"5 puntos clave extraídos"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Resumen ejecutivo → validado con Pydantic
# ═══════════════════════════════════════════════════════════════════════════

@retry_on_validation_error(max_retries=3, temperatures=[0.3, 0.1, 0.0])
async def step_summarize(
    points: ExtractedPoints,
    model_name: str = "sonnet",
    temperature: float = 0.3,
    retry_count: int = 0,
) -> Summary:
    """
    Paso 2 — Genera resumen ejecutivo y valida que tenga 2+ párrafos.
    """
    step_name = "Resumen"
    start = time.time()

    SYSTEM = """Eres un ingeniero de software con excelentes habilidades de redacción.
Genera resúmenes ejecutivos claros y concisos en español."""

    PROMPT = f"""Genera un resumen ejecutivo de exactamente 2 párrafos en español 
a partir de estos puntos clave. 

Cada párrafo debe tener al menos 3 oraciones.
Separa los párrafos con una línea en blanco.

<key_points>
{json.dumps(points.model_dump(), indent=2, ensure_ascii=False)}
</key_points>"""

    response = await client.messages.create(
        model=MODELS[model_name]["id"],
        system=SYSTEM,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=temperature,
        max_tokens=1500,
    )

    registrar_tokens(response.usage, step_name, model_name, retry_count)
    elapsed = time.time() - start

    # Validar con Pydantic (puede lanzar ValidationError → decorator reintenta)
    result = Summary(text=response.content[0].text)

    logger.info(
        f"[{step_name}] OK Completado en {elapsed:.2f}s | "
        f"Longitud: {len(result.text)} chars"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Tweet thread → validado longitud y cantidad
# ═══════════════════════════════════════════════════════════════════════════

@retry_on_validation_error(max_retries=3, temperatures=[0.7, 0.5, 0.3])
async def step_tweet_thread(
    summary: Summary,
    model_name: str = "sonnet",
    temperature: float = 0.7,
    retry_count: int = 0,
) -> TweetThread:
    """
    Paso 3 — Genera tweet thread y valida que cada tweet ≤280 chars.
    Si algún tweet excede 280 chars, el decorator reintenta.
    """
    step_name = "Tweet Thread"
    start = time.time()

    SYSTEM = """Eres un community manager experto en redes sociales para 
proyectos de software. Generas tweets breves, informativos y con hashtags."""

    # Instrucción extra en retries para forzar tweets más cortos
    extra = ""
    if retry_count > 0:
        extra = """
IMPORTANTE: En un intento anterior, los tweets excedieron 280 caracteres.
CADA tweet DEBE tener MÁXIMO 280 caracteres incluyendo hashtags.
Sé más conciso. Usa abreviaciones si es necesario."""

    PROMPT = f"""Genera un tweet thread de 3-5 tweets a partir de este resumen.
{extra}

REGLAS ESTRICTAS:
- Cada tweet MÁXIMO 280 caracteres (incluyendo hashtags y espacios)
- Entre 3 y 5 tweets en total
- Incluir hashtags relevantes
- Formato: un tweet por línea, numerado

<summary>
{summary.text}
</summary>

Devuelve SOLO un JSON array de strings, cada string es un tweet:
["tweet 1...", "tweet 2...", "tweet 3..."]"""

    response = await client.messages.create(
        model=MODELS[model_name]["id"],
        system=SYSTEM,
        messages=[
            {"role": "user", "content": PROMPT},
            {"role": "assistant", "content": "["},
        ],
        temperature=temperature,
        max_tokens=1500,
    )

    registrar_tokens(response.usage, step_name, model_name, retry_count)
    elapsed = time.time() - start

    # Parse JSON array
    raw_json = "[" + response.content[0].text
    tweets_list = json.loads(raw_json)

    # Validar con Pydantic — cada tweet ≤280 chars, 3-5 tweets total
    tweets = [Tweet(text=t) for t in tweets_list]
    result = TweetThread(tweets=tweets)

    # Log de longitudes
    lengths = [len(t.text) for t in result.tweets]
    logger.info(
        f"[{step_name}] OK Completado en {elapsed:.2f}s | "
        f"{len(result.tweets)} tweets | "
        f"Longitudes: {lengths} (max: {max(lengths)})"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════

async def run_pipeline(input_file: Path, component: str, version: str):
    """Ejecuta la chain completa de 3 pasos con validación."""
    pipeline_start = time.time()
    run_id = f"{component}_{version}"
    logger.info(
        f"[Pipeline] Iniciando | "
        f"archivo={input_file} componente={component} versión={version}"
    )

    try:
        # ── Paso 1: Extracción (Haiku, barato) ───────────────────
        cached = load_checkpoint("extracción", run_id)
        if cached:
            logger.info("[Pipeline] Paso 1 cargado desde cache")
            points = ExtractedPoints(**json.loads(cached["bullet_points"]))
        else:
            points = await step_extract(input_file, component, version, model_name="haiku")
            save_checkpoint("extracción", {"bullet_points": points.model_dump_json()}, run_id)

        print(f"\n{'─'*60}")
        print(f"PASO 1 — Puntos Clave:")
        for i, p in enumerate(points.key_points, 1):
            print(f"   {i}. {p}")

        # ── Paso 2: Resumen (Sonnet, calidad) ─────────────────────
        cached = load_checkpoint("resumen", run_id)
        if cached:
            logger.info("[Pipeline] Paso 2 cargado desde cache")
            summary = Summary(text=cached["summary"])
        else:
            summary = await step_summarize(points, model_name="sonnet")
            save_checkpoint("resumen", {"summary": summary.text}, run_id)

        print(f"\n{'─'*60}")
        print(f"PASO 2 — Resumen Ejecutivo:")
        print(f"   {summary.text}")

        # ── Paso 3: Tweets (Sonnet, creativo) ─────────────────────
        tweet_thread = await step_tweet_thread(summary, model_name="sonnet")
        save_checkpoint("tweet_thread", {
            "tweets": [t.text for t in tweet_thread.tweets]
        }, run_id)

        print(f"\n{'─'*60}")
        print(f"PASO 3 — Tweet Thread ({len(tweet_thread.tweets)} tweets):")
        for i, tweet in enumerate(tweet_thread.tweets, 1):
            print(f"   [{i}] ({len(tweet.text)} chars) {tweet.text}")

        # ── Resumen final ─────────────────────────────────────────
        elapsed = time.time() - pipeline_start
        print(f"\n{'═'*60}")
        logger.info(f"[Pipeline] OK Completado en {elapsed:.2f}s")
        resumen_costo_total()

    except (ValidationError, json.JSONDecodeError) as e:
        elapsed = time.time() - pipeline_start
        logger.error(
            f"[Pipeline] FATAL Falló después de todos los retries | "
            f"Error: {type(e).__name__}: {e} | "
            f"Tiempo: {elapsed:.2f}s"
        )
        resumen_costo_total()
        raise

    except ValueError as e:
        logger.error(f"[Pipeline] Error de datos: {e}")
        raise


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chain de 3 pasos con validación Pydantic y retry"
    )
    parser.add_argument(
        "--component", required=True,
        choices=["backend", "frontend", "mobile"],
        help="Componente: backend, frontend, mobile",
    )
    parser.add_argument(
        "--version", required=True,
        help="Versión, ej: v2.9.0",
    )
    parser.add_argument(
        "--input_file", type=Path, required=True,
        help="Archivo de texto de entrada",
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Limpiar cache antes de ejecutar",
    )
    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: archivo no encontrado: {args.input_file}")
        exit(1)

    if args.clear_cache:
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()
        logger.info("[Cache] Limpiado")

    asyncio.run(run_pipeline(args.input_file, args.component, args.version))