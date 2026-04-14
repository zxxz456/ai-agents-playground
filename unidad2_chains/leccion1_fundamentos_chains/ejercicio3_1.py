"""
ejercicio3_1.py 
===========================
Ejercicio 3.1 — Chain de 3 pasos (Python puro)


Description:
-----------
Paso 1 (Extracción): Recibe un texto largo (ej. README de Klimbook) y extrae 
                     los 5 puntos clave en JSON
Paso 2 (Transformación): Toma los puntos clave y genera un resumen ejecutivo de 
                         2 párrafos en español
Paso 3 (Formato): Toma el resumen y genera un tweet thread de 3-5 tweets con 
                          hashtags relevantes

Implementar como funciones async: 
 async def step_extract(text) -> dict, async def step_summarize(points) -> str, etc.
Medir tokens y costo acumulado de todo el chain


Metadata:
----------
* Author: zxxz6 
* Version: 1.1.0


History:
------------
Author      Date            Description
zxxz6       09/02/2026      Creation

"""

from anthropic import AsyncAnthropic
import json
from pathlib import Path
import argparse
import asyncio
import logging
import time
import re

CACHE_DIR = Path(".pipeline_cache")
CACHE_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

client = AsyncAnthropic()

# ── Modelos disponibles y costos (USD por millón de tokens) ──────────
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

tokens_acumulados = {"input": 0, "output": 0}

def registrar_tokens(usage, step_name: str, model_name: str):
    """Registra tokens de una respuesta y acumula el total."""
    cfg = MODELS[model_name]
    inp = usage.input_tokens
    out = usage.output_tokens
    tokens_acumulados["input"] += inp
    tokens_acumulados["output"] += out
    costo_step = (inp / 1_000_000) * cfg["input"] + (out / 1_000_000) * cfg["output"]
    logger.info(
        f"[{step_name}] Tokens → input: {inp}, output: {out} | "
        f"Costo step: ${costo_step:.6f} | "
        f"Acumulado → input: {tokens_acumulados['input']}, output: {tokens_acumulados['output']}"
    )

def resumen_costo_total(model_name: str):
    """Imprime el resumen final de tokens y costo."""
    cfg = MODELS[model_name]
    total = (
        (tokens_acumulados["input"] / 1_000_000) * cfg["input"]
        + (tokens_acumulados["output"] / 1_000_000) * cfg["output"]
    )
    logger.info(
        f"[TOTAL] Tokens input: {tokens_acumulados['input']} | "
        f"Tokens output: {tokens_acumulados['output']} | "
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

def extract_raw_text(text, component, version):
    """
    Extrae las notas de un componente específico y versión del texto proporcionado.

    Args:
        text (str): Texto completo del que se extraerán las notas.
        component (str): Nombre del componente.
        version (str): Versión del componente.

    Returns:
        str: Notas extraídas del componente y versión especificados, o None si no se encuentran.

    """
    # Construir el patrón: "#### Component `version` — ..." hasta el siguiente "####" o fin
    step_name = "Raw text"
    logger.info(f"[{step_name}] Extrayendo notas para {component} {version} | Longitud del texto: {len(text)} caracteres")
    pattern = rf"(#### {component} `{re.escape(version)}` — .+?)(?=\n#### |\Z)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_formated_text(file_path, component, version):
    """
    Lee un archivo y extrae las notas de un componente específico y versión.

    Args:
        file_path (str): Ruta al archivo que contiene el texto.
        component (str): Nombre del componente.
        version (str): Versión del componente.

    Returns:
        str: Notas extraídas del componente y versión especificados, o None si no se encuentran.
    """
    with open(file_path, "r") as f:
        input_text = f.read()
        raw_notes = extract_raw_text(input_text, component, version)
        step_name = "Extracción"
        
        if raw_notes:
            logger.info(f"[{step_name}] Notas extraídas para {component} {version} | Longitud: {len(raw_notes) if raw_notes else 0} caracteres")
            print(f"Notas extraídas para {component} {version}:\n{raw_notes}")
        else:
            logger.info(f"[{step_name}] No se encontraron notas para {component} {version}.")
            print(f"No se encontraron notas para {component} {version}.")
            return None
        f.close()
    return raw_notes
    
async def step_extract(file_path, component, version, model_name: str = "sonnet") -> dict:
    """
    Paso 1 — Genera los puntos clave a partir de las notas extraídas usando un LLM.
    """
    step_name = "Extracción LLM"
    logger.info(f"[{step_name}] Iniciando para {component} {version} | modelo={model_name}")
    start = time.time()

    SYSTEM = """
    Eres un ingeniero de software experto en documentacion técnica. 
    Usa lenguaje claro y conciso para extraer los puntos clave de las notas de versión.
    Extrae exactamente 5 puntos clave, cada uno con un título y una breve descripción.
    """

    TASK = """ 
        Tu tarea es extraer los puntos clave de las notas de versión proporcionadas, 
        enfocándote en:
        - Cambios importantes
        - Nuevas funcionalidades
        - Mejoras de rendimiento
        - Correcciones de bugs
        - Cambios de API
        - Cualquier otro detalle relevante para desarrolladores.
        Devuelve los puntos clave en formato JSON con la siguiente estructura:
    {
        "component": "nombre del componente",
        "version": "versión del componente",
        "key_points": [
            "Punto clave 1",
            "Punto clave 2",
            "Punto clave 3",
            "Punto clave 4",
            "Punto clave 5"
        ]
    }
    """
    raw_notes = extract_formated_text(file_path, component, version)
    if not raw_notes:
        return None
    PROMPT = f"""
    {TASK}
    Aquí están las notas de versión para {component} {version}:
    <raw_notes>
    {raw_notes}
    </raw_notes>
    """
    response = await client.messages.create(
        model=MODELS[model_name]["id"],
        system=SYSTEM,
        messages=[
            {"role": "user", "content": PROMPT },
            {"role": "assistant", "content": "{"},
        ],
        temperature=0.3,
        max_tokens=1500
    )
    registrar_tokens(response.usage, step_name, model_name)
    elapsed = time.time() - start
    logger.info(f"[{step_name}] Completado en {elapsed:.2f}s")
    return "{" + response.content[0].text
    
async def step_summarize(points, model_name: str = "sonnet") -> str:
    """
    Paso 2 — Genera un resumen ejecutivo a partir de los puntos clave.

    Args:
        points (dict): Diccionario con los puntos clave extraídos.

    Returns:
        str: Resumen ejecutivo generado.
    """
    step_name = "Resumen"
    logger.info(f"[{step_name}] Iniciando generación de resumen ejecutivo | modelo={model_name}")
    start = time.time()

    SYSTEM = """
    Eres un ingeniero de software experto en documentacion técnica. 
    Tu area principal es la ingenieria de software, pero tienes habilidades de redacción y comunicación.
    """

    TASK = """
    Tu tarea es generar un resumen ejecutivo de 2 párrafos en español a partir de los puntos clave proporcionados. 
    El resumen debe ser claro, conciso y destacar los aspectos más importantes de las notas de versión, enfocándose en:
    - Cambios importantes
    - Nuevas funcionalidades
    - Mejoras de rendimiento
    - Correcciones de bugs
    - Cambios de API
    - Cualquier otro detalle relevante para desarrolladores.
    """

    PROMPT = f"""
    {TASK}
    Aquí están los puntos clave extraídos:
    <key_points>
    {json.dumps(points, indent=2, ensure_ascii=False)}
    </key_points>
    """

    response = await client.messages.create(
        model=MODELS[model_name]["id"],
        system=SYSTEM,
        messages=[
            {"role": "user", "content": PROMPT },
        ],
        temperature=0.3,
        max_tokens=1500
    )
    registrar_tokens(response.usage, step_name, model_name)
    elapsed = time.time() - start
    logger.info(f"[{step_name}] Completado en {elapsed:.2f}s")
    return response.content[0].text

async def step_tweet_thread(summary, model_name: str = "sonnet") -> str:
    """
    Paso 3 — Genera un tweet thread a partir del resumen ejecutivo.

    Args:
        summary (str): Resumen ejecutivo generado.
    Returns:
        str: Tweet thread generado.
    """
    step_name = "Tweet Thread"
    logger.info(f"[{step_name}] Iniciando generación de tweet thread | modelo={model_name}")
    start = time.time()

    SYSTEM = """
    Eres un community manager experto en redes sociales para proyectos de software. 
    Tu objetivo es comunicar de manera efectiva las novedades de un proyecto a través de Twitter, utilizando un lenguaje claro, conciso y atractivo para la comunidad de desarrolladores.
    """

    TASK = """
    Tu tarea es generar un tweet thread de 3-5 tweets a partir del resumen ejecutivo proporcionado. 
    Los tweets deben ser breves, informativos y contener hashtags relevantes para maximizar su alcance entre la comunidad de desarrolladores interesados en el proyecto.
    Enfócate en destacar los aspectos más importantes del resumen ejecutivo, como:
    - Cambios importantes
    - Nuevas funcionalidades
    - Mejoras de rendimiento
    - Correcciones de bugs
    - Cambios de API
    - Cualquier otro detalle relevante para desarrolladores.

    DEvuelve el thread en formato de lista de tweets, por ejemplo:
[
    "Tweet 1: ... #hashtag1 #hashtag2",
    "Tweet 2: ... #hashtag3 #hashtag4",
    "Tweet 3: ... #hashtag5 #hashtag6"
]   
    """

    PROMPT = f"""
    {TASK}
    Aquí está el resumen ejecutivo:
    <summary>
    {summary}
    </summary>
    """

    response = await client.messages.create(
        model=MODELS[model_name]["id"],
        system=SYSTEM,
        messages=[
            {"role": "user", "content": PROMPT },
        ],
        temperature=0.9,
        max_tokens=1500
    )
    registrar_tokens(response.usage, step_name, model_name)
    elapsed = time.time() - start
    logger.info(f"[{step_name}] Completado en {elapsed:.2f}s")
    return response.content[0].text


async def run_pipeline(input_file: Path, component: str, version: str, model_name: str = "sonnet"):
    """Ejecuta la chain completa de 3 pasos."""
    cfg = MODELS[model_name]
    pipeline_start = time.time()
    run_id = f"{component}_{version}"
    logger.info(
        f"[Pipeline] Iniciando | modelo={model_name} ({cfg['id']}) "
        f"archivo={input_file} componente={component} versión={version}"
    )

    # Paso 1: Extracción → puntos clave (JSON)
    bullet_points = await step_extract(input_file, component, version, model_name)
    if not bullet_points:
        logger.error("[Pipeline] Paso 1 falló — no se obtuvieron puntos clave")
        return
    save_checkpoint("extracción", {"bullet_points": bullet_points}, run_id)

    # Paso 2: Resumen ejecutivo
    summary = await step_summarize(bullet_points, model_name)
    save_checkpoint("resumen", {"summary": summary}, run_id)

    # Paso 3: Tweet thread
    tweet_thread = await step_tweet_thread(summary, model_name)
    save_checkpoint("tweet_thread", {"tweet_thread": tweet_thread}, run_id)

    elapsed = time.time() - pipeline_start
    logger.info(f"[Pipeline] Completado en {elapsed:.2f}s")
    resumen_costo_total(model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejemplo de uso de la chain completa")
    parser.add_argument('--component', required=True, 
                        choices=['backend', 'frontend', 'mobile'],
                        help='Componente: backend, frontend, mobile')
    parser.add_argument('--version', required=True, 
                        help='Versión, ej: v2.9.0')
    parser.add_argument("--input_file", type=Path, help="Archivo de texto de entrada")
    args = parser.parse_args()

    if not args.input_file or not args.input_file.exists():
        print("Por favor, proporciona un archivo de texto válido con --input_file")
        exit(1)

    asyncio.run(run_pipeline(args.input_file, args.component, args.version, model_name="sonnet"))