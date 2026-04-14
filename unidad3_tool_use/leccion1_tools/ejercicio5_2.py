"""
ejercicio5_2.py 
===========================
Ejercicio 5.2 — Multi-tool agent


Description:
-----------
- 3 tools: get_weather(city, units), get_time(timezone), calculate(expression)
- get_weather: llama a Open-Meteo API (gratis, no requiere key): api.open-meteo.com
- get_time: usa datetime + zoneinfo para devolver hora actual en la zona solicitada
- El agente debe resolver preguntas compuestas: "¿Qué hora es en Tokyo y qué temperatura hace?"
- Implementar el loop completo: enviar → detectar tool_use → ejecutar → devolver resultado → repetir si necesario


Metadata:
----------
* Author: zxxz6 
* Version: 1.1.0


History:
------------
Author      Date            Description
zxxz6       22/02/2026      Creation

"""

from anthropic import Anthropic
import json
import time
import logging
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

from unidad3_tool_use.leccion1_tools.ejercicio5_1 import calculate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("multi-tool-agent")

client = Anthropic()

MODEL = "claude-sonnet-4-20250514"


# ═══════════════════════════════════════════════════════════════════════════
# Pricing
# ═══════════════════════════════════════════════════════════════════════════

PRICING = {"input": 3.00, "output": 15.00}  # USD por millón de tokens (Sonnet)

tokens_acumulados = {"input": 0, "output": 0}
costos_acumulados = []


def registrar_tokens(usage, step_name: str):
    """Registra tokens y acumula costo."""
    inp, out = usage.input_tokens, usage.output_tokens
    tokens_acumulados["input"] += inp
    tokens_acumulados["output"] += out
    costo = (inp / 1e6) * PRICING["input"] + (out / 1e6) * PRICING["output"]
    costos_acumulados.append(costo)
    logger.info(
        f"  [{step_name}] Tokens → in: {inp}, out: {out} | ${costo:.6f}"
    )


def resumen_costo_total():
    """Imprime resumen final."""
    total = sum(costos_acumulados)
    logger.info(
        f"[TOTAL] Tokens in: {tokens_acumulados['input']} | "
        f"out: {tokens_acumulados['output']} | "
        f"Costo: ${total:.6f}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# City Database
# ═══════════════════════════════════════════════════════════════════════════

CITIES = {
    "tokyo": (35.6762, 139.6503, "Asia/Tokyo"),
    "new york": (40.7128, -74.0060, "America/New_York"),
    "london": (51.5074, -0.1278, "Europe/London"),
    "paris": (48.8566, 2.3522, "Europe/Paris"),
    "berlin": (52.5200, 13.4050, "Europe/Berlin"),
    "madrid": (40.4168, -3.7038, "Europe/Madrid"),
    "mexico city": (19.4326, -99.1332, "America/Mexico_City"),
    "puebla": (19.0414, -98.2063, "America/Mexico_City"),
    "buenos aires": (-34.6037, -58.3816, "America/Argentina/Buenos_Aires"),
    "sydney": (-33.8688, 151.2093, "Australia/Sydney"),
    "beijing": (39.9042, 116.4074, "Asia/Shanghai"),
    "mumbai": (19.0760, 72.8777, "Asia/Kolkata"),
    "cairo": (30.0444, 31.2357, "Africa/Cairo"),
    "los angeles": (34.0522, -118.2437, "America/Los_Angeles"),
    "santiago": (-33.4489, -70.6693, "America/Santiago"),
    "bogota": (4.7110, -74.0721, "America/Bogota"),
    "lima": (-12.0464, -77.0428, "America/Lima"),
    "sao paulo": (-23.5505, -46.6333, "America/Sao_Paulo"),
    "rome": (41.9028, 12.4964, "Europe/Rome"),
    "seoul": (37.5665, 126.9780, "Asia/Seoul"),
    "bangkok": (13.7563, 100.5018, "Asia/Bangkok"),
}


# ═══════════════════════════════════════════════════════════════════════════
# Tool Functions
# ═══════════════════════════════════════════════════════════════════════════

def get_weather(city: str, units: str = "celsius") -> dict:
    """
    Obtiene el clima actual de una ciudad usando la API de Open-Meteo.

    Args:
        city: Nombre de la ciudad
        units: "celsius" o "fahrenheit"

    Returns:
        Dict con datos del clima o error
    """
    logger.info(f"  [Weather] Consultando clima para: '{city}' (units={units})")

    city_lower = city.lower().strip()
    city_data = CITIES.get(city_lower)
    coords = (city_data[0], city_data[1]) if city_data else None

    if not coords:
        try:
            geo_resp = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "en"},
                timeout=10,
            )
            geo_resp.raise_for_status()
            geo_data = geo_resp.json()

            if not geo_data.get("results"):
                return {"error": f"Ciudad no encontrada: {city}"}

            lat = geo_data["results"][0]["latitude"]
            lon = geo_data["results"][0]["longitude"]
            coords = (lat, lon)
        except requests.RequestException as e:
            return {"error": f"Error en geocoding: {e}"}

    lat, lon = coords
    temp_unit = "fahrenheit" if units.lower() == "fahrenheit" else "celsius"

    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "temperature_unit": temp_unit,
                "wind_speed_unit": "kmh",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return {"error": f"Error consultando clima: {e}"}

    current = data.get("current", {})
    unit_symbol = "°F" if temp_unit == "fahrenheit" else "°C"

    result = {
        "city": city,
        "temperature": current.get("temperature_2m"),
        "unit": unit_symbol,
        "humidity_percent": current.get("relative_humidity_2m"),
        "wind_speed_kmh": current.get("wind_speed_10m"),
        "weather_code": current.get("weather_code"),
    }
    logger.info(f"  [Weather] → {result['temperature']}{unit_symbol}, humidity {result['humidity_percent']}%")
    return result


def get_time(timezone: str) -> dict:
    """
    Devuelve la hora actual en la zona horaria solicitada.

    Args:
        timezone: Nombre de zona horaria IANA (ej. "America/New_York")
                  o nombre de ciudad (ej. "Tokyo")

    Returns:
        Dict con la hora actual o error
    """
    logger.info(f"  [Time] Consultando hora para: '{timezone}'")

    city_data = CITIES.get(timezone.lower().strip())
    tz_name = city_data[2] if city_data else timezone

    try:
        tz = ZoneInfo(tz_name)
        now = datetime.now(tz)
        result = {
            "timezone": tz_name,
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day_of_week": now.strftime("%A"),
        }
        logger.info(f"  [Time] → {result['time']} ({tz_name})")
        return result
    except KeyError:
        return {"error": f"Zona horaria no válida: {timezone}"}


# ═══════════════════════════════════════════════════════════════════════════
# Tool Definitions
# ═══════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "calculate",
        "description": (
            "Performs mathematical calculations safely. Use this tool whenever "
            "you need to compute arithmetic operations like addition, subtraction, "
            "multiplication, division, modulo, or exponentiation. "
            "Supports: numbers, +, -, *, /, //, %, **, and parentheses. "
            "Does NOT support variables or functions like sqrt(), sin(), etc. "
            "For percentages, convert them first: 23% of 150 → 150 * 23 / 100. "
            "Example: calculate(expression='150 * 23 / 100', precision=0)"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "Mathematical expression to evaluate. "
                        "Use standard operators: +, -, *, /, //, %, **. "
                        "Examples: '150 * 0.23', '(100 - 23) / 4', '2 ** 10'"
                    ),
                },
                "precision": {
                    "type": "integer",
                    "description": (
                        "Number of decimal places in the result (default: 2). "
                        "Use 0 for whole numbers."
                    ),
                    "default": 2,
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_weather",
        "description": (
            "Gets current weather for a city using Open-Meteo API. "
            "Use when the user asks about weather, temperature, climate, "
            "humidity, or wind conditions in any city. "
            "Example: get_weather(city='Puebla', units='celsius')"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name (e.g. 'Tokyo', 'Puebla', 'New York')",
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units (default: celsius)",
                },
            },
            "required": ["city"],
        },
    },
    {
        "name": "get_time",
        "description": (
            "Gets current date and time in a specific timezone or city. "
            "Use when the user asks about the current time or date in any location. "
            "Accepts IANA timezone names (e.g. 'Asia/Tokyo') or city names (e.g. 'Tokyo'). "
            "Example: get_time(timezone='Tokyo')"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": (
                        "IANA timezone (e.g. 'America/New_York', 'Asia/Tokyo') "
                        "or city name (e.g. 'Tokyo', 'London')"
                    ),
                },
            },
            "required": ["timezone"],
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Tool Executor
# ═══════════════════════════════════════════════════════════════════════════

TOOL_FUNCTIONS = {
    "calculate": calculate,
    "get_weather": get_weather,
    "get_time": get_time,
}

tool_call_counts = {}
MAX_CALLS_PER_CONVERSATION = 15


def execute_tool(name: str, input_data: dict) -> str:
    """Ejecuta una tool de forma segura con rate limiting."""
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return json.dumps({"error": f"Tool desconocida: {name}"})

    count = tool_call_counts.get(name, 0)
    if count >= MAX_CALLS_PER_CONVERSATION:
        return json.dumps({
            "error": f"Rate limit: máximo {MAX_CALLS_PER_CONVERSATION} llamadas a '{name}'"
        })
    tool_call_counts[name] = count + 1

    try:
        result = func(**input_data)
        return json.dumps(result)
    except TypeError as e:
        return json.dumps({"error": f"Parámetros inválidos: {e}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


# ═══════════════════════════════════════════════════════════════════════════
# Agent: Single Question (handles full tool loop)
# ═══════════════════════════════════════════════════════════════════════════

def ask(question: str) -> str:
    """
    Envía una pregunta a Claude con las 3 tools.
    Maneja el loop de tool_use incluyendo parallel tool calls.

    Args:
        question: Pregunta del usuario

    Returns:
        Respuesta final de Claude
    """
    logger.info(f"\n{'═'*60}")
    logger.info(f"[Agent] Pregunta: \"{question}\"")
    logger.info(f"{'═'*60}")

    messages = [{"role": "user", "content": question}]
    step = 0
    start = time.time()

    while True:
        step += 1
        logger.info(f"\n  [Step {step}] Llamando a Claude...")

        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        registrar_tokens(response.usage, f"Step {step}")

        # Mostrar lo que Claude dijo/hizo
        for block in response.content:
            if block.type == "text" and block.text.strip():
                logger.info(f"  [Step {step}] \"{block.text[:150]}\"")
            elif block.type == "tool_use":
                logger.info(
                    f"  [Step {step}] Tool call: {block.name}({json.dumps(block.input)})"
                )

        # Si Claude terminó, retornar la respuesta
        if response.stop_reason == "end_turn":
            elapsed = time.time() - start
            final_text = ""
            for block in response.content:
                if block.type == "text":
                    final_text += block.text

            logger.info(f"\n  [Agent] Completado en {step} steps | {elapsed:.2f}s")
            return final_text

        # Si Claude quiere usar tools, ejecutar TODAS
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            # Procesar TODAS las tool calls (parallel tool use)
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    logger.info(f"  [Step {step}]   Ejecutando {block.name}...")
                    result = execute_tool(block.name, block.input)
                    logger.info(f"  [Step {step}] Resultado: {block.name} -> {result[:150]}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Enviar TODOS los resultados juntos
            messages.append({"role": "user", "content": tool_results})

        # Protección contra loops infinitos
        if step >= 10:
            logger.error("  [Agent] Maximo de steps alcanzado")
            return "Error: No pude resolver la pregunta en 10 pasos."


# ═══════════════════════════════════════════════════════════════════════════
# Interactive Chat
# ═══════════════════════════════════════════════════════════════════════════

def interactive_chat():
    """Chatbot interactivo con las 3 tools."""
    print("=" * 60)
    print("  MULTI-TOOL AGENT")
    print("  Tools: calculate, get_weather, get_time")
    print("  Escribe 'salir' para terminar")
    print("=" * 60)

    while True:
        user_message = input("\n Tú: ").strip()

        if not user_message:
            continue

        if user_message.lower() in ("salir", "quit", "exit"):
            print("\n¡Hasta luego!")
            resumen_costo_total()
            break

        answer = ask(user_message)
        print(f"\n Claude: {answer}")

        # Reset rate limits entre preguntas
        tool_call_counts.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Batch Test Mode
# ═══════════════════════════════════════════════════════════════════════════

def run_tests():
    """Ejecuta preguntas de prueba para validar las 3 tools."""
    print("=" * 60)
    print("  MULTI-TOOL AGENT — Test Mode")
    print("=" * 60)

    QUESTIONS = [
        # Solo get_time
        "¿Qué hora es en Tokyo?",

        # Solo get_weather
        "¿Qué temperatura hace en Puebla?",

        # Solo calculate
        "¿Cuánto es 2048 / 16?",

        # Parallel: get_time + get_weather
        "¿Qué hora es en Tokyo y qué clima hace en Puebla?",

        # Secuencial: get_weather + calculate
        "Si en Puebla hace cierta temperatura en celsius, ¿cuánto sería en fahrenheit? "
        "Primero consulta el clima y luego haz la conversión con la fórmula: (C * 9/5) + 32",

        # Multi-step con calculate
        "Tengo 250 rutas. El 30% son boulder, el 45% son sport, y el resto son trad. "
        "¿Cuántas rutas de cada tipo tengo?",

        # Sin tools
        "¿Qué es un dyno en escalada?",

        # Parallel: 3 tools a la vez
        "Necesito saber: qué hora es en New York, qué clima hace en London, "
        "y cuánto es 1500 * 0.16",
    ]

    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n{'─'*60}")
        print(f"  📨 Test {i}/{len(QUESTIONS)}: \"{question[:80]}{'...' if len(question) > 80 else ''}\"")

        answer = ask(question)
        print(f"\n  Respuesta: {answer}")

        tool_call_counts.clear()

    print(f"\n{'═'*60}")
    resumen_costo_total()


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-tool agent")
    parser.add_argument(
        "--test", action="store_true",
        help="Ejecutar preguntas de prueba en vez de chat interactivo"
    )
    args = parser.parse_args()

    if args.test:
        run_tests()
    else:
        interactive_chat()