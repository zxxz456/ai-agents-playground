"""
ejercicio4_2.py 
===========================
Ejercicio 4.2 — Router chain


Description:
-----------
- Clasificador: recibe un mensaje de usuario y lo clasifica como 
  'bug_report', 'feature_request', 'question', o 'feedback'
- Chains especializados: cada categoría tiene un chain diferente 
  con prompts optimizados
- Bug report chain: extrae pasos para reproducir, severidad estimada, 
  componente afectado, y genera un draft de issue para GitHub
- Feature request chain: analiza viabilidad, estima complejidad, y 
  genera un draft de issue con labels sugeridos
- Implementar como un dict de funciones: 
  routes = {"bug_report": handle_bug, "feature_request": handle_feature, ...}


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
from pydantic import BaseModel, ValidationError, field_validator
from typing import Literal
import asyncio
import logging
import json
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("router")

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

costos_acumulados = []
tokens_acumulados = {"input": 0, "output": 0}


def registrar_tokens(usage, step_name: str, model_name: str):
    """Registra tokens y acumula costo."""
    cfg = MODELS[model_name]
    inp, out = usage.input_tokens, usage.output_tokens
    tokens_acumulados["input"] += inp
    tokens_acumulados["output"] += out
    costo = (inp / 1e6) * cfg["input"] + (out / 1e6) * cfg["output"]
    costos_acumulados.append(costo)
    logger.info(
        f"  [{step_name}] Tokens → in: {inp}, out: {out} | "
        f"Costo: ${costo:.6f}"
    )


def resumen_costo_total():
    """Imprime resumen final."""
    total = sum(costos_acumulados)
    logger.info(
        f"[TOTAL] Tokens in: {tokens_acumulados['input']} | "
        f"out: {tokens_acumulados['output']} | "
        f"Steps: {len(costos_acumulados)} | "
        f"Costo: ${total:.6f}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════════════════════════════════════

class Classification(BaseModel):
    """Output del clasificador."""
    category: Literal["bug_report", "feature_request", "question", "feedback"]
    confidence: float

    @field_validator("confidence")
    @classmethod
    def confidence_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence debe estar entre 0.0 y 1.0, recibido: {v}")
        return v


class BugReport(BaseModel):
    """Output del handler de bug reports."""
    title: str
    steps_to_reproduce: list[str]
    expected_behavior: str
    actual_behavior: str
    severity: Literal["low", "medium", "high", "critical"]
    component: str
    labels: list[str]


class FeatureRequest(BaseModel):
    """Output del handler de feature requests."""
    title: str
    description: str
    use_case: str
    feasibility: Literal["easy", "moderate", "complex", "needs_research"]
    complexity: Literal["S", "M", "L", "XL"]
    labels: list[str]
    priority_suggestion: Literal["low", "medium", "high"]


class QuestionResponse(BaseModel):
    """Output del handler de preguntas."""
    answer: str
    related_docs: list[str]
    needs_escalation: bool


class FeedbackResponse(BaseModel):
    """Output del handler de feedback."""
    acknowledgment: str
    sentiment: Literal["positive", "neutral", "negative"]
    actionable: bool
    suggested_action: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Step 0: Clasificador
# ═══════════════════════════════════════════════════════════════════════════

CONFIDENCE_THRESHOLD = 0.7

async def step_classify(message: str) -> Classification:
    """
    Clasifica un mensaje de usuario en una categoría.
    Usa Haiku con temp=0 para máxima consistencia.
    """
    step_name = "Clasificador"
    start = time.time()

    response = await client.messages.create(
        model=MODELS["haiku"]["id"],
        system="""Eres un clasificador de mensajes de soporte para Klimbook, 
una red social para escaladores.

Clasifica cada mensaje en UNA categoría:
- bug_report: el usuario reporta un error, mal funcionamiento, o crash
- feature_request: el usuario pide una funcionalidad nueva o mejora
- question: el usuario pregunta cómo usar algo o pide información
- feedback: el usuario da una opinión, comentario, o agradecimiento

Responde SOLO con JSON válido:
{"category": "...", "confidence": 0.0-1.0}

Reglas:
- Si el mensaje es ambiguo, elige la categoría más probable y baja el confidence
- Si el mensaje no tiene sentido o es spam, clasifica como "feedback" con confidence bajo
- Analiza el tono y las palabras clave para determinar la categoría""",
        messages=[
            {"role": "user", "content": message},
            {"role": "assistant", "content": "{"},
        ],
        temperature=0.0,
        max_tokens=100,
    )

    registrar_tokens(response.usage, step_name, "haiku")
    elapsed = time.time() - start

    raw = "{" + response.content[0].text
    parsed = json.loads(raw)
    result = Classification(**parsed)

    logger.info(
        f"  [{step_name}] → {result.category} "
        f"(confidence: {result.confidence:.2f}) | {elapsed:.2f}s"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Handler: Bug Reports
# ═══════════════════════════════════════════════════════════════════════════

async def handle_bug(message: str) -> dict:
    """
    Chain para bug reports.
    Extrae pasos para reproducir, severidad, componente, y genera draft de issue.
    Usa Sonnet para análisis detallado.
    """
    step_name = "Bug Report"
    start = time.time()

    response = await client.messages.create(
        model=MODELS["sonnet"]["id"],
        system="""Eres un QA engineer senior para Klimbook, una red social 
para escaladores construida con FastAPI, PostgreSQL/PostGIS, React Native, 
y desplegada con Docker en un VPS.

Analiza el bug report del usuario y genera un draft de issue para GitHub.

Componentes conocidos de Klimbook:
- auth: autenticación, login, registro, tokens
- climbing: rutas, ascensos, grados, bloques
- social: seguidores, feed, notificaciones
- book: libro de escalador, estadísticas
- profile: perfil de usuario, preferencias
- navigation: navegación, deep links
- media: fotos, imágenes de rutas
- search: búsqueda de rutas, crags, usuarios
- general: si no encaja en ninguno específico

Responde SOLO con JSON válido con esta estructura exacta:
{
    "title": "Bug: <título conciso>",
    "steps_to_reproduce": ["paso 1", "paso 2", "..."],
    "expected_behavior": "qué debería pasar",
    "actual_behavior": "qué pasa realmente",
    "severity": "low|medium|high|critical",
    "component": "<componente afectado>",
    "labels": ["bug", "<severity>", "<componente>"]
}""",
        messages=[
            {"role": "user", "content": message},
            {"role": "assistant", "content": "{"},
        ],
        temperature=0.2,
        max_tokens=1000,
    )

    registrar_tokens(response.usage, step_name, "sonnet")
    elapsed = time.time() - start

    raw = "{" + response.content[0].text
    parsed = json.loads(raw)
    result = BugReport(**parsed)

    logger.info(f"  [{step_name}]  Issue draft generado | {elapsed:.2f}s")
    return result.model_dump()


# ═══════════════════════════════════════════════════════════════════════════
# Handler: Feature Requests
# ═══════════════════════════════════════════════════════════════════════════

async def handle_feature(message: str) -> dict:
    """
    Chain para feature requests.
    Analiza viabilidad, complejidad, y genera draft de issue con labels.
    Usa Sonnet para análisis detallado.
    """
    step_name = "Feature Request"
    start = time.time()

    response = await client.messages.create(
        model=MODELS["sonnet"]["id"],
        system="""Eres un product manager técnico para Klimbook, una red social 
para escaladores. Conoces el stack (FastAPI, PostgreSQL/PostGIS, React Native, 
Docker) y las capacidades actuales de la app.

Analiza el feature request y genera un draft de issue para GitHub.

Guía de complejidad:
- S: cambio menor, < 1 día (ajuste de UI, nuevo campo)
- M: feature pequeña, 1-3 días (nueva pantalla, endpoint simple)
- L: feature mediana, 1-2 semanas (sistema nuevo, integración)
- XL: feature grande, 2+ semanas (arquitectura nueva, migración)

Guía de viabilidad:
- easy: se puede hacer con el stack actual sin cambios mayores
- moderate: requiere algo de investigación o cambios moderados
- complex: requiere cambios significativos en arquitectura
- needs_research: no está claro cómo implementarlo, requiere POC

Responde SOLO con JSON válido:
{
    "title": "Feature: <título descriptivo>",
    "description": "<descripción clara del feature>",
    "use_case": "<caso de uso principal>",
    "feasibility": "easy|moderate|complex|needs_research",
    "complexity": "S|M|L|XL",
    "labels": ["enhancement", "<complexity>", "<otro label relevante>"],
    "priority_suggestion": "low|medium|high"
}""",
        messages=[
            {"role": "user", "content": message},
            {"role": "assistant", "content": "{"},
        ],
        temperature=0.3,
        max_tokens=1000,
    )

    registrar_tokens(response.usage, step_name, "sonnet")
    elapsed = time.time() - start

    raw = "{" + response.content[0].text
    parsed = json.loads(raw)
    result = FeatureRequest(**parsed)

    logger.info(f"  [{step_name}]  Issue draft generado | {elapsed:.2f}s")
    return result.model_dump()


# ═══════════════════════════════════════════════════════════════════════════
# Handler: Questions
# ═══════════════════════════════════════════════════════════════════════════

async def handle_question(message: str) -> dict:
    """
    Chain para preguntas.
    Responde basándose en conocimiento de Klimbook.
    Usa Sonnet para respuestas detalladas.
    """
    step_name = "Question"
    start = time.time()

    response = await client.messages.create(
        model=MODELS["sonnet"]["id"],
        system="""Eres un agente de soporte amigable de Klimbook, una red social 
para escaladores. Respondes preguntas sobre cómo usar la app.

Funcionalidades de Klimbook:
- Registrar ascensos (boulder y sport climbing)
- Buscar rutas y crags con mapa interactivo
- Seguir a otros escaladores y ver su actividad
- Libro de escalador con estadísticas personales
- Sistema de notificaciones
- Soporte multi-idioma (español e inglés)
- Tutorial de onboarding para nuevos usuarios

Si no sabes la respuesta, di honestamente que no la tienes y sugiere 
contactar al desarrollador.

Responde con JSON:
{
    "answer": "<respuesta clara y amigable>",
    "related_docs": ["<doc o sección relevante>"],
    "needs_escalation": true/false
}""",
        messages=[
            {"role": "user", "content": message},
            {"role": "assistant", "content": "{"},
        ],
        temperature=0.5,
        max_tokens=800,
    )

    registrar_tokens(response.usage, step_name, "sonnet")
    elapsed = time.time() - start

    raw = "{" + response.content[0].text
    parsed = json.loads(raw)
    result = QuestionResponse(**parsed)

    logger.info(f"  [{step_name}]  Respuesta generada | escalation={result.needs_escalation} | {elapsed:.2f}s")
    return result.model_dump()


# ═══════════════════════════════════════════════════════════════════════════
# Handler: Feedback
# ═══════════════════════════════════════════════════════════════════════════

async def handle_feedback(message: str) -> dict:
    """
    Chain para feedback.
    Agradece y analiza el sentimiento.
    Usa Haiku porque es una tarea simple.
    """
    step_name = "Feedback"
    start = time.time()

    response = await client.messages.create(
        model=MODELS["haiku"]["id"],
        system="""Eres el community manager de Klimbook. Agradece el feedback 
del usuario de forma genuina, breve y personal. Eres escalador tú mismo.

Analiza si el feedback es accionable (sugiere algo concreto que se pueda 
implementar o cambiar).

Responde con JSON:
{
    "acknowledgment": "<agradecimiento genuino y breve>",
    "sentiment": "positive|neutral|negative",
    "actionable": true/false,
    "suggested_action": "<qué hacer con este feedback, si aplica>"
}""",
        messages=[
            {"role": "user", "content": message},
            {"role": "assistant", "content": "{"},
        ],
        temperature=0.7,
        max_tokens=400,
    )

    registrar_tokens(response.usage, step_name, "haiku")
    elapsed = time.time() - start

    raw = "{" + response.content[0].text
    parsed = json.loads(raw)
    result = FeedbackResponse(**parsed)

    logger.info(f"  [{step_name}]  sentiment={result.sentiment} actionable={result.actionable} | {elapsed:.2f}s")
    return result.model_dump()


# ═══════════════════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════════════════

# Dict que mapea categoría → handler
ROUTES = {
    "bug_report": handle_bug,
    "feature_request": handle_feature,
    "question": handle_question,
    "feedback": handle_feedback,
}

async def router(message: str) -> dict:
    """
    Router principal:
    1. Clasifica el mensaje (Haiku, temp=0)
    2. Verifica confidence threshold
    3. Despacha al chain especializado
    """
    logger.info(f"\n{'═'*60}")
    logger.info(f"[Router] Nuevo mensaje: \"{message[:80]}...\"")
    logger.info(f"{'═'*60}")

    pipeline_start = time.time()

    # Paso 1: Clasificar
    try:
        classification = await step_classify(message)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"[Router] Clasificación falló: {e}")
        logger.info("[Router] Fallback → feedback")
        classification = Classification(category="feedback", confidence=0.0)

    # Paso 2: Verificar confidence
    if classification.confidence < CONFIDENCE_THRESHOLD:
        logger.warning(
            f"[Router] Baja confianza ({classification.confidence:.2f}) "
            f"para categoría '{classification.category}'"
        )
        return {
            "category": classification.category,
            "confidence": classification.confidence,
            "result": {
                "message": "No estoy seguro de entender tu mensaje. "
                "¿Podrías dar más detalles sobre lo que necesitas? "
                "¿Es un error, una sugerencia, o una pregunta?"
            },
            "low_confidence": True,
        }

    # Paso 3: Despachar al handler correcto
    handler = ROUTES.get(classification.category, handle_feedback)

    try:
        result = await handler(message)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"[Router] Handler '{classification.category}' falló: {e}")
        result = {"error": str(e), "raw_category": classification.category}

    elapsed = time.time() - pipeline_start

    logger.info(f"[Router]  Completado en {elapsed:.2f}s")

    return {
        "category": classification.category,
        "confidence": classification.confidence,
        "result": result,
        "low_confidence": False,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Pretty Print
# ═══════════════════════════════════════════════════════════════════════════

def print_result(response: dict):
    """Imprime el resultado del router de forma legible."""
    category = response["category"]
    confidence = response["confidence"]
    result = response["result"]

    print(f"\n  📋 Categoría: {category} (confidence: {confidence:.2f})")

    if response.get("low_confidence"):
        print(f"  ⚠️  {result['message']}")
        return

    if category == "bug_report":
        print(f"  🐛 {result['title']}")
        print(f"     Severidad: {result['severity']} | Componente: {result['component']}")
        print(f"     Pasos para reproducir:")
        for i, step in enumerate(result["steps_to_reproduce"], 1):
            print(f"       {i}. {step}")
        print(f"     Esperado: {result['expected_behavior']}")
        print(f"     Actual:   {result['actual_behavior']}")
        print(f"     Labels:   {', '.join(result['labels'])}")

    elif category == "feature_request":
        print(f"  🚀 {result['title']}")
        print(f"     {result['description']}")
        print(f"     Caso de uso: {result['use_case']}")
        print(f"     Viabilidad: {result['feasibility']} | Complejidad: {result['complexity']}")
        print(f"     Prioridad: {result['priority_suggestion']}")
        print(f"     Labels: {', '.join(result['labels'])}")

    elif category == "question":
        print(f"  ❓ Respuesta:")
        print(f"     {result['answer']}")
        if result["related_docs"]:
            print(f"     Docs: {', '.join(result['related_docs'])}")
        if result["needs_escalation"]:
            print(f"     ⚠️  Necesita escalación")

    elif category == "feedback":
        emoji = {"positive": "😊", "neutral": "😐", "negative": "😔"}
        print(f"  {emoji.get(result['sentiment'], '💬')} {result['acknowledgment']}")
        if result["actionable"]:
            print(f"     📌 Acción sugerida: {result['suggested_action']}")


# ═══════════════════════════════════════════════════════════════════════════
# Test Messages
# ═══════════════════════════════════════════════════════════════════════════

TEST_MESSAGES = [
    # Bug reports
    "The app crashes every time I try to log a new ascent on my boulder project. "
    "I tap 'Log Ascent', select the route, choose 'Flash', and the app just closes. "
    "This started happening after the last update. I'm on Android 14.",

    "When I search for crags near Puebla, the map shows locations in Spain instead. "
    "The GPS seems to be working fine because other apps show my correct location.",

    # Feature requests
    "It would be awesome if Klimbook had a training log where I can track my "
    "hangboard sessions, campus board workouts, and stretching. Maybe with progress "
    "charts over time?",

    "Can you add a way to compare my climbing stats with my friends? Like a "
    "leaderboard for our climbing group showing who has the most ascents this month.",

    # Questions
    "How do I change the grading system in the app? I want to use the French "
    "scale instead of the Yosemite Decimal System.",

    "Is there a way to export my climbing data? I want to analyze my progress "
    "in a spreadsheet.",

    # Feedback
    "Just wanted to say I love this app! The route search is amazing and the "
    "community features make it so much better than other climbing apps.",

    "The new tutorial was helpful but a bit long. Maybe you could add a 'skip' "
    "option for experienced users who already know how climbing apps work.",

    # Ambiguous (low confidence expected)
    "hmm interesting app",
]


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    print("=" * 60)
    print("  KLIMBOOK MESSAGE ROUTER")
    print("  Clasificador + Chains especializados")
    print("=" * 60)

    for i, message in enumerate(TEST_MESSAGES, 1):
        print(f"\n{'─'*60}")
        print(f"   Mensaje {i}/{len(TEST_MESSAGES)}:")
        print(f"  \"{message[:100]}{'...' if len(message) > 100 else ''}\"")

        response = await router(message)
        print_result(response)

    print(f"\n{'═'*60}")
    resumen_costo_total()


if __name__ == "__main__":
    asyncio.run(main())