"""
ejercicio2_3.py 
===========================
Ejercicio 2.3 — Code reviewer automático


Description:
-----------
- Input: bloque de código Python
- Output: JSON con {issues: [{line, severity, description, suggestion}], 
  overall_score: 1-10, summary: string}
- El reviewer debe conocer tu stack (FastAPI, SQLAlchemy, Pydantic)
- Usa CoT: primero analiza en <analysis> tags, luego resultado en <result> tags
- Probar con 5 funciones reales de Klimbook


Metadata:
----------
* Author: zxxz6 
* Version: 1.0.0


History:
------------
Author      Date            Description
zxxz6       02/02/2026      Creation

"""

from anthropic import Anthropic
from pydantic import BaseModel, ValidationError
from typing import Literal
import json
import time


client = Anthropic()


# ─── Modelos Pydantic ─────────────────────────────────────────────────────

class CodeIssue(BaseModel):
    line: int
    severity: Literal["low", "medium", "high", "critical"]
    description: str
    suggestion: str


class CodeReview(BaseModel):
    issues: list[CodeIssue]
    overall_score: int  # 1 a 10
    summary: str


# ─── Prompt de Sistema ────────────────────────────────────────────────────

SYSTEM_PROMPT = """Eres un revisor de código Python senior con más de 10 años de experiencia. 
Te especializas en el siguiente stack:

<stack>
- FastAPI (APIs REST, inyección de dependencias, middleware)
- SQLAlchemy (ORM, sesiones async, relaciones, consultas)
- Pydantic (validación de datos, BaseModel, Field, validadores)
- PostgreSQL / PostGIS (consultas espaciales, índices)
- asyncpg (conexiones async a base de datos)
- Docker (microservicios en contenedores)
- Redis (caché, sesiones)
</stack>

<review_criteria>
Al revisar código, evalúa los siguientes aspectos:
1. **Bugs y Errores**: Errores de lógica, excepciones no manejadas, tipos incorrectos
2. **Seguridad**: Inyección SQL, validación de entrada, problemas de autenticación, secretos expuestos
3. **Rendimiento**: Consultas N+1, índices faltantes, cálculos innecesarios, llamadas bloqueantes en async
4. **Buenas Prácticas**: Convenciones de nombres, principio DRY, SOLID, uso adecuado de type hints
5. **Específico de FastAPI**: Uso correcto de Depends, HTTPException, códigos de estado, modelos de respuesta
6. **Específico de SQLAlchemy**: Gestión de sesiones, carga eager/lazy, optimización de consultas
7. **Específico de Pydantic**: Validación de modelos, uso de Field, clase Config
8. **Manejo de Errores**: Falta de try/except, captura amplia de excepciones, registro de errores
9. **Documentación**: Docstrings faltantes, nombres de variables poco claros
10. **Testing**: Testeabilidad del código, valores hardcodeados que deberían ser configurables
</review_criteria>

<severity_guide>
- critical: Vulnerabilidades de seguridad, riesgo de pérdida de datos, crashes en producción
- high: Bugs que causarán comportamiento incorrecto, problemas de rendimiento a escala
- medium: Problemas de calidad de código, validación faltante, mal manejo de errores
- low: Problemas de estilo, docstrings faltantes, mejoras menores
</severity_guide>

<output_instructions>
1. Primero, analiza el código a fondo dentro de tags <analysis>. Piensa paso a paso:
   - ¿Qué hace este código?
   - ¿Qué podría salir mal?
   - ¿Qué patrones se están usando correcta o incorrectamente?
   - ¿Hay problemas de seguridad?
   - ¿Hay problemas de rendimiento?

2. Luego proporciona tu revisión dentro de tags <result> como un objeto JSON válido con este esquema:
{
    "issues": [
        {
            "line": <número_de_línea>,
            "severity": "low|medium|high|critical",
            "description": "<qué_está_mal>",
            "suggestion": "<cómo_arreglarlo>"
        }
    ],
    "overall_score": <1_a_10>,
    "summary": "<evaluación_general_breve>"
}

- Si el código es excelente y no tiene problemas, devuelve un array de issues vacío y una puntuación alta.
- Siempre devuelve JSON válido dentro de los tags <result>.
- Los números de línea deben hacer referencia al código original.
</output_instructions>
"""


# ─── Función Principal ────────────────────────────────────────────────────

def review_code(code: str, max_retries: int = 3) -> CodeReview | None:
    """
    Envía un bloque de código Python a Claude para revisión.
    
    Args:
        code: El código Python a revisar
        max_retries: Número de reintentos si falla el parseo de JSON
        
    Returns:
        Objeto CodeReview con problemas, puntuación y resumen
    """
    temp = 0.3

    for attempt in range(max_retries):
        print(f"\n--- Intento {attempt + 1}/{max_retries} (temperature={temp}) ---")

        start = time.time()

        response = client.messages.create(
            model="claude-opus-4-20250514",
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Review the following Python code:\n\n<code>\n{code}\n</code>"
                }
            ],
            temperature=temp,
            max_tokens=10000,
            stop_sequences=["</result>"]
        )

        elapsed = time.time() - start
        raw = response.content[0].text

        print(f"Tokens — entrada: {response.usage.input_tokens}, "
              f"salida: {response.usage.output_tokens}")
        print(f"Tiempo: {elapsed:.2f}s")

        # Extraer análisis (para mostrar)
        if "<analysis>" in raw and "</analysis>" in raw:
            analysis = raw.split("<analysis>")[1].split("</analysis>")[0].strip()
            print(f"\nAnálisis:\n{analysis}")

        # Extraer JSON del resultado
        try:
            if "<result>" in raw:
                json_str = raw.split("<result>")[1].strip()
            else:
                json_str = raw.strip()

            parsed = json.loads(json_str)
            review = CodeReview(**parsed)

            # Mostrar resultados
            print(f"\n{'='*60}")
            print(f"Resultados de la Revisión (Puntuación: {review.overall_score}/10)")
            print(f"{'='*60}")
            print(f"\nResumen: {review.summary}\n")

            if review.issues:
                print(f"Problemas encontrados ({len(review.issues)}):")
                print(f"{'-'*60}")
                for i, issue in enumerate(review.issues, 1):
                    severity_emoji = {
                        "critical": "🔴",
                        "high": "🟠",
                        "medium": "🟡",
                        "low": "🔵"
                    }
                    emoji = severity_emoji.get(issue.severity, "⚪")
                    print(f"\n  {i}. {emoji} [{issue.severity.upper()}] Línea {issue.line}")
                    print(f"     Problema:   {issue.description}")
                    print(f"     Sugerencia: {issue.suggestion}")
            else:
                print("Sin problemas encontrados!")

            return review

        except json.JSONDecodeError as e:
            print(f"Error al parsear JSON: {e}")
            print(f"Respuesta cruda: {raw[:300]}...")
            temp = 0.0

        except ValidationError as e:
            print(f"Error de validación Pydantic: {e}")
            temp = 0.0

    print("\nNo se pudo obtener una revisión válida después de todos los reintentos.")
    return None


# ─── Muestras de Código de Prueba ─────────────────────────────────────────

if __name__ == "__main__":

    # Muestra 1: Endpoint de FastAPI con varios problemas
    sample_1 = '''
@router.get("/users/{user_id}")
async def get_user(user_id, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if user == None:
        return {"error": "not found"}
    return user
'''

    # Muestra 2: Consulta SQLAlchemy con problemas potenciales
    sample_2 = '''
async def get_routes_near_location(lat, lon, radius_km, db):
    query = f"SELECT * FROM routes WHERE ST_DWithin(geom, ST_MakePoint({lon}, {lat})::geography, {radius_km * 1000})"
    result = await db.execute(text(query))
    routes = result.fetchall()
    return [dict(r) for r in routes]
'''

    # Muestra 3: Modelo Pydantic con problemas
    sample_3 = '''
class CreateRouteRequest(BaseModel):
    name: str
    grade: str
    latitude: float
    longitude: float
    description: str = None
    
    def validate_grade(self):
        valid_grades = ["5.6", "5.7", "5.8", "5.9", "5.10a", "5.10b"]
        if self.grade not in valid_grades:
            raise ValueError("Invalid grade")
'''

    # Muestra 4: Middleware de autenticación
    sample_4 = '''
async def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload["user_id"]
        return user_id
    except:
        return None

@router.post("/ascents")
async def create_ascent(request: Request, data: dict):
    token = request.headers.get("Authorization")
    user_id = await verify_token(token)
    if not user_id:
        return {"error": "unauthorized"}
    
    db = get_db()
    new_ascent = Ascent(user_id=user_id, **data)
    db.add(new_ascent)
    db.commit()
    return {"status": "created"}
'''

    # Muestra 5: Función de caché con Redis
    sample_5 = '''
import redis
import json
import pickle

r = redis.Redis(host="localhost", port=6379)

def get_popular_routes(limit=10):
    cached = r.get("popular_routes")
    if cached:
        return pickle.loads(cached)
    
    db = SessionLocal()
    routes = db.query(Route).order_by(Route.ascent_count.desc()).limit(limit).all()
    db.close()
    
    r.set("popular_routes", pickle.dumps(routes), ex=3600)
    return routes
'''

    samples = {
        "FastAPI endpoint": sample_1,
        "PostGIS query": sample_2,
        "Pydantic model": sample_3,
        "Auth middleware": sample_4,
        "Redis caching": sample_5,
    }

    for name, code in samples.items():
        print(f"\n{'#'*60}")
        print(f"# Revisando: {name}")
        print(f"{'#'*60}")
        print(f"\nCódigo:\n{code}")

        review = review_code(code)

        if review:
            print(f"\nRevisión completa — Puntuación: {review.overall_score}/10")
        else:
            print(f"\nRevisión fallida")

        print(f"\n{'='*60}\n")