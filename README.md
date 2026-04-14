# AI Agents Playground

Repositorio de aprendizaje práctico sobre **agentes de IA** con la API de Claude (Anthropic). Cubre desde los fundamentos de la API hasta patrones avanzados como RAG, sistemas multiagente y MCP.

> Gran parte del material está basado en los cursos oficiales de Anthropic: [https://github.com/anthropics/courses](https://github.com/anthropics/courses)

---

## Estructura del Repositorio

| Unidad | Tema | Descripción |
|--------|------|-------------|
| **Unidad 1** | Fundamentos | API de Claude, parámetros, streaming, técnicas de prompting |
| **Unidad 2** | Chains | Pipelines secuenciales, paralelos, routers y map-reduce |
| **Unidad 3** | Tool Use | Definición de herramientas, agent loop, integración con bases de datos |
| **Unidad 4** | Proyecto | Proyecto integrador (en progreso) |
| **Unidad 5** | Avanzados | RAG, sistemas multiagente, MCP |

---

## Unidad 1 — Fundamentos

### Lección 1: API de Claude

Introducción a la API de Anthropic: modelos, parámetros y streaming.

| Ejercicio | Descripción |
|-----------|-------------|
| `ejercicio1_1.py` | Exploración de modelos (Sonnet, Haiku), temperatura, `max_tokens` y system prompts. Compara el comportamiento del modelo con distintas configuraciones. |
| `ejercicio1_2.py` | Chatbot multi-turno con historial de conversación, comandos especiales (`/clear`, `/system`, `/tokens`, `/quit`) y seguimiento de consumo de tokens con estimación de costos en USD. |
| `ejercicio1_3.py` | Chatbot con streaming asíncrono (`AsyncAnthropic`). Muestra respuestas token a token, mide el tiempo al primer token (TTFT) vs. tiempo total, y maneja interrupciones con Ctrl+C. |
| `notes.ipynb` | Notebook con teoría y experimentación sobre los fundamentos de la API. |

### Lección 2: Prompting

Técnicas avanzadas de ingeniería de prompts: few-shot, chain-of-thought, salida estructurada y encadenamiento.

| Ejercicio | Descripción |
|-----------|-------------|
| `ejercicio2_1.py` | Extrae changelogs estructurados de logs de git usando few-shot learning. Clasifica commits (feature/fix/refactor/docs/chore), detecta breaking changes y valida con Pydantic. |
| `ejercicio2_2.py` | Traductor técnico de release notes (EN→ES) con glosario de dominio. Preserva formato markdown, emojis y terminología de escalada. |
| `ejercicio2_3.py` | Revisor de código (Python/FastAPI/SQLAlchemy/Pydantic). Analiza bugs, seguridad, rendimiento y principios SOLID. Usa chain-of-thought y devuelve JSON con score 1-10. |
| `ejercicio2_4.py` | Encadenamiento de prompts en 3 pasos: extraer cambios del diff → generar release notes → crear post para Ko-fi. Compara tokens, costo y calidad vs. un solo prompt. |
| `notes.ipynb` | Notebook sobre técnicas de prompting: system prompts, few-shot, CoT, salidas estructuradas y patrones de validación. |

---

## Unidad 2 — Chains

### Lección 1: Fundamentos de Chains

Pipelines asincrónicos de múltiples pasos con validación y reintentos.

| Ejercicio | Descripción |
|-----------|-------------|
| `ejercicio3_1.py` | Pipeline de 3 pasos: extraer puntos clave → resumir en español → generar hilo de tweets. Cachea checkpoints y rastrea tokens/costo acumulado. |
| `ejercicio3_2.py` | Extiende el anterior con validación Pydantic entre pasos. Reintenta con temperatura decreciente (0.7→0.3→0.0) en caso de fallo. Decorador `@retry_on_validation_error` y logging estructurado. |
| `notes.ipynb` | Teoría sobre cadenas: procesamiento secuencial, cadenas paralelas, ramificación y manejo de errores. |

### Lección 2: Chains Avanzadas

Patrones avanzados: paralelización, routing y map-reduce.

| Ejercicio | Descripción |
|-----------|-------------|
| `ejercicio4_1.py` | Chain paralela con 4 tareas concurrentes usando `asyncio.gather` y `Semaphore`. Compara rendimiento secuencial vs. paralelo (speedup de 3-4x). |
| `ejercicio4_2.py` | Router chain: clasifica mensajes (bug/feature/pregunta/feedback) y los enruta a cadenas especializadas que generan borradores de issues de GitHub. |
| `ejercicio4_3.py` | Resumen map-reduce: mapea archivos de un repo a resúmenes individuales y los reduce en un resumen ejecutivo. Fragmenta archivos grandes con overlap de 100 tokens. |
| `notes.ipynb` | Patrones avanzados de cadenas: paralelización, routing, map-reduce y optimización. |

---

## Unidad 3 — Tool Use

### Lección 1: Tools

Definición y uso de herramientas (tools) con la API de Claude.

| Ejercicio | Descripción |
|-----------|-------------|
| `ejercicio5_1.py` | Herramienta de calculadora con parsing AST seguro (sin `eval()`). Define el esquema de la herramienta, valida entradas y maneja errores (división por cero, overflow, sintaxis inválida). |
| `ejercicio5_2.py` | Agente multi-herramienta con 3 tools: `calculate()`, `get_weather()` (API Open-Meteo) y `get_time()` (zonas horarias). Orquesta múltiples herramientas en un solo flujo. |
| `notes.ipynb` | Teoría sobre tool use: esquemas JSON, definición de herramientas, cómo el LLM invoca tools y manejo de respuestas. |

### Lección 2: Agent Loop

Implementación del ciclo de agente con herramientas y restricciones de seguridad.

| Ejercicio | Descripción |
|-----------|-------------|
| `ejercicio6_1.py` | Agente explorador de archivos con sandbox de seguridad (`PathSandbox`). Tools: `list_directory()`, `read_file()`, `search_text()`. Límites de iteraciones, tokens y errores. |
| `ejercicio6_2.py` | Agente de la API de GitHub. Tools: listar repos, obtener info, listar issues/commits. Maneja rate limits y permite crear issues con confirmación del usuario. |
| `notes.ipynb` | Mecánica del agent loop: llamadas a herramientas, parsing de respuestas, condiciones de parada y razonamiento multi-paso. |

### Lección 3: Bases de Datos

Agente que genera y ejecuta consultas SQL a partir de lenguaje natural.

| Ejercicio | Descripción |
|-----------|-------------|
| `ejercicio7_1/agent.py` | Agente de consultas a BD. Tools: `list_tables`, `get_table_schema`, `get_sample_data`, `query_database`. Validación SQL con sqlparse (solo SELECT), conexión AsyncPG con pool, y soporte PostGIS. Usuario readonly por seguridad. |
| `ejercicio7_1/setup.py` | Script de setup que crea la BD PostgreSQL con esquema Klimbook: usuarios, crags, rutas y ascensos. Genera datos de prueba (50 usuarios, 20 crags de México, 200 rutas, 500 ascensos) con extensión PostGIS. |
| `notes.ipynb` | Patrones de integración con bases de datos: diseño de esquema, consultas desde lenguaje natural y consideraciones de seguridad. |

---

## Unidad 4 — Proyecto

Proyecto integrador. *En progreso.*

---

## Unidad 5 — Avanzados

### Lección 11: RAG (Retrieval-Augmented Generation)

| Archivo | Descripción |
|---------|-------------|
| `notes.ipynb` | Teoría y práctica de RAG: estrategias de chunking (semántico, ventana deslizante), modelos de embeddings, bases de datos vectoriales, búsqueda por similitud y re-ranking. |

### Lección 12: Sistemas Multiagente

| Archivo | Descripción |
|---------|-------------|
| `notes.ipynb` | Sistemas multiagente: coordinación, comunicación entre agentes, delegación de tareas y mecanismos de consenso. Patrones secuenciales, paralelos y jerárquicos. |

### Lección 13: MCP (Model Context Protocol)

| Archivo | Descripción |
|---------|-------------|
| `notes.ipynb` | Fundamentos del protocolo MCP: integración de Claude con sistemas externos, definiciones estandarizadas de herramientas/recursos y comunicación bidireccional. Ejemplos de implementación de servidor MCP. |

---

## Stack Tecnológico

| Categoría | Tecnologías |
|-----------|-------------|
| **LLM** | Anthropic Claude API (Sonnet, Haiku) |
| **Python** | `anthropic`, `asyncio`, `pydantic`, `httpx` |
| **Base de Datos** | PostgreSQL + PostGIS, `asyncpg`, `sqlparse` |
| **APIs externas** | Open-Meteo (clima), GitHub API |
| **Otros** | `zoneinfo`, streaming, AST parsing |

## Patrones Clave

- Streaming de respuestas y conversaciones multi-turno
- Prompt chaining (encadenamiento de 2-3 pasos)
- Pipelines asincrónicos con `asyncio.gather`
- Tool calling y ciclo de agente (agent loop)
- Validación con Pydantic y reintentos con temperatura decreciente
- Embeddings vectoriales y RAG
- Coordinación multiagente
- Model Context Protocol (MCP)
