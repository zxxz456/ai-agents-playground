"""
ejercicio7_1_agent.py
===========================
Ejercicio 7.1 — Database agent


Description:
-----------
Agente que consulta la base de datos de Klimbook usando lenguaje natural.
4 tools: list_tables, get_table_schema, get_sample_data, query_database
Validacion SQL con sqlparse (solo SELECT permitido).
Soporte PostGIS para queries geograficos.

Ejecutar DESPUES del setup:
    python ejercicio7_1_setup.py   (crear DB y datos)
    python ejercicio7_1_agent.py   (ejecutar agente)


Metadata:
----------
* Author: zxxz6
* Version: 1.0.0


History:
------------
Author      Date            Description
zxxz6       17/03/2026      Creation

"""

from anthropic import Anthropic
from dataclasses import dataclass, field
import asyncpg
import asyncio
import sqlparse
import json
import time
import logging
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("db-agent")

client = Anthropic()

MODEL = "claude-sonnet-4-20250514"


# =====================================================================
# Database Config
# =====================================================================

# Usamos el usuario readonly para que el agente NO pueda
# modificar datos incluso si Claude genera un INSERT/UPDATE/DELETE.
# Es una doble proteccion: validacion en codigo + permisos en DB.
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "klimbook_test",
    "user": "readonly_agent",
    "password": "readonly_pass",
}

# El pool se inicializa en main()
pool: asyncpg.Pool = None


async def init_pool():
    """
    Crea el pool de conexiones a PostgreSQL.
    
    Un pool mantiene varias conexiones abiertas a la DB y las
    reparte entre requests. Esto evita el overhead de abrir/cerrar
    conexiones (cada apertura tarda ~50-100ms).
    
    min_size=2: siempre tener al menos 2 conexiones listas
    max_size=5: nunca tener mas de 5 (para no sobrecargar la DB)
    command_timeout=10: si un query tarda mas de 10s, cancelarlo
    """
    global pool
    pool = await asyncpg.create_pool(
        **DB_CONFIG,
        min_size=2,
        max_size=5,
        command_timeout=10,
    )
    logger.info("[DB] Pool de conexiones creado")


async def close_pool():
    """Cierra el pool de conexiones."""
    if pool:
        await pool.close()
        logger.info("[DB] Pool cerrado")


# =====================================================================
# SQL Validation
# =====================================================================
#
# Este es el componente de seguridad mas critico.
# Claude puede generar CUALQUIER SQL, incluyendo DROP TABLE.
# Estas funciones aseguran que solo se ejecuten SELECTs seguros.
#

# Palabras que NUNCA deben aparecer en un query del agente.
# Buscamos la palabra completa (con espacios alrededor) para no
# bloquear columnas como "updated_at" o "deleted_flag".
DANGEROUS_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "TRUNCATE", "GRANT", "REVOKE", "EXEC", "EXECUTE",
    "INTO",  # bloquea SELECT INTO que crea tablas
]


def is_safe_query(sql: str) -> tuple[bool, str]:
    """
    Verifica que el SQL sea un SELECT seguro.
    
    Retorna una tupla (es_seguro, razon).
    Si es seguro: (True, "")
    Si no: (False, "explicacion del problema")
    
    El proceso de validacion tiene dos capas:
    1. sqlparse analiza la estructura del SQL y determina el tipo
    2. Busqueda de keywords peligrosos como respaldo
    """
    sql_stripped = sql.strip()

    if not sql_stripped:
        return False, "Query vacio"

    # Capa 1: Analizar con sqlparse
    # sqlparse parsea el SQL en un arbol y determina el tipo
    # de statement (SELECT, INSERT, UPDATE, etc.)
    try:
        parsed_statements = sqlparse.parse(sql_stripped)
    except Exception as e:
        return False, f"Error parseando SQL: {e}"

    if not parsed_statements:
        return False, "No se pudo parsear el SQL"

    for statement in parsed_statements:
        stmt_type = statement.get_type()

        # Solo permitir SELECT y queries que sqlparse no puede tipear
        # (subconsultas complejas a veces retornan None como tipo)
        if stmt_type and stmt_type != "SELECT":
            return False, f"Tipo de query no permitido: {stmt_type}. Solo SELECT."

    # Capa 2: Buscar keywords peligrosos manualmente
    # Esto es un respaldo por si sqlparse no detecta algo.
    # Agregamos espacios alrededor para buscar la palabra completa.
    sql_upper = f" {sql_stripped.upper()} "
    for keyword in DANGEROUS_KEYWORDS:
        if f" {keyword} " in sql_upper:
            return False, f"Keyword peligroso detectado: {keyword}"

    return True, ""


def add_limit(sql: str, max_rows: int = 100) -> str:
    """
    Agrega LIMIT al query si no lo tiene.
    
    Esto previene que queries sin LIMIT retornen millones de rows
    y saturen la memoria del agente o excedan el context window de Claude.
    """
    sql_upper = sql.upper().strip()
    if "LIMIT" not in sql_upper:
        sql = sql.rstrip(";").strip()
        sql = f"{sql} LIMIT {max_rows}"
    return sql


# =====================================================================
# Row Serialization
# =====================================================================
#
# asyncpg retorna objetos Record que no son serializables a JSON.
# Ademas, PostgreSQL tiene tipos que JSON no soporta:
# - datetime -> string ISO 8601
# - Decimal -> float
# - UUID -> string
# - bytes -> skip
# - geometry -> WKT string
#

def row_to_dict(row) -> dict:
    """
    Convierte un asyncpg Record a un dict serializable a JSON.
    
    Maneja tipos especiales de PostgreSQL que no son compatibles
    con json.dumps directamente.
    """
    result = {}
    for key, value in dict(row).items():
        if value is None:
            result[key] = None
        elif isinstance(value, (str, int, float, bool)):
            result[key] = value
        elif hasattr(value, "isoformat"):
            # datetime, date, time -> string ISO 8601
            result[key] = value.isoformat()
        elif isinstance(value, bytes):
            # Datos binarios (ej: geometrias crudas) -> saltar
            result[key] = "<binary>"
        else:
            # Fallback: convertir a string
            # Esto captura Decimal, UUID, y otros tipos
            result[key] = str(value)
    return result


# =====================================================================
# Tool Functions
# =====================================================================

async def list_tables() -> dict:
    """
    Lista todas las tablas en la base de datos.
    
    Consulta information_schema.tables, que es una "meta-tabla"
    de PostgreSQL que contiene info sobre la estructura de la DB.
    Filtramos por table_schema='public' para excluir tablas internas.
    """
    logger.info("  [Tool] list_tables()")

    query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query)

    tables = [row["table_name"] for row in rows]
    logger.info(f"  [Tool] -> {len(tables)} tablas: {tables}")

    return {"tables": tables, "count": len(tables)}


async def get_table_schema(table_name: str) -> dict:
    """
    Retorna el schema de una tabla: columnas, tipos, PKs, FKs.
    
    Claude necesita esta informacion para generar SQL correcto.
    Sin conocer el schema, Claude inventaria nombres de columnas.
    
    Las foreign keys son especialmente importantes porque le dicen
    a Claude como hacer JOINs entre tablas.
    """
    logger.info(f"  [Tool] get_table_schema(table_name='{table_name}')")

    # Validar nombre de tabla para prevenir SQL injection.
    # Solo permitimos letras, numeros, y underscore.
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
        return {"error": f"Nombre de tabla invalido: '{table_name}'"}

    # Obtener columnas con tipos y propiedades
    columns_query = """
        SELECT 
            column_name, 
            data_type,
            udt_name,
            is_nullable,
            column_default,
            character_maximum_length
        FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = $1
        ORDER BY ordinal_position
    """

    # Obtener primary keys
    pk_query = """
        SELECT kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu 
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        WHERE tc.table_name = $1 
        AND tc.table_schema = 'public'
        AND tc.constraint_type = 'PRIMARY KEY'
    """

    # Obtener foreign keys
    # Las FKs le dicen a Claude:
    # "ascents.route_id referencia a routes.id"
    # Con eso Claude sabe hacer: JOIN routes ON ascents.route_id = routes.id
    fk_query = """
        SELECT
            kcu.column_name,
            ccu.table_name AS foreign_table,
            ccu.column_name AS foreign_column
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu 
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu 
            ON tc.constraint_name = ccu.constraint_name
            AND tc.table_schema = ccu.table_schema
        WHERE tc.table_name = $1 
        AND tc.table_schema = 'public'
        AND tc.constraint_type = 'FOREIGN KEY'
    """

    # Obtener CHECK constraints
    # Estos le dicen a Claude que valores son validos.
    # Ej: type IN ('sport', 'boulder', 'trad')
    check_query = """
        SELECT
            cc.check_clause
        FROM information_schema.table_constraints tc
        JOIN information_schema.check_constraints cc 
            ON tc.constraint_name = cc.constraint_name
            AND tc.constraint_schema = cc.constraint_schema
        WHERE tc.table_name = $1 
        AND tc.table_schema = 'public'
        AND tc.constraint_type = 'CHECK'
    """

    async with pool.acquire() as conn:
        columns = await conn.fetch(columns_query, table_name)
        pks = await conn.fetch(pk_query, table_name)
        fks = await conn.fetch(fk_query, table_name)
        checks = await conn.fetch(check_query, table_name)

    if not columns:
        return {"error": f"Tabla no encontrada: '{table_name}'"}

    pk_names = [row["column_name"] for row in pks]

    result = {
        "table_name": table_name,
        "columns": [
            {
                "name": col["column_name"],
                "type": col["data_type"],
                # udt_name da el tipo mas especifico
                # (ej: "geometry" en vez de "USER-DEFINED")
                "pg_type": col["udt_name"],
                "nullable": col["is_nullable"] == "YES",
                "default": col["column_default"],
                "max_length": col["character_maximum_length"],
                "is_primary_key": col["column_name"] in pk_names,
            }
            for col in columns
        ],
        "primary_keys": pk_names,
        "foreign_keys": [
            {
                "column": fk["column_name"],
                "references_table": fk["foreign_table"],
                "references_column": fk["foreign_column"],
            }
            for fk in fks
        ],
        "check_constraints": [
            check["check_clause"] for check in checks
        ],
    }

    logger.info(
        f"  [Tool] -> {len(result['columns'])} columnas, "
        f"{len(result['primary_keys'])} PKs, "
        f"{len(result['foreign_keys'])} FKs"
    )
    return result


async def get_sample_data(table_name: str, n: int = 5) -> dict:
    """
    Retorna N rows de ejemplo de una tabla.
    
    Claude usa esto para entender los datos reales:
    - Que formato tienen los grados? ("5.10a", "V3")
    - Que valores tiene la columna 'style'? ("onsight", "flash")
    - Las fechas estan en que formato?
    
    Sin ver datos reales, Claude podria generar queries con
    valores que no existen en la DB.
    """
    logger.info(f"  [Tool] get_sample_data(table_name='{table_name}', n={n})")

    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
        return {"error": f"Nombre de tabla invalido: '{table_name}'"}

    # Limitar N para seguridad
    n = min(max(1, n), 20)

    # Para la tabla crags, incluimos las coordenadas de forma legible
    # en vez del dato binario de PostGIS
    if table_name == "crags":
        query = f"""
            SELECT id, name, region,
                   ST_X(geom) as longitude,
                   ST_Y(geom) as latitude,
                   created_at
            FROM crags LIMIT {n}
        """
    else:
        query = f"SELECT * FROM {table_name} LIMIT {n}"

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(query)
    except asyncpg.exceptions.UndefinedTableError:
        return {"error": f"Tabla no encontrada: '{table_name}'"}

    if not rows:
        return {"table": table_name, "sample": [], "count": 0}

    sample = [row_to_dict(row) for row in rows]

    logger.info(f"  [Tool] -> {len(sample)} rows de ejemplo")
    return {"table": table_name, "sample": sample, "count": len(sample)}


async def query_database(sql: str) -> dict:
    """
    Ejecuta un query SELECT contra la base de datos.
    
    Este es el pipeline de validacion completo:
    
    1. Validar que sea SELECT (rechazar INSERT/UPDATE/DELETE/DROP)
       -> is_safe_query() parsea con sqlparse y busca keywords peligrosos
    
    2. Agregar LIMIT si no lo tiene (maximo 100 rows)
       -> Previene queries que retornen millones de rows
    
    3. Configurar timeout de 5 segundos
       -> Previene queries lentos que bloqueen la DB
       -> Un JOIN cartesiano entre dos tablas grandes puede tardar minutos
    
    4. Ejecutar con usuario read-only
       -> Incluso si las validaciones de arriba fallaran,
          PostgreSQL rechaza la operacion porque el usuario
          no tiene permisos de escritura
    
    5. Convertir resultados a JSON serializable
       -> asyncpg retorna Records con tipos de PostgreSQL
          que json.dumps no puede serializar
    """
    logger.info(f"  [Tool] query_database(sql='{sql[:150]}...')")

    # Paso 1: Validar seguridad
    is_safe, reason = is_safe_query(sql)
    if not is_safe:
        logger.warning(f"  [Tool] Query rechazado: {reason}")
        return {
            "error": f"Query no permitido: {reason}",
            "hint": "Solo se permiten queries SELECT. "
                    "Si necesitas ver la estructura, usa get_table_schema().",
        }

    # Paso 2: Agregar LIMIT
    sql = add_limit(sql, max_rows=100)

    # Paso 3+4: Ejecutar con timeout y usuario read-only
    try:
        async with pool.acquire() as conn:
            # statement_timeout limita la ejecucion del query en PostgreSQL.
            # Si el query tarda mas de 5 segundos, PostgreSQL lo cancela
            # y lanza QueryCanceledError.
            await conn.execute("SET statement_timeout = '5000'")
            rows = await conn.fetch(sql)

    except asyncpg.exceptions.QueryCanceledError:
        return {
            "error": "Query cancelado: excedio el timeout de 5 segundos.",
            "hint": "Intenta un query mas especifico o con filtros. "
                    "Evita JOINs entre tablas grandes sin WHERE.",
        }
    except asyncpg.exceptions.UndefinedTableError as e:
        return {
            "error": f"Tabla no encontrada: {e}",
            "hint": "Usa list_tables() para ver las tablas disponibles.",
        }
    except asyncpg.exceptions.UndefinedColumnError as e:
        return {
            "error": f"Columna no encontrada: {e}",
            "hint": "Usa get_table_schema(table_name) para ver las columnas.",
        }
    except asyncpg.exceptions.PostgresSyntaxError as e:
        return {
            "error": f"Error de sintaxis SQL: {e}",
            "hint": "Verifica la sintaxis. Esta DB usa PostgreSQL 15.",
        }
    except asyncpg.exceptions.InsufficientPrivilegeError:
        return {
            "error": "Permisos insuficientes. Solo se permiten lecturas.",
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    # Paso 5: Convertir resultados
    results = [row_to_dict(row) for row in rows]

    logger.info(f"  [Tool] -> {len(results)} rows retornados")

    return {
        "sql": sql,
        "rows": results,
        "row_count": len(results),
    }


# =====================================================================
# Tool Definitions (JSON Schema para Claude)
# =====================================================================

TOOLS = [
    {
        "name": "list_tables",
        "description": (
            "Lists all tables in the database. Use this first to understand "
            "what data is available. Returns table names."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_table_schema",
        "description": (
            "Returns the schema of a specific table: column names, types, "
            "primary keys, foreign keys, and check constraints. Use this "
            "before writing queries to understand the table structure and "
            "know which columns exist and how tables relate to each other. "
            "Foreign keys tell you how to JOIN tables. "
            "Example: get_table_schema(table_name='ascents')"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "Name of the table to inspect.",
                },
            },
            "required": ["table_name"],
        },
    },
    {
        "name": "get_sample_data",
        "description": (
            "Returns N sample rows from a table. Use this to see actual "
            "data values, understand formats (grade formats like '5.10a' "
            "or 'V3', date formats, style values like 'onsight'), and "
            "verify your understanding of the data before writing queries. "
            "Example: get_sample_data(table_name='routes', n=5)"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "Name of the table.",
                },
                "n": {
                    "type": "integer",
                    "description": "Number of sample rows (default: 5, max: 20).",
                    "default": 5,
                },
            },
            "required": ["table_name"],
        },
    },
    {
        "name": "query_database",
        "description": (
            "Executes a SELECT query on the PostgreSQL database and returns "
            "results. ONLY SELECT queries are allowed. The database has PostGIS "
            "extension installed for geospatial queries.\n\n"
            "PostGIS tips:\n"
            "- ST_MakePoint(longitude, latitude) -- order is (lon, lat)\n"
            "- ST_DWithin(geom::geography, point::geography, meters) -- radius search\n"
            "- ST_Distance(geom::geography, point::geography) -- distance in meters\n"
            "- Always cast to ::geography for real-world distances in meters\n"
            "- Puebla coordinates: longitude=-98.2063, latitude=19.0414\n\n"
            "Results are automatically limited to 100 rows.\n"
            "Example: query_database(sql='SELECT name, grade FROM routes LIMIT 10')"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": (
                        "SQL SELECT query to execute. Only SELECT is allowed. "
                        "Use PostgreSQL syntax. PostGIS functions are available."
                    ),
                },
            },
            "required": ["sql"],
        },
    },
]


# =====================================================================
# Tool Executor
# =====================================================================

# Mapeo de nombre de tool -> funcion.
# Todas las funciones son async, las envolvemos para manejar
# la ejecucion desde el agent loop sincrono.
TOOL_FUNCTIONS = {
    "list_tables": list_tables,
    "get_table_schema": get_table_schema,
    "get_sample_data": get_sample_data,
    "query_database": query_database,
}

tool_call_counts = {}
MAX_CALLS_PER_TOOL = 20


async def execute_tool(name: str, input_data: dict) -> str:
    """
    Ejecuta una tool de forma segura.
    
    Todas las validaciones pasan por aqui:
    1. Whitelist (solo tools conocidas)
    2. Rate limiting (maximo N llamadas por tool)
    3. Ejecucion con error handling
    """
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return json.dumps({
            "error": f"Tool desconocida: '{name}'",
            "available": list(TOOL_FUNCTIONS.keys()),
        })

    count = tool_call_counts.get(name, 0)
    if count >= MAX_CALLS_PER_TOOL:
        return json.dumps({
            "error": f"Rate limit: maximo {MAX_CALLS_PER_TOOL} llamadas a '{name}'"
        })
    tool_call_counts[name] = count + 1

    try:
        # Todas las tool functions son async
        result = await func(**input_data)
        return json.dumps(result, ensure_ascii=False, default=str)
    except TypeError as e:
        return json.dumps({"error": f"Parametros invalidos: {e}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


# =====================================================================
# System Prompt
# =====================================================================

SYSTEM_PROMPT = """You are a database analyst for Klimbook, a social network 
for rock climbers. You have access to a PostgreSQL database with PostGIS.

Strategy for answering questions:
1. Start with list_tables() to see what tables exist
2. Use get_table_schema() to understand table structure and relationships
3. Optionally use get_sample_data() to see actual data values
4. Write and execute SELECT queries with query_database()
5. Interpret the results and give a clear answer

Important rules:
- Always check the schema before writing queries
- Only use column names that actually exist in the schema
- Foreign keys tell you how to JOIN tables
- For geographic queries, use PostGIS functions:
  - ST_MakePoint(longitude, latitude) -- note: lon first, lat second
  - ST_DWithin(geom::geography, point::geography, distance_meters)
  - ST_Distance(geom::geography, point::geography) returns meters
  - Puebla, Mexico coordinates: lon=-98.2063, lat=19.0414
- Always cast geometry to ::geography for real-world distances
- Results are limited to 100 rows automatically
- If a query fails, read the error message and try to fix it

When presenting results:
- Be concise and clear
- Include specific numbers and data
- For rankings or top-N queries, present as a numbered list
- For geographic queries, include distances"""


# =====================================================================
# Agent Loop
# =====================================================================

@dataclass
class AgentConfig:
    model: str = MODEL
    max_iterations: int = 15  # DB agents suelen necesitar mas steps
    max_token_budget: int = 80_000
    max_consecutive_errors: int = 3
    verbose: bool = True


@dataclass
class AgentResult:
    answer: str
    iterations: int
    total_tokens: int
    total_cost: float
    elapsed_seconds: float
    tool_calls: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stopped_reason: str = "completed"


def log_agent_step(iteration: int, response):
    """Imprime un resumen de cada step."""
    print(f"\n  ------- Step {iteration} -------")
    for block in response.content:
        if block.type == "text" and block.text.strip():
            text = block.text.strip()
            if len(text) > 400:
                text = text[:400] + "..."
            print(f"  [Thinking] {text}")
        elif block.type == "tool_use":
            input_str = json.dumps(block.input, ensure_ascii=False)
            if len(input_str) > 200:
                input_str = input_str[:200] + "..."
            print(f"  [Tool] {block.name}({input_str})")
    print(f"  [Stop] {response.stop_reason}")
    print(f"  [Tokens] in={response.usage.input_tokens}, "
          f"out={response.usage.output_tokens}")


async def run_agent(
    message: str,
    config: AgentConfig = AgentConfig(),
) -> AgentResult:
    """
    Agent loop para el database agent.
    
    El flujo tipico de un database agent es:
    
    Step 1: list_tables()                 -> conocer las tablas
    Step 2: get_table_schema("ascents")   -> conocer columnas y FKs
    Step 3: get_table_schema("routes")    -> conocer la otra tabla
    Step 4: query_database("SELECT ...")  -> ejecutar la consulta
    Step 5: respuesta final al usuario
    
    Son ~5 steps en promedio, por eso max_iterations es 15
    (para dar margen a errores y retries).
    """
    messages = [{"role": "user", "content": message}]
    total_tokens = 0
    total_cost = 0.0
    consecutive_errors = 0
    call_history = []
    tool_call_log = []
    error_log = []
    start = time.time()

    logger.info(f"\n{'='*60}")
    logger.info(f"[Agent] Pregunta: \"{message[:120]}\"")
    logger.info(f"{'='*60}")

    for iteration in range(config.max_iterations):

        response = client.messages.create(
            model=config.model,
            max_tokens=4096,
            tools=TOOLS,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        step_tokens = response.usage.input_tokens + response.usage.output_tokens
        total_tokens += step_tokens
        step_cost = (
            response.usage.input_tokens / 1e6 * 3.0
            + response.usage.output_tokens / 1e6 * 15.0
        )
        total_cost += step_cost

        if config.verbose:
            log_agent_step(iteration + 1, response)

        # Token budget
        if total_tokens > config.max_token_budget:
            logger.warning(f"[Agent] Token budget excedido: {total_tokens}")
            return AgentResult(
                answer="Presupuesto de tokens excedido.",
                iterations=iteration + 1,
                total_tokens=total_tokens, total_cost=total_cost,
                elapsed_seconds=time.time() - start,
                tool_calls=tool_call_log, errors=error_log,
                stopped_reason="token_budget",
            )

        # Respuesta final
        if response.stop_reason == "end_turn":
            final_text = ""
            for block in response.content:
                if block.type == "text":
                    final_text += block.text

            elapsed = time.time() - start
            logger.info(
                f"\n[Agent] Completado en {iteration + 1} steps | "
                f"{elapsed:.2f}s | ${total_cost:.6f}"
            )

            return AgentResult(
                answer=final_text,
                iterations=iteration + 1,
                total_tokens=total_tokens, total_cost=total_cost,
                elapsed_seconds=elapsed,
                tool_calls=tool_call_log,
                stopped_reason="completed",
            )

        # Tool use
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                # Loop detection
                sig = f"{block.name}:{json.dumps(block.input, sort_keys=True)}"
                if sig in call_history:
                    logger.warning(f"  [Agent] Loop detectado: {sig}")
                    error_log.append(f"Loop: {sig}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Error: ya hiciste esta misma llamada. "
                                   "Usa la informacion que ya tienes.",
                        "is_error": True,
                    })
                    continue
                call_history.append(sig)

                # Ejecutar tool (async)
                logger.info(f"  Ejecutando {block.name}...")
                result_str = await execute_tool(block.name, block.input)

                # Log truncado
                display = result_str[:250] + "..." if len(result_str) > 250 else result_str
                logger.info(f"  Resultado: {display}")

                tool_call_log.append({
                    "iteration": iteration + 1,
                    "tool": block.name,
                    "input": block.input,
                    "output_preview": result_str[:400],
                })

                # Error tracking
                is_error = False
                try:
                    parsed = json.loads(result_str)
                    if isinstance(parsed, dict) and "error" in parsed:
                        is_error = True
                        consecutive_errors += 1
                        error_log.append(f"{block.name}: {parsed['error'][:200]}")
                    else:
                        consecutive_errors = 0
                except json.JSONDecodeError:
                    consecutive_errors = 0

                if consecutive_errors >= config.max_consecutive_errors:
                    logger.error("[Agent] Demasiados errores consecutivos")
                    return AgentResult(
                        answer="Demasiados errores consecutivos en la base de datos.",
                        iterations=iteration + 1,
                        total_tokens=total_tokens, total_cost=total_cost,
                        elapsed_seconds=time.time() - start,
                        tool_calls=tool_call_log, errors=error_log,
                        stopped_reason="error_threshold",
                    )

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                    "is_error": is_error,
                })

            messages.append({"role": "user", "content": tool_results})

    # Max iterations
    return AgentResult(
        answer="No pude completar la consulta en el numero maximo de pasos.",
        iterations=config.max_iterations,
        total_tokens=total_tokens, total_cost=total_cost,
        elapsed_seconds=time.time() - start,
        tool_calls=tool_call_log, errors=error_log,
        stopped_reason="max_iterations",
    )


# =====================================================================
# Print Result
# =====================================================================

def print_result(result: AgentResult):
    """Imprime el resultado de forma legible."""
    print(f"\n{'='*60}")
    print(f"  RESULTADO")
    print(f"{'='*60}")
    print(f"\n{result.answer}")
    print(f"\n{'-'*60}")
    print(f"  Metricas:")
    print(f"     Steps:       {result.iterations}")
    print(f"     Tokens:      {result.total_tokens:,}")
    print(f"     Costo:       ${result.total_cost:.6f}")
    print(f"     Tiempo:      {result.elapsed_seconds:.2f}s")
    print(f"     Tool calls:  {len(result.tool_calls)}")
    print(f"     Errors:      {len(result.errors)}")
    print(f"     Stop reason: {result.stopped_reason}")

    if result.tool_calls:
        print(f"\n  Tool calls:")
        for tc in result.tool_calls:
            input_str = json.dumps(tc["input"], ensure_ascii=False)
            if len(input_str) > 80:
                input_str = input_str[:80] + "..."
            print(f"     Step {tc['iteration']}: {tc['tool']}({input_str})")

    if result.errors:
        print(f"\n  Errores:")
        for err in result.errors:
            print(f"     {err}")


# =====================================================================
# Entry Point
# =====================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Klimbook Database Agent")
    parser.add_argument(
        "--test", action="store_true",
        help="Ejecutar preguntas de prueba automaticas",
    )
    args = parser.parse_args()

    # Inicializar pool de conexiones
    await init_pool()

    try:
        if args.test:
            # ---- Modo test ----
            print("=" * 60)
            print("  KLIMBOOK DATABASE AGENT -- Test Mode")
            print("=" * 60)

            QUESTIONS = [
                # Queries basicos
                "Cuantos usuarios hay registrados en total?",
                "Cuantos usuarios se registraron este mes?",

                # Queries con JOIN
                "Cual es la ruta mas repetida? Dame el nombre, grado, y cuantos ascensos tiene.",
                "Quienes son los top 10 climbers por cantidad de ascensos?",

                # Queries con agregacion
                "Cuantas rutas hay de cada tipo (sport, boulder, trad)?",
                "Cual es el estilo de ascenso mas comun (onsight, flash, redpoint, repeat)?",

                # PostGIS
                "Que crags estan a menos de 100km de Puebla? Incluye la distancia.",
            ]

            for i, question in enumerate(QUESTIONS, 1):
                print(f"\n{'#'*60}")
                print(f"  Pregunta {i}/{len(QUESTIONS)}: \"{question}\"")
                print(f"{'#'*60}")

                tool_call_counts.clear()
                result = await run_agent(question)
                print_result(result)

        else:
            # ---- Modo interactivo ----
            print("=" * 60)
            print("  KLIMBOOK DATABASE AGENT (Interactive)")
            print("  Escribe 'salir' para terminar")
            print("=" * 60)

            while True:
                question = input("\nPregunta: ").strip()

                if not question:
                    continue

                if question.lower() in ("salir", "quit", "exit"):
                    print("Hasta luego!")
                    break

                tool_call_counts.clear()
                result = await run_agent(question)
                print_result(result)

    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())