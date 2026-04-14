"""
ejercicio6_1.py 
===========================
Ejercicio 6.1 — File explorer agent


Description:
-----------
- Tools: list_directory(path), read_file(path, max_lines), search_text(pattern, directory)
- Implementar el agent loop completo con max_iterations=10 y verbose logging
- Tarea: "Encuentra todos los archivos Python en este proyecto que importen 
  FastAPI y dime qué endpoints definen"
- El agente debe razonar: primero listar, luego buscar, luego leer los archivos 
  relevantes, y sintetizar
- Seguridad: restringir paths a un directorio específico (no permitir acceso a 
  /etc, /home, etc.)


Metadata:
----------
* Author: zxxz6 
* Version: 1.0.0


History:
------------
Author      Date            Description
zxxz6       01/03/2026      Creation

"""

from anthropic import Anthropic
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import logging
import os
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("file-explorer-agent")

client = Anthropic()

MODEL = "claude-sonnet-4-20250514"


# ═══════════════════════════════════════════════════════════════════════════
# Agent Config & Result
# ═══════════════════════════════════════════════════════════════════════════
# Usamos dataclasses para tener configuracion y resultado tipados.
# AgentConfig centraliza todos los limites en un solo lugar.
# AgentResult captura metricas completas para observabilidad.

@dataclass
class AgentConfig:
    model: str = MODEL
    max_iterations: int = 10          # Limite duro de pasos del agent loop
    max_token_budget: int = 50_000    # Limite de costo total en tokens
    max_consecutive_errors: int = 3   # Abort si N tools fallan seguidas
    verbose: bool = True              # Loguear cada step


@dataclass
class AgentResult:
    answer: str                                            # Respuesta final del agente
    iterations: int                                        # Cuantos steps tomo
    total_tokens: int                                      # Tokens consumidos (input + output)
    total_cost: float                                      # Costo estimado en USD
    elapsed_seconds: float                                 # Tiempo total de ejecucion
    tool_calls: list[dict] = field(default_factory=list)   # Log de cada tool call
    errors: list[str] = field(default_factory=list)        # Errores acumulados
    stopped_reason: str = "completed"                      # Por que se detuvo


# ═══════════════════════════════════════════════════════════════════════════
# Security: Path Sandbox
# ═══════════════════════════════════════════════════════════════════════════

class PathSandbox:
    """
    Restringe el acceso a archivos dentro de un directorio permitido.
    Previene path traversal (../../etc/passwd) y acceso fuera del sandbox.
    
    Esto es CRITICO para seguridad: sin sandbox, Claude podria pedirle
    al agente que lea /etc/passwd, ~/.ssh/id_rsa, o cualquier archivo
    del sistema. El sandbox asegura que TODOS los paths resueltos
    esten dentro del directorio raiz permitido.
    """

    def __init__(self, allowed_root: str):
        self.root = Path(allowed_root).resolve()
        if not self.root.exists():
            raise ValueError(f"Directorio raíz no existe: {self.root}")
        if not self.root.is_dir():
            raise ValueError(f"La raíz no es un directorio: {self.root}")
        logger.info(f"[Sandbox] Raíz: {self.root}")

    def validate(self, path: str) -> Path:
        """
        Valida y resuelve un path asegurando que esté dentro del sandbox.
        
        Args:
            path: Path relativo o absoluto proporcionado por Claude
            
        Returns:
            Path resuelto y validado
            
        Raises:
            PermissionError: Si el path está fuera del sandbox
        """
        # Claude puede enviar paths absolutos o relativos.
        # .resolve() convierte symlinks y "../" a paths reales,
        # lo que previene ataques de path traversal como "../../etc/passwd"
        if Path(path).is_absolute():
            resolved = Path(path).resolve()
        else:
            resolved = (self.root / path).resolve()

        # relative_to() lanza ValueError si resolved NO esta dentro de self.root.
        # Esto es la verificacion central de seguridad del sandbox.
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise PermissionError(
                f"Acceso denegado: '{path}' esta fuera del directorio permitido. "
                f"Solo se permite acceso dentro de: {self.root}"
            )

        return resolved

    def relative(self, path: Path) -> str:
        """Retorna el path relativo al sandbox para display."""
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# Tool Functions
# ═══════════════════════════════════════════════════════════════════════════

# El sandbox se inicializa en main().
# Se usa como variable global porque todas las tool functions lo necesitan,
# y las tool functions deben tener la firma exacta que Claude espera
# (no podemos pasarle el sandbox como argumento extra).
sandbox: PathSandbox = None


def list_directory(path: str = ".") -> dict:
    """
    Lista el contenido de un directorio.
    
    Args:
        path: Ruta del directorio (relativa al sandbox)
        
    Returns:
        Dict con archivos y subdirectorios
    """
    try:
        resolved = sandbox.validate(path)
    except PermissionError as e:
        return {"error": str(e)}

    if not resolved.exists():
        return {"error": f"Directorio no encontrado: {path}"}
    if not resolved.is_dir():
        return {"error": f"No es un directorio: {path}"}

    entries = {"directories": [], "files": []}

    try:
        for entry in sorted(resolved.iterdir()):
            # Ignorar archivos ocultos y directorios comunes
            if entry.name.startswith(".") or entry.name in (
                "node_modules", "__pycache__", ".git", "venv", ".venv",
                "env", ".env", "dist", "build", ".mypy_cache", ".pytest_cache",
            ):
                continue

            rel_path = sandbox.relative(entry)

            if entry.is_dir():
                # Contar archivos dentro del subdirectorio
                try:
                    child_count = len(list(entry.iterdir()))
                except PermissionError:
                    child_count = "?"
                entries["directories"].append({
                    "name": entry.name,
                    "path": rel_path,
                    "items": child_count,
                })
            elif entry.is_file():
                entries["files"].append({
                    "name": entry.name,
                    "path": rel_path,
                    "size_bytes": entry.stat().st_size,
                    "extension": entry.suffix,
                })
    except PermissionError as e:
        return {"error": f"Permiso denegado: {e}"}

    entries["total_directories"] = len(entries["directories"])
    entries["total_files"] = len(entries["files"])
    entries["current_path"] = sandbox.relative(resolved)

    logger.info(
        f"  [list_directory] {path} → "
        f"{entries['total_directories']} dirs, {entries['total_files']} files"
    )
    return entries


def read_file(path: str, max_lines: int = 100) -> dict:
    """
    Lee el contenido de un archivo.
    
    Args:
        path: Ruta del archivo (relativa al sandbox)
        max_lines: Máximo de líneas a leer (default: 100)
        
    Returns:
        Dict con contenido del archivo
    """
    try:
        resolved = sandbox.validate(path)
    except PermissionError as e:
        return {"error": str(e)}

    if not resolved.exists():
        return {"error": f"Archivo no encontrado: {path}"}
    if not resolved.is_file():
        return {"error": f"No es un archivo: {path}"}

    # Limitar max_lines
    max_lines = min(max(1, max_lines), 500)

    # Verificar tamaño
    size = resolved.stat().st_size
    if size > 1_000_000:  # 1MB
        return {
            "error": f"Archivo muy grande ({size:,} bytes). "
                     "Usa search_text para buscar contenido específico."
        }

    try:
        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line.rstrip("\n"))

        total_lines_in_file = sum(1 for _ in open(resolved, "r", errors="replace"))
        truncated = total_lines_in_file > max_lines

        content = "\n".join(lines)

        logger.info(
            f"  [read_file] {path} → {len(lines)} lines "
            f"{'(truncated)' if truncated else ''}"
        )

        return {
            "path": sandbox.relative(resolved),
            "content": content,
            "lines_read": len(lines),
            "total_lines": total_lines_in_file,
            "truncated": truncated,
            "size_bytes": size,
        }
    except Exception as e:
        return {"error": f"Error leyendo archivo: {type(e).__name__}: {e}"}


def search_text(pattern: str, directory: str = ".") -> dict:
    """
    Busca un patrón de texto en archivos dentro de un directorio.
    
    Args:
        pattern: Texto o regex a buscar
        directory: Directorio donde buscar (relativo al sandbox)
        
    Returns:
        Dict con archivos y líneas que coinciden
    """
    try:
        resolved_dir = sandbox.validate(directory)
    except PermissionError as e:
        return {"error": str(e)}

    if not resolved_dir.exists():
        return {"error": f"Directorio no encontrado: {directory}"}
    if not resolved_dir.is_dir():
        return {"error": f"No es un directorio: {directory}"}

    # Compilar el regex
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return {"error": f"Patrón regex inválido: {e}"}

    # Extensiones a buscar (solo archivos de texto)
    TEXT_EXTENSIONS = {
        ".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".txt", ".yaml", ".yml",
        ".json", ".toml", ".cfg", ".ini", ".html", ".css", ".sql", ".sh",
        ".dockerfile", ".env.example", ".gitignore",
    }

    # Directorios a ignorar
    SKIP_DIRS = {
        "node_modules", "__pycache__", ".git", "venv", ".venv",
        "env", ".env", "dist", "build", ".mypy_cache", ".pytest_cache",
    }

    matches = []
    files_searched = 0
    max_matches = 50  # Limitar resultados

    for root, dirs, files in os.walk(resolved_dir):
        # Filtrar directorios a ignorar
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]

        for filename in files:
            filepath = Path(root) / filename

            # Verificar que esté en el sandbox
            try:
                sandbox.validate(str(filepath))
            except PermissionError:
                continue

            # Solo archivos de texto
            if filepath.suffix.lower() not in TEXT_EXTENSIONS and filepath.name not in TEXT_EXTENSIONS:
                continue

            # Saltar archivos grandes
            try:
                if filepath.stat().st_size > 500_000:
                    continue
            except OSError:
                continue

            files_searched += 1

            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append({
                                "file": sandbox.relative(filepath),
                                "line_number": line_num,
                                "line": line.strip()[:200],  # Limitar longitud
                            })
                            if len(matches) >= max_matches:
                                break
            except (OSError, UnicodeDecodeError):
                continue

            if len(matches) >= max_matches:
                break

        if len(matches) >= max_matches:
            break

    # Agrupar por archivo
    files_with_matches = {}
    for match in matches:
        fname = match["file"]
        if fname not in files_with_matches:
            files_with_matches[fname] = []
        files_with_matches[fname].append({
            "line_number": match["line_number"],
            "line": match["line"],
        })

    result = {
        "pattern": pattern,
        "directory": sandbox.relative(resolved_dir),
        "files_searched": files_searched,
        "total_matches": len(matches),
        "files_with_matches": len(files_with_matches),
        "truncated": len(matches) >= max_matches,
        "matches": files_with_matches,
    }

    logger.info(
        f"  [search_text] '{pattern}' in {directory} → "
        f"{len(matches)} matches in {len(files_with_matches)} files "
        f"({files_searched} files searched)"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Tool Definitions
# ═══════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "list_directory",
        "description": (
            "Lists the contents of a directory, showing files and subdirectories "
            "with their sizes. Use this to explore the project structure. "
            "Automatically skips hidden files, node_modules, __pycache__, etc. "
            "Start with '.' to list the root of the project. "
            "Example: list_directory(path='src') or list_directory(path='.')"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Directory path relative to the project root. "
                        "Use '.' for the root directory. "
                        "Examples: '.', 'src', 'src/services', 'tests'"
                    ),
                    "default": ".",
                },
            },
            "required": [],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Reads the content of a file and returns its text. "
            "Use this to inspect source code, configuration files, or documentation. "
            "Limited to 500 lines and 1MB file size. "
            "Example: read_file(path='src/main.py', max_lines=50)"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the project root.",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (default: 100, max: 500).",
                    "default": 100,
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_text",
        "description": (
            "Searches for a text pattern (regex supported) across all text files "
            "in a directory recursively. Returns matching lines with file paths "
            "and line numbers. Limited to 50 matches. "
            "Use this to find imports, function definitions, class names, etc. "
            "Example: search_text(pattern='from fastapi', directory='src')"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "Text or regex pattern to search for (case-insensitive). "
                        "Examples: 'from fastapi', 'def.*endpoint', '@router\\.(get|post)'"
                    ),
                },
                "directory": {
                    "type": "string",
                    "description": (
                        "Directory to search in, relative to project root. "
                        "Default: '.' (entire project)"
                    ),
                    "default": ".",
                },
            },
            "required": ["pattern"],
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Tool Executor
# ═══════════════════════════════════════════════════════════════════════════

# Whitelist: SOLO estas funciones pueden ser invocadas por Claude.
# Si Claude inventa un nombre de tool (hallucination), se rechaza.
TOOL_FUNCTIONS = {
    "list_directory": list_directory,
    "read_file": read_file,
    "search_text": search_text,
}

# Rate limiting por tool: evita que Claude llame la misma tool
# demasiadas veces (ej. leer 50 archivos uno por uno).
tool_call_counts = {}
MAX_CALLS_PER_TOOL = 20


def execute_tool(name: str, input_data: dict) -> str:
    """
    Ejecuta una tool con whitelist y rate limiting.
    
    Flujo:
    1. Verificar que el nombre este en la whitelist
    2. Verificar rate limit (max llamadas por tool)
    3. Ejecutar la funcion con los parametros que Claude envio
    4. Retornar resultado como JSON string (la API espera strings)
    
    Retorna siempre un JSON string, nunca lanza excepciones.
    """
    # 1. Whitelist: rechazar tools que no existen
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return json.dumps({"error": f"Tool desconocida: {name}"})

    # 2. Rate limit: evitar abuso
    count = tool_call_counts.get(name, 0)
    if count >= MAX_CALLS_PER_TOOL:
        return json.dumps({"error": f"Rate limit: maximo {MAX_CALLS_PER_TOOL} llamadas a '{name}'"})
    tool_call_counts[name] = count + 1

    # 3. Ejecutar: **input_data desempaqueta el dict como keyword args.
    #    Si Claude manda {"path": "src"}, se ejecuta func(path="src").
    #    TypeError ocurre si Claude manda parametros que la funcion no acepta.
    try:
        result = func(**input_data)
        return json.dumps(result, ensure_ascii=False)
    except TypeError as e:
        return json.dumps({"error": f"Parametros invalidos: {e}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


# ═══════════════════════════════════════════════════════════════════════════
# Agent Loop
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a senior software engineer exploring a codebase. 
You have access to tools that let you list directories, read files, and search 
for text patterns across the project.

Strategy for exploring codebases:
1. Start by listing the root directory to understand the project structure
2. Use search_text to find specific patterns (imports, decorators, function names)
3. Use read_file to inspect relevant files in detail
4. Synthesize your findings into a clear, structured answer

Rules:
- Be systematic: explore the structure first, then search, then read
- Don't read entire large files — use search_text to find relevant sections first
- If a search returns many results, focus on the most relevant ones
- Always explain your reasoning as you explore
- Give your final answer in a clear, organized format"""


def log_agent_step(iteration: int, response):
    """Pretty print de un step del agente."""
    print(f"\n  {'--'*28}")
    print(f"  Step {iteration}")
    print(f"  {'--'*28}")

    # response.content es una lista mixta de TextBlock y ToolUseBlock.
    # TextBlock = razonamiento de Claude ("Voy a buscar...")
    # ToolUseBlock = tool que Claude quiere ejecutar
    for block in response.content:
        if block.type == "text" and block.text.strip():
            text = block.text.strip()
            if len(text) > 300:
                text = text[:300] + "..."
            print(f"  Thinking: {text}")
        elif block.type == "tool_use":
            print(f"  Tool call: {block.name}({json.dumps(block.input)})")

    print(f"  Stop: {response.stop_reason}")
    print(f"  Tokens: in={response.usage.input_tokens}, out={response.usage.output_tokens}")


def run_agent(message: str, config: AgentConfig = AgentConfig()) -> AgentResult:
    """
    Agent loop robusto con todas las protecciones.
    
    Este es el CORE del agente. El flujo es:
    
    1. Enviar mensaje + tools a Claude
    2. Claude responde con stop_reason:
       - "end_turn" -> tiene la respuesta final, salir del loop
       - "tool_use" -> quiere ejecutar tools, procesarlas y volver a 1
    3. Repetir hasta que Claude termine o se alcance una condicion de parada
    
    Condiciones de parada implementadas:
    - Token budget: evita costos descontrolados
    - Max iterations: evita loops infinitos
    - Error threshold: aborta si las tools fallan repetidamente
    - Loop detection: detecta llamadas repetidas con mismos parametros
    """
    # El historial de mensajes crece con cada iteracion.
    # Cada step agrega 2 mensajes: assistant (respuesta de Claude) + 
    # user (tool_results). Esto significa que los input tokens crecen
    # linealmente con cada step.
    messages = [{"role": "user", "content": message}]
    total_tokens = 0
    total_cost = 0.0
    consecutive_errors = 0     # Se resetea a 0 en cada exito
    call_history = []          # Para loop detection: "name:json(input)"
    tool_call_log = []         # Log completo de tool calls para AgentResult
    error_log = []             # Errores acumulados para AgentResult
    start = time.time()

    logger.info(f"\n{'═'*60}")
    logger.info(f"[Agent] Tarea: \"{message[:100]}\"")
    logger.info(f"[Agent] Config: max_iter={config.max_iterations}, "
                f"max_tokens={config.max_token_budget}")
    logger.info(f"{'═'*60}")

    for iteration in range(config.max_iterations):
        # --- Llamar a Claude con el historial completo + tools ---
        # En cada iteracion, Claude recibe TODO el historial acumulado.
        # Por eso los input tokens crecen: step 1 = ~500, step 5 = ~3000+
        response = client.messages.create(
            model=config.model,
            max_tokens=4096,
            tools=TOOLS,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        # --- Metricas: acumular tokens y costo ---
        # Precios Sonnet: $3/M input, $15/M output
        step_tokens = response.usage.input_tokens + response.usage.output_tokens
        total_tokens += step_tokens
        step_cost = (response.usage.input_tokens / 1e6 * 3.0 +
                     response.usage.output_tokens / 1e6 * 15.0)
        total_cost += step_cost

        if config.verbose:
            log_agent_step(iteration + 1, response)

        # --- Condicion de parada: token budget ---
        # Si ya gastamos mas tokens de los permitidos, parar.
        if total_tokens > config.max_token_budget:
            logger.warning(f"[Agent] Token budget excedido: {total_tokens}")
            return AgentResult(
                answer="Presupuesto de tokens excedido.",
                iterations=iteration + 1,
                total_tokens=total_tokens,
                total_cost=total_cost,
                elapsed_seconds=time.time() - start,
                tool_calls=tool_call_log,
                errors=error_log,
                stopped_reason="token_budget",
            )

        # --- Condicion de parada: natural stop ---
        # "end_turn" significa que Claude tiene toda la info y genero
        # la respuesta final. Este es el caso ideal.
        if response.stop_reason == "end_turn":
            # Extraer solo los bloques de texto (ignorar cualquier otro tipo)
            final_text = ""
            for block in response.content:
                if block.type == "text":
                    final_text += block.text

            elapsed = time.time() - start
            logger.info(f"\n[Agent] Completado en {iteration + 1} steps | {elapsed:.2f}s")

            return AgentResult(
                answer=final_text,
                iterations=iteration + 1,
                total_tokens=total_tokens,
                total_cost=total_cost,
                elapsed_seconds=elapsed,
                tool_calls=tool_call_log,
                stopped_reason="completed",
            )

        # --- Procesar tool calls ---
        # "tool_use" significa que Claude NO termino -- quiere ejecutar
        # una o mas tools. Debemos:
        # 1. Guardar la respuesta de Claude en el historial (assistant)
        # 2. Ejecutar cada tool call
        # 3. Devolver los resultados como tool_result (user)
        # 4. Volver al inicio del loop para que Claude procese los resultados
        if response.stop_reason == "tool_use":
            # Paso 1: Guardar la respuesta completa de Claude.
            # response.content incluye TANTO el texto de razonamiento
            # como los ToolUseBlocks. Debemos guardar todo.
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                # Saltar bloques de texto (razonamiento), solo procesar tool calls
                if block.type != "tool_use":
                    continue

                # --- Loop detection ---
                # Crear firma unica: nombre + parametros serializados.
                # sort_keys=True es critico: sin esto,
                # {"a": 1, "b": 2} y {"b": 2, "a": 1} serian firmas
                # diferentes aunque sean la misma llamada.
                sig = f"{block.name}:{json.dumps(block.input, sort_keys=True)}"
                if sig in call_history:
                    logger.warning(f"  [Agent] Loop detectado: {sig}")
                    error_log.append(f"Loop: {sig}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Error: ya hiciste esta misma llamada antes. "
                                   "Usa la información que ya tienes para responder.",
                        "is_error": True,
                    })
                    continue
                call_history.append(sig)

                # --- Paso 2: Ejecutar la tool ---
                logger.info(f"  Ejecutando {block.name}...")
                result_str = execute_tool(block.name, block.input)

                # Log resultado (truncado para no saturar la consola)
                display = result_str[:200] + "..." if len(result_str) > 200 else result_str
                logger.info(f"  Resultado {block.name}: {display}")

                tool_call_log.append({
                    "iteration": iteration + 1,
                    "tool": block.name,
                    "input": block.input,
                    "output_preview": result_str[:300],
                })

                # --- Error tracking: contar errores consecutivos ---
                # Si la tool retorno un dict con key "error", es un error.
                # consecutive_errors se resetea a 0 en cada exito, asi que
                # solo se dispara el threshold si fallan N tools SEGUIDAS.
                is_error = False
                try:
                    parsed = json.loads(result_str)
                    if isinstance(parsed, dict) and "error" in parsed:
                        is_error = True
                        consecutive_errors += 1
                        error_log.append(f"{block.name}: {parsed['error'][:200]}")
                    else:
                        consecutive_errors = 0  # Reset en exito
                except json.JSONDecodeError:
                    consecutive_errors = 0

                if consecutive_errors >= config.max_consecutive_errors:
                    logger.error("[Agent] Demasiados errores consecutivos")
                    return AgentResult(
                        answer="Demasiados errores consecutivos en las herramientas.",
                        iterations=iteration + 1,
                        total_tokens=total_tokens,
                        total_cost=total_cost,
                        elapsed_seconds=time.time() - start,
                        tool_calls=tool_call_log,
                        errors=error_log,
                        stopped_reason="error_threshold",
                    )

                # Paso 3: Formatear el resultado para devolverlo a Claude.
                # tool_use_id DEBE coincidir con block.id, si no la API da error.
                # is_error=True le dice a Claude que la tool fallo,
                # asi puede decidir reintentar con otros parametros.
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                    "is_error": is_error,
                })

            # Paso 4: Devolver TODOS los resultados a Claude.
            # Se envian como role="user" porque en la API de Anthropic
            # los tool_results van en mensajes del usuario.
            messages.append({"role": "user", "content": tool_results})

    # --- Condicion de parada: max iterations ---
    logger.error("[Agent] Maximo de iteraciones alcanzado")
    return AgentResult(
        answer="No pude completar la tarea en el número máximo de pasos.",
        iterations=config.max_iterations,
        total_tokens=total_tokens,
        total_cost=total_cost,
        elapsed_seconds=time.time() - start,
        tool_calls=tool_call_log,
        errors=error_log,
        stopped_reason="max_iterations",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Print Result
# ═══════════════════════════════════════════════════════════════════════════

def print_result(result: AgentResult):
    """Pretty print del resultado del agente."""
    print(f"\n{'='*60}")
    print(f"  RESULTADO DEL AGENTE")
    print(f"{'='*60}")
    print(f"\n{result.answer}")
    print(f"\n{'--'*30}")
    print(f"  Metricas:")
    print(f"     Iterations:  {result.iterations}")
    print(f"     Tokens:      {result.total_tokens:,}")
    print(f"     Costo:       ${result.total_cost:.6f}")
    print(f"     Tiempo:      {result.elapsed_seconds:.2f}s")
    print(f"     Tool calls:  {len(result.tool_calls)}")
    print(f"     Errors:      {len(result.errors)}")
    print(f"     Stop reason: {result.stopped_reason}")

    if result.tool_calls:
        print(f"\n  Tool calls realizados:")
        for tc in result.tool_calls:
            print(f"     Step {tc['iteration']}: {tc['tool']}({json.dumps(tc['input'])})")

    if result.errors:
        print(f"\n  Errores:")
        for err in result.errors:
            print(f"     {err}")


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="File Explorer Agent")
    parser.add_argument(
        "--dir", type=str, default=".",
        help="Directorio raíz del proyecto a explorar (default: directorio actual)"
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Tarea para el agente (default: buscar archivos FastAPI)"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=10,
        help="Máximo de iteraciones del agente (default: 10)"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Modo interactivo: hacer múltiples preguntas"
    )
    args = parser.parse_args()

    # Inicializar sandbox
    sandbox = PathSandbox(args.dir)

    DEFAULT_TASK = (
        "Encuentra todos los archivos Python en este proyecto que importen FastAPI "
        "y dime qué endpoints definen. Para cada endpoint, indica: el método HTTP "
        "(GET, POST, etc.), la ruta, y una breve descripción basada en el código."
    )

    config = AgentConfig(
        max_iterations=args.max_iterations,
        verbose=True,
    )

    if args.interactive:
        # Modo interactivo
        print("=" * 60)
        print("  FILE EXPLORER AGENT (Interactive)")
        print(f"  Sandbox: {sandbox.root}")
        print("  Escribe 'salir' para terminar")
        print("=" * 60)

        while True:
            task = input("\nTarea: ").strip()
            if not task:
                continue
            if task.lower() in ("salir", "quit", "exit"):
                print("¡Hasta luego!")
                break

            tool_call_counts.clear()
            result = run_agent(task, config)
            print_result(result)
    else:
        # Modo single task
        task = args.task or DEFAULT_TASK

        print("=" * 60)
        print("  FILE EXPLORER AGENT")
        print(f"  Sandbox: {sandbox.root}")
        print(f"  Tarea: {task[:80]}...")
        print("=" * 60)

        result = run_agent(task, config)
        print_result(result)