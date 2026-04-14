"""
ejercicio6_2.py 
===========================
Ejercicio 6.2 — GitHub agent


Description:
-----------
- Tools: list_repos(user), get_repo_info(owner, repo), list_issues(owner, repo, state),
  get_issue(owner, repo, number), list_commits(owner, repo, since)
- Usar httpx para llamar a la GitHub API (rate limit: 60 req/hour sin auth, 5000 con token)
- Tareas de prueba: "Cuales son los 5 issues abiertos mas recientes de klimbook?",
  "Cuantos commits hubo esta semana?"
- Implementar error handling para rate limits de GitHub (429) separado del retry del LLM
- Bonus: agregar herramienta create_issue(owner, repo, title, body) pero con confirmacion
  del usuario antes de ejecutar


Metadata:
----------
* Author: zxxz6 
* Version: 1.0.0


History:
------------
Author      Date            Description
zxxz6       22/02/2026      Creation

"""

from anthropic import Anthropic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import httpx
import json
import time
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("github-agent")

client = Anthropic()

MODEL = "claude-sonnet-4-20250514"


# =====================================================================
# Agent Config & Result (misma estructura del ejercicio 6.1)
# =====================================================================

@dataclass
class AgentConfig:
    model: str = MODEL
    max_iterations: int = 10
    max_token_budget: int = 50_000
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


# =====================================================================
# GitHub API Client
# =====================================================================
#
# Esta clase encapsula todas las llamadas a la API de GitHub.
# Maneja autenticacion (opcional), rate limits, y reintentos.
#
# SIN token: 60 requests/hora (muy limitado)
# CON token: 5,000 requests/hora
#
# Para obtener un token:
# 1. Ve a github.com/settings/tokens
# 2. Genera un "Personal access token (classic)"
# 3. Exporta como: export GITHUB_TOKEN="ghp_..."
#
# =====================================================================

class GitHubClient:
    """
    Cliente para la API de GitHub con manejo de rate limits y reintentos.
    """

    BASE_URL = "https://api.github.com"

    def __init__(self, token: str | None = None):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        
        # Headers base para todas las requests
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "klimbook-github-agent",
        }
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"
            logger.info("[GitHub] Autenticado con token (5000 req/hora)")
        else:
            logger.info("[GitHub] Sin token (60 req/hora). "
                        "Exporta GITHUB_TOKEN para mas capacidad.")

        # httpx client con timeout de 15 segundos
        self.http = httpx.Client(
            base_url=self.BASE_URL,
            headers=self.headers,
            timeout=15.0,
        )

        # Contadores para monitorear el rate limit
        self.requests_made = 0
        self.rate_limit_remaining = None
        self.rate_limit_reset = None

    def request(self, method: str, path: str, **kwargs) -> dict:
        """
        Hace una request a la GitHub API con manejo de rate limits.

        Este metodo es el punto central por donde pasan TODAS las llamadas
        a GitHub. Maneja tres casos:
        
        1. Request exitosa (200) -> retorna los datos
        2. Rate limit (403/429) -> espera y reintenta
        3. Error (404, 500, etc.) -> retorna el error
        
        Args:
            method: "GET" o "POST"
            path: Ruta de la API (ej: "/repos/zxxz456/klimbook")
            **kwargs: Parametros adicionales para httpx (params, json, etc.)
            
        Returns:
            Dict con la respuesta parseada o un error
        """
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = self.http.request(method, path, **kwargs)
                self.requests_made += 1

                # ----- Extraer info de rate limit de los headers -----
                # GitHub siempre incluye estos headers en sus respuestas.
                # Los usamos para saber cuantas requests nos quedan.
                self.rate_limit_remaining = response.headers.get(
                    "X-RateLimit-Remaining"
                )
                self.rate_limit_reset = response.headers.get(
                    "X-RateLimit-Reset"
                )

                # Log del rate limit para monitoreo
                if self.rate_limit_remaining:
                    logger.info(
                        f"  [GitHub] Rate limit restante: "
                        f"{self.rate_limit_remaining} requests"
                    )

                # ----- Caso: Rate limit alcanzado -----
                # GitHub responde con 403 o 429 cuando excedes el limite.
                # El header X-RateLimit-Reset contiene el timestamp (Unix)
                # de cuando se reinicia el contador.
                if response.status_code in (403, 429):
                    reset_timestamp = self.rate_limit_reset
                    if reset_timestamp:
                        # Calcular cuantos segundos faltan para el reset
                        reset_time = datetime.fromtimestamp(int(reset_timestamp))
                        wait_seconds = (reset_time - datetime.now()).total_seconds()
                        wait_seconds = max(wait_seconds, 1)  # minimo 1 segundo
                    else:
                        # Si no hay header, usar backoff exponencial
                        # Intento 1: 10s, Intento 2: 20s, Intento 3: 40s
                        wait_seconds = 10 * (2 ** attempt)

                    logger.warning(
                        f"  [GitHub] Rate limit alcanzado (intento {attempt + 1}/{max_retries}). "
                        f"Esperando {wait_seconds:.0f}s..."
                    )

                    # Si la espera es demasiado larga, no esperar
                    if wait_seconds > 120:
                        return {
                            "error": "Rate limit de GitHub alcanzado. "
                                     f"Se reinicia en {wait_seconds:.0f}s. "
                                     "Intenta mas tarde o usa un GITHUB_TOKEN.",
                            "rate_limit_remaining": 0,
                        }

                    time.sleep(wait_seconds)
                    continue  # Reintentar despues de esperar

                # ----- Caso: Request exitosa -----
                if response.status_code == 200:
                    return response.json()

                # ----- Caso: Recurso creado (POST exitoso) -----
                if response.status_code == 201:
                    return response.json()

                # ----- Caso: Not Found -----
                if response.status_code == 404:
                    return {"error": f"No encontrado: {path}"}

                # ----- Caso: Otros errores -----
                return {
                    "error": f"GitHub API error {response.status_code}: "
                             f"{response.text[:200]}"
                }

            except httpx.TimeoutException:
                logger.warning(
                    f"  [GitHub] Timeout (intento {attempt + 1}/{max_retries})"
                )
                if attempt == max_retries - 1:
                    return {"error": "Timeout al conectar con GitHub API"}

            except httpx.HTTPError as e:
                logger.error(f"  [GitHub] Error HTTP: {e}")
                return {"error": f"Error de conexion: {type(e).__name__}: {e}"}

        return {"error": "Se agotaron los reintentos para la API de GitHub"}

    def get(self, path: str, **params) -> dict:
        """Shortcut para GET requests."""
        return self.request("GET", path, params=params)

    def post(self, path: str, data: dict) -> dict:
        """Shortcut para POST requests."""
        return self.request("POST", path, json=data)

    def close(self):
        """Cierra el cliente HTTP."""
        self.http.close()


# =====================================================================
# Tool Functions
# =====================================================================
#
# Cada funcion corresponde a una tool que Claude puede llamar.
# Todas reciben parametros simples y retornan un dict.
# El dict puede contener datos o un campo "error" si algo fallo.
#
# =====================================================================

# Se inicializa en main()
gh: GitHubClient = None


def list_repos(user: str) -> dict:
    """
    Lista los repositorios publicos de un usuario de GitHub.
    
    Args:
        user: Username de GitHub (ej: "zxxz456")
        
    Returns:
        Lista de repos con nombre, descripcion, lenguaje, y estrellas
    """
    logger.info(f"  [Tool] list_repos(user='{user}')")

    # La API de GitHub pagina resultados. per_page=30 es el default,
    # nosotros pedimos 100 (el maximo) para reducir llamadas.
    data = gh.get(f"/users/{user}/repos", per_page=100, sort="updated")

    if isinstance(data, dict) and "error" in data:
        return data

    # Transformar la respuesta cruda de GitHub a algo mas limpio
    # La API retorna ~80 campos por repo, solo necesitamos unos pocos.
    repos = []
    for repo in data:
        repos.append({
            "name": repo["name"],
            "full_name": repo["full_name"],
            "description": repo.get("description", ""),
            "language": repo.get("language", ""),
            "stars": repo.get("stargazers_count", 0),
            "forks": repo.get("forks_count", 0),
            "open_issues": repo.get("open_issues_count", 0),
            "updated_at": repo.get("updated_at", ""),
            "private": repo.get("private", False),
        })

    return {
        "user": user,
        "total_repos": len(repos),
        "repos": repos,
    }


def get_repo_info(owner: str, repo: str) -> dict:
    """
    Obtiene informacion detallada de un repositorio.
    
    Args:
        owner: Dueno del repo (ej: "zxxz456")
        repo: Nombre del repo (ej: "klimbook")
    """
    logger.info(f"  [Tool] get_repo_info(owner='{owner}', repo='{repo}')")

    data = gh.get(f"/repos/{owner}/{repo}")

    if isinstance(data, dict) and "error" in data:
        return data

    return {
        "name": data["name"],
        "full_name": data["full_name"],
        "description": data.get("description", ""),
        "language": data.get("language", ""),
        "stars": data.get("stargazers_count", 0),
        "forks": data.get("forks_count", 0),
        "open_issues": data.get("open_issues_count", 0),
        "default_branch": data.get("default_branch", "main"),
        "created_at": data.get("created_at", ""),
        "updated_at": data.get("updated_at", ""),
        "topics": data.get("topics", []),
        "license": data.get("license", {}).get("name", "None"),
        "size_kb": data.get("size", 0),
    }


def list_issues(owner: str, repo: str, state: str = "open") -> dict:
    """
    Lista issues de un repositorio.
    
    Args:
        owner: Dueno del repo
        repo: Nombre del repo
        state: "open", "closed", o "all"
    """
    logger.info(f"  [Tool] list_issues(owner='{owner}', repo='{repo}', state='{state}')")

    # Validar el parametro state
    if state not in ("open", "closed", "all"):
        return {"error": f"State invalido: '{state}'. Usa 'open', 'closed', o 'all'."}

    data = gh.get(
        f"/repos/{owner}/{repo}/issues",
        state=state,
        per_page=30,
        sort="created",
        direction="desc",
    )

    if isinstance(data, dict) and "error" in data:
        return data

    # Nota: la API de GitHub mezcla issues y pull requests en /issues.
    # Filtramos los PRs checando si tienen el campo "pull_request".
    issues = []
    for item in data:
        # Ignorar pull requests (tienen campo "pull_request")
        if "pull_request" in item:
            continue

        issues.append({
            "number": item["number"],
            "title": item["title"],
            "state": item["state"],
            "author": item["user"]["login"],
            "labels": [l["name"] for l in item.get("labels", [])],
            "created_at": item["created_at"],
            "updated_at": item["updated_at"],
            "comments": item.get("comments", 0),
            # Truncar body para no exceder tokens
            "body_preview": (item.get("body") or "")[:300],
        })

    return {
        "owner": owner,
        "repo": repo,
        "state": state,
        "total_returned": len(issues),
        "issues": issues,
    }


def get_issue(owner: str, repo: str, number: int) -> dict:
    """
    Obtiene los detalles completos de un issue especifico.
    
    Args:
        owner: Dueno del repo
        repo: Nombre del repo
        number: Numero del issue (ej: 42)
    """
    logger.info(f"  [Tool] get_issue(owner='{owner}', repo='{repo}', number={number})")

    data = gh.get(f"/repos/{owner}/{repo}/issues/{number}")

    if isinstance(data, dict) and "error" in data:
        return data

    # Obtener comentarios del issue (endpoint separado)
    comments_data = gh.get(
        f"/repos/{owner}/{repo}/issues/{number}/comments",
        per_page=10,
    )

    comments = []
    if isinstance(comments_data, list):
        for c in comments_data:
            comments.append({
                "author": c["user"]["login"],
                "body": (c.get("body") or "")[:500],
                "created_at": c["created_at"],
            })

    return {
        "number": data["number"],
        "title": data["title"],
        "state": data["state"],
        "author": data["user"]["login"],
        "labels": [l["name"] for l in data.get("labels", [])],
        "created_at": data["created_at"],
        "updated_at": data["updated_at"],
        "body": (data.get("body") or "")[:2000],
        "comments_count": data.get("comments", 0),
        "comments": comments,
    }


def list_commits(owner: str, repo: str, since: str = "") -> dict:
    """
    Lista commits recientes de un repositorio.
    
    Args:
        owner: Dueno del repo
        repo: Nombre del repo
        since: Fecha ISO 8601 desde la cual listar commits.
               Si esta vacio, lista los mas recientes.
               Ejemplo: "2026-02-15T00:00:00Z"
    """
    logger.info(
        f"  [Tool] list_commits(owner='{owner}', repo='{repo}', "
        f"since='{since or 'latest'}')"
    )

    params = {"per_page": 30}
    if since:
        params["since"] = since

    data = gh.get(f"/repos/{owner}/{repo}/commits", **params)

    if isinstance(data, dict) and "error" in data:
        return data

    commits = []
    for item in data:
        commit_info = item.get("commit", {})
        author_info = commit_info.get("author", {})
        commits.append({
            "sha": item["sha"][:7],  # solo los primeros 7 chars
            "message": commit_info.get("message", "")[:200],
            "author": author_info.get("name", "unknown"),
            "date": author_info.get("date", ""),
        })

    return {
        "owner": owner,
        "repo": repo,
        "since": since or "latest",
        "total_returned": len(commits),
        "commits": commits,
    }


def create_issue(owner: str, repo: str, title: str, body: str = "") -> dict:
    """
    Crea un nuevo issue en un repositorio.
    
    IMPORTANTE: Esta funcion pide confirmacion al usuario antes de ejecutar.
    Claude NO puede crear issues sin que el usuario lo apruebe explicitamente.
    
    Args:
        owner: Dueno del repo
        repo: Nombre del repo
        title: Titulo del issue
        body: Cuerpo/descripcion del issue (markdown)
    """
    logger.info(f"  [Tool] create_issue(owner='{owner}', repo='{repo}', title='{title}')")

    # ---------------------------------------------------------------
    # CONFIRMACION DEL USUARIO
    # ---------------------------------------------------------------
    # Esta es una operacion de escritura. No queremos que Claude
    # cree issues sin supervision. Mostramos los detalles y pedimos
    # confirmacion explicita antes de ejecutar.
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  CONFIRMACION REQUERIDA: Crear Issue en GitHub")
    print("=" * 60)
    print(f"  Repo:   {owner}/{repo}")
    print(f"  Titulo: {title}")
    if body:
        # Mostrar preview del body (primeras 5 lineas)
        body_lines = body.strip().split("\n")[:5]
        print(f"  Body:")
        for line in body_lines:
            print(f"    {line[:80]}")
        if len(body.strip().split("\n")) > 5:
            print(f"    ... ({len(body.strip().split(chr(10)))} lineas total)")
    print("=" * 60)

    confirmation = input("  Crear este issue? (si/no): ").strip().lower()

    if confirmation not in ("si", "s", "yes", "y"):
        logger.info("  [Tool] create_issue cancelado por el usuario")
        return {
            "status": "cancelled",
            "message": "El usuario cancelo la creacion del issue.",
        }

    # Verificar que tenemos token (POST requiere autenticacion)
    if not gh.token:
        return {
            "error": "Se requiere un GITHUB_TOKEN para crear issues. "
                     "Exporta tu token: export GITHUB_TOKEN='ghp_...'"
        }

    # Ejecutar la creacion
    data = gh.post(
        f"/repos/{owner}/{repo}/issues",
        data={"title": title, "body": body},
    )

    if isinstance(data, dict) and "error" in data:
        return data

    return {
        "status": "created",
        "number": data.get("number"),
        "title": data.get("title"),
        "url": data.get("html_url"),
        "message": f"Issue #{data.get('number')} creado exitosamente.",
    }


# =====================================================================
# Tool Definitions (JSON Schema para Claude)
# =====================================================================

TOOLS = [
    {
        "name": "list_repos",
        "description": (
            "Lists public repositories for a GitHub user, sorted by most "
            "recently updated. Returns name, description, language, stars, "
            "and open issues count for each repo. "
            "Example: list_repos(user='zxxz456')"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user": {
                    "type": "string",
                    "description": "GitHub username (e.g. 'zxxz456')",
                },
            },
            "required": ["user"],
        },
    },
    {
        "name": "get_repo_info",
        "description": (
            "Gets detailed information about a specific repository including "
            "stars, forks, language, topics, license, and dates. "
            "Example: get_repo_info(owner='zxxz456', repo='klimbook')"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner (e.g. 'zxxz456')",
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name (e.g. 'klimbook')",
                },
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "list_issues",
        "description": (
            "Lists issues for a repository. Filters out pull requests. "
            "Returns title, state, author, labels, and body preview. "
            "Use state='open' for active issues, 'closed' for resolved, "
            "'all' for both. Sorted by creation date, newest first. "
            "Example: list_issues(owner='zxxz456', repo='klimbook', state='open')"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner",
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name",
                },
                "state": {
                    "type": "string",
                    "enum": ["open", "closed", "all"],
                    "description": "Issue state filter (default: 'open')",
                },
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "get_issue",
        "description": (
            "Gets full details of a specific issue including body text and "
            "comments. Use this when you need the complete content of an "
            "issue, not just the title. "
            "Example: get_issue(owner='zxxz456', repo='klimbook', number=42)"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner",
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name",
                },
                "number": {
                    "type": "integer",
                    "description": "Issue number",
                },
            },
            "required": ["owner", "repo", "number"],
        },
    },
    {
        "name": "list_commits",
        "description": (
            "Lists recent commits for a repository. Can filter by date. "
            "Returns SHA (short), commit message, author, and date. "
            "For 'since', use ISO 8601 format: '2026-02-15T00:00:00Z'. "
            "If 'since' is empty, returns the most recent commits. "
            "Example: list_commits(owner='zxxz456', repo='klimbook', "
            "since='2026-02-15T00:00:00Z')"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner",
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name",
                },
                "since": {
                    "type": "string",
                    "description": (
                        "Only show commits after this date (ISO 8601). "
                        "Example: '2026-02-15T00:00:00Z'. "
                        "Leave empty for most recent commits."
                    ),
                },
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "create_issue",
        "description": (
            "Creates a new issue on a GitHub repository. REQUIRES user "
            "confirmation before executing. Use this only when the user "
            "explicitly asks to create an issue. Requires GITHUB_TOKEN. "
            "Example: create_issue(owner='zxxz456', repo='klimbook', "
            "title='Bug: login fails', body='Steps to reproduce...')"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner",
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name",
                },
                "title": {
                    "type": "string",
                    "description": "Issue title (concise and descriptive)",
                },
                "body": {
                    "type": "string",
                    "description": (
                        "Issue body in markdown. Include: description, "
                        "steps to reproduce (if bug), expected behavior, "
                        "and any relevant context."
                    ),
                },
            },
            "required": ["owner", "repo", "title"],
        },
    },
]


# =====================================================================
# Tool Executor
# =====================================================================

TOOL_FUNCTIONS = {
    "list_repos": list_repos,
    "get_repo_info": get_repo_info,
    "list_issues": list_issues,
    "get_issue": get_issue,
    "list_commits": list_commits,
    "create_issue": create_issue,
}

tool_call_counts = {}
MAX_CALLS_PER_TOOL = 15


def execute_tool(name: str, input_data: dict) -> str:
    """
    Ejecuta una tool con whitelist y rate limiting.
    
    Este es el punto de entrada unico para ejecutar tools.
    Claude nunca llama a las funciones directamente -- siempre
    pasa por aqui, donde aplicamos las validaciones.
    """
    # 1. Verificar que la tool existe (whitelist)
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return json.dumps({
            "error": f"Tool desconocida: '{name}'",
            "available_tools": list(TOOL_FUNCTIONS.keys()),
        })

    # 2. Rate limiting por tool
    count = tool_call_counts.get(name, 0)
    if count >= MAX_CALLS_PER_TOOL:
        return json.dumps({
            "error": f"Rate limit interno: maximo {MAX_CALLS_PER_TOOL} "
                     f"llamadas a '{name}' por conversacion"
        })
    tool_call_counts[name] = count + 1

    # 3. Ejecutar
    try:
        result = func(**input_data)
        return json.dumps(result, ensure_ascii=False, default=str)
    except TypeError as e:
        # TypeError ocurre cuando Claude pasa parametros que la funcion
        # no espera (ej: un parametro extra que no existe)
        return json.dumps({"error": f"Parametros invalidos: {e}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


# =====================================================================
# System Prompt
# =====================================================================
#
# Este prompt le da a Claude el contexto sobre Klimbook y le indica
# como usar las tools de forma eficiente.
#

SYSTEM_PROMPT = """You are a GitHub assistant for the Klimbook project, a social 
network for rock climbers. The main repository is owned by user 'zxxz456'.

You have access to GitHub API tools to query repositories, issues, and commits.

Strategy for answering questions:
1. Identify which tool(s) you need based on the question
2. Call the minimum number of tools necessary
3. If you need details about a specific issue, first list issues to find the 
   number, then get the full issue details
4. For time-based queries (like "this week"), calculate the correct ISO date

Important context:
- Klimbook repos: klimbook (backend), klimbook-mobile (React Native), 
  klimbook-web (React/Vite)
- Owner: zxxz456
- When asked about "klimbook" without specifying which repo, assume the user 
  means the main backend repo unless context suggests otherwise

Rules:
- Be concise in your answers
- If GitHub returns an error, explain it clearly
- For create_issue: ONLY use this when the user explicitly asks to create an issue
- Always mention the issue number and title when discussing issues
- For date calculations: today is the current date, "this week" means since 
  last Monday"""


# =====================================================================
# Agent Loop
# =====================================================================

def log_agent_step(iteration: int, response):
    """Imprime un resumen legible de cada step del agente."""
    print(f"\n  ------- Step {iteration} -------")

    for block in response.content:
        if block.type == "text" and block.text.strip():
            text = block.text.strip()
            if len(text) > 300:
                text = text[:300] + "..."
            print(f"  [Thinking] {text}")
        elif block.type == "tool_use":
            # Mostrar la tool call de forma legible
            input_str = json.dumps(block.input, ensure_ascii=False)
            if len(input_str) > 150:
                input_str = input_str[:150] + "..."
            print(f"  [Tool] {block.name}({input_str})")

    print(f"  [Stop] {response.stop_reason}")
    print(f"  [Tokens] in={response.usage.input_tokens}, "
          f"out={response.usage.output_tokens}")


def run_agent(message: str, config: AgentConfig = AgentConfig()) -> AgentResult:
    """
    Agent loop principal.
    
    El flujo es:
    1. Enviar mensaje del usuario a Claude (con las tool definitions)
    2. Si Claude responde con texto -> es la respuesta final, terminar
    3. Si Claude quiere usar tools -> ejecutar TODAS las tools que pidio
    4. Enviar los resultados de vuelta a Claude
    5. Volver al paso 2
    
    El loop continua hasta que:
    - Claude da una respuesta final (stop_reason="end_turn")
    - Se alcanzan las max_iterations
    - Se excede el token budget
    - Hay demasiados errores consecutivos
    """
    messages = [{"role": "user", "content": message}]
    total_tokens = 0
    total_cost = 0.0
    consecutive_errors = 0
    call_history = []  # Para detectar loops (misma tool + mismos params)
    tool_call_log = []
    error_log = []
    start = time.time()

    logger.info(f"\n{'='*60}")
    logger.info(f"[Agent] Tarea: \"{message[:100]}\"")
    logger.info(f"{'='*60}")

    for iteration in range(config.max_iterations):

        # -- Llamar a Claude --
        response = client.messages.create(
            model=config.model,
            max_tokens=4096,
            tools=TOOLS,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        # -- Actualizar metricas --
        step_tokens = response.usage.input_tokens + response.usage.output_tokens
        total_tokens += step_tokens
        # Pricing de Sonnet: $3/M input, $15/M output
        step_cost = (
            response.usage.input_tokens / 1e6 * 3.0
            + response.usage.output_tokens / 1e6 * 15.0
        )
        total_cost += step_cost

        if config.verbose:
            log_agent_step(iteration + 1, response)

        # -- Verificar token budget --
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

        # -- Respuesta final (Claude ya no necesita tools) --
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
                total_tokens=total_tokens,
                total_cost=total_cost,
                elapsed_seconds=elapsed,
                tool_calls=tool_call_log,
                stopped_reason="completed",
            )

        # -- Tool use: Claude quiere llamar una o mas tools --
        if response.stop_reason == "tool_use":
            # Primero, agregar la respuesta de Claude al historial.
            # Esto incluye tanto su razonamiento (TextBlock) como
            # las tool calls (ToolUseBlock). Claude necesita "ver"
            # lo que dijo antes para mantener coherencia.
            messages.append({"role": "assistant", "content": response.content})

            # Ahora procesamos TODAS las tool calls.
            # Claude puede pedir multiples tools en una sola respuesta
            # (parallel tool use). Necesitamos ejecutar todas y enviar
            # todos los resultados juntos.
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue  # saltar TextBlocks

                # -- Loop detection --
                # Si Claude llama la misma tool con los mismos parametros
                # que ya llamo antes, probablemente esta en un loop.
                # Le enviamos un error para que cambie de estrategia.
                sig = f"{block.name}:{json.dumps(block.input, sort_keys=True)}"
                if sig in call_history:
                    logger.warning(f"  [Agent] Loop detectado: {sig}")
                    error_log.append(f"Loop: {sig}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Error: ya hiciste esta misma llamada antes. "
                                   "Usa la informacion que ya tienes.",
                        "is_error": True,
                    })
                    continue
                call_history.append(sig)

                # -- Ejecutar la tool --
                logger.info(f"  Ejecutando {block.name}...")
                result_str = execute_tool(block.name, block.input)

                # Log del resultado (truncado para no spamear)
                display = result_str[:200] + "..." if len(result_str) > 200 else result_str
                logger.info(f"  Resultado: {display}")

                # Guardar en el log para el AgentResult final
                tool_call_log.append({
                    "iteration": iteration + 1,
                    "tool": block.name,
                    "input": block.input,
                    "output_preview": result_str[:300],
                })

                # -- Error tracking --
                # Verificamos si el resultado contiene un error.
                # Si hay muchos errores seguidos, abortamos.
                is_error = False
                try:
                    parsed = json.loads(result_str)
                    if isinstance(parsed, dict) and "error" in parsed:
                        is_error = True
                        consecutive_errors += 1
                        error_log.append(
                            f"{block.name}: {parsed['error'][:200]}"
                        )
                    else:
                        # Un exito resetea el contador de errores
                        consecutive_errors = 0
                except json.JSONDecodeError:
                    consecutive_errors = 0

                # Demasiados errores seguidos = algo esta mal
                if consecutive_errors >= config.max_consecutive_errors:
                    logger.error("[Agent] Demasiados errores consecutivos")
                    return AgentResult(
                        answer="Demasiados errores consecutivos al consultar GitHub.",
                        iterations=iteration + 1,
                        total_tokens=total_tokens,
                        total_cost=total_cost,
                        elapsed_seconds=time.time() - start,
                        tool_calls=tool_call_log,
                        errors=error_log,
                        stopped_reason="error_threshold",
                    )

                # -- Agregar resultado al batch --
                # is_error le dice a Claude que la tool fallo,
                # para que pueda decidir si reintentar o cambiar
                # de estrategia.
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                    "is_error": is_error,
                })

            # Enviar TODOS los resultados juntos como un solo mensaje.
            # Esto es importante: no se envian uno por uno.
            messages.append({"role": "user", "content": tool_results})

    # -- Si llegamos aqui, se acabaron las iteraciones --
    logger.error("[Agent] Maximo de iteraciones alcanzado")
    return AgentResult(
        answer="No pude completar la tarea en el numero maximo de pasos.",
        iterations=config.max_iterations,
        total_tokens=total_tokens,
        total_cost=total_cost,
        elapsed_seconds=time.time() - start,
        tool_calls=tool_call_log,
        errors=error_log,
        stopped_reason="max_iterations",
    )


# =====================================================================
# Print Result
# =====================================================================

def print_result(result: AgentResult):
    """Imprime el resultado del agente de forma legible."""
    print(f"\n{'='*60}")
    print(f"  RESULTADO")
    print(f"{'='*60}")
    print(f"\n{result.answer}")
    print(f"\n{'-'*60}")
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GitHub Agent para Klimbook")
    parser.add_argument(
        "--test", action="store_true",
        help="Ejecutar preguntas de prueba automaticas"
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="GitHub token (o usa GITHUB_TOKEN env var)"
    )
    args = parser.parse_args()

    # Inicializar el cliente de GitHub
    gh = GitHubClient(token=args.token)

    if args.test:
        # ---- Modo test: preguntas automaticas ----
        print("=" * 60)
        print("  GITHUB AGENT -- Test Mode")
        print(f"  GitHub requests disponibles: "
              f"{'5000/hora (con token)' if gh.token else '60/hora (sin token)'}")
        print("=" * 60)

        QUESTIONS = [
            "Cuales son los repos de zxxz456?",
            "Cuales son los 5 issues abiertos mas recientes de klimbook?",
            "Cuantos commits hubo esta semana en klimbook?",
            "Dame los detalles del issue mas reciente de klimbook",
            "Que lenguajes usa el proyecto klimbook?",
        ]

        for i, question in enumerate(QUESTIONS, 1):
            print(f"\n{'#'*60}")
            print(f"  Pregunta {i}/{len(QUESTIONS)}: \"{question}\"")
            print(f"{'#'*60}")

            tool_call_counts.clear()
            result = run_agent(question)
            print_result(result)

        print(f"\n[GitHub] Total requests realizados: {gh.requests_made}")
        gh.close()

    else:
        # ---- Modo interactivo ----
        print("=" * 60)
        print("  GITHUB AGENT (Interactive)")
        print(f"  GitHub: {'autenticado' if gh.token else 'sin token (60 req/hora)'}")
        print("  Escribe 'salir' para terminar")
        print("=" * 60)

        while True:
            question = input("\nPregunta: ").strip()

            if not question:
                continue

            if question.lower() in ("salir", "quit", "exit"):
                print(f"\n[GitHub] Total requests: {gh.requests_made}")
                gh.close()
                print("Hasta luego!")
                break

            tool_call_counts.clear()
            result = run_agent(question)
            print_result(result)