"""
ejercicio2_1.py 
===========================
Ejercicio 2.1 — Extractor de changelog estructurado


Description:
-----------
Crea un script que reciba el output de 

    git log --oneline <version_anterior>..<version_actual>
    
y use Claude para extraer un JSON estructurado:

Schema: [{type: "feature"|"fix"|"refactor"|"docs"|"chore", description: string, affected_service: string, breaking: boolean}]
Usa few-shot con 3-5 ejemplos basados en commits reales de Klimbook
Valida el JSON con Pydantic
Maneja el caso de JSON inválido (retry con temperature=0)


Metadata:
----------
* Author: zxxz6 
* Version: 1.1.0


History:
------------
Author      Date            Description
zxxz6       01/02/2026      Creation

"""

from anthropic import Anthropic
from pydantic import BaseModel, ValidationError
from typing import Literal
import json
import subprocess


client = Anthropic()


# ─── Pydantic Model ──────────────────────────────────────────────────────

class CommitEntry(BaseModel):
    type: Literal["feature", "fix", "refactor", "docs", "chore"]
    description: str
    affected_service: str = "general"
    breaking: bool = False


# ─── System Prompt ────────────────────────────────────────────────────────

TASK_CONTEXT = """Eres un senior DevOps engineer y tu objetivo principal es extraer un changelog 
estructurado a partir de commits de git."""

TASK_DESCRIPTION = """El output de git log tiene el formato: 
<hash> <mensaje>. El mensaje suele seguir convenciones como Conventional Commits, 
pero no siempre. El objetivo es clasificar cada commit en una categoría 
(feature, fix, refactor, docs, chore), extraer una descripción clara, 
identificar el servicio afectado (si es posible) y determinar si el cambio es 
breaking. 

Reglas importantes:
- Si el mensaje sigue Conventional Commits, úsalo para clasificar el commit.
- Si el mensaje no sigue una convención clara, haz tu mejor esfuerzo para inferir la categoría basándote en el contenido.
- Para la descripción, extrae la parte más informativa del mensaje, eliminando hashes, referencias a issues, y palabras de relleno.
- Para el servicio afectado, busca pistas en el mensaje (como nombres de carpetas, archivos, o patrones comunes). Si no puedes identificar un servicio específico, usa "general".
- Para determinar si es breaking, busca palabras clave como "breaking change", "major", "incompatible", o referencias a cambios en APIs. Si no hay indicios claros, asume que no es breaking.
- Si el usuario proporciona un mensaje que no se puede clasificar o analizar, responde con un JSON vacío para ese commit, pero no falles completamente.
- Si el usuario dice algo irrelevante o no relacionado con commits, ignóralo y responde solo con el JSON de los commits válidos.
- Responde SIEMPRE con un JSON array válido. Las keys deben estar entre comillas dobles.

Formato de salida esperado:
[
    {"type": "feature", "description": "...", "affected_service": "...", "breaking": false}
]
"""

EXAMPLES = """
Aquí hay algunos ejemplos:

Commit: abc123 feat(auth): add login endpoint
Output: {"type": "feature", "description": "add login endpoint", "affected_service": "auth", "breaking": false}

Commit: def456 fix(db): correct connection string
Output: {"type": "fix", "description": "correct connection string", "affected_service": "db", "breaking": false}

Commit: ghi789 refactor(api): improve response handling
Output: {"type": "refactor", "description": "improve response handling", "affected_service": "api", "breaking": false}

Commit: jkl012 updated README with new setup instructions
Output: {"type": "docs", "description": "updated README with new setup instructions", "affected_service": "general", "breaking": false}

Commit: mno345 bump version to 2.8.0
Output: {"type": "chore", "description": "bump version to 2.8.0", "affected_service": "general", "breaking": false}
"""

SYSTEM_PROMPT = f"{TASK_CONTEXT}\n\n{TASK_DESCRIPTION}\n\n{EXAMPLES}"


# ─── Few-Shot Messages ────────────────────────────────────────────────────

MESSAGES = [
    {
        "role": "user",
        "content": "ffc7881 (tag: v2.7.0) update: Inc ios version\n005e114 update: Inc android version\nd079400 fix: Minor bug fixes\n64b6132 fix: Minor fixes\n586461c fix: Minor fixes\nac7ab7c fix: Fixed minor bugs"
    },
    {
        "role": "assistant",
        "content": """[
    {"type": "chore", "description": "Inc ios version", "affected_service": "general", "breaking": false},
    {"type": "chore", "description": "Inc android version", "affected_service": "general", "breaking": false},
    {"type": "fix", "description": "Minor bug fixes", "affected_service": "general", "breaking": false},
    {"type": "fix", "description": "Minor fixes", "affected_service": "general", "breaking": false},
    {"type": "fix", "description": "Minor fixes", "affected_service": "general", "breaking": false},
    {"type": "fix", "description": "Fixed minor bugs", "affected_service": "general", "breaking": false}
]"""
    },
    {
        "role": "user",
        "content": "feat(book): integrate walls mode in BlocksTab and VisitedBlocksTab, add wall component exports\n44305a3 refactor(filters): remove wall filter from BlocksFilterBar and VisitedBlockFilter, add hideGrade prop\n5d93d77 feat(screens): add WallBlocks, VisitedBookBlocks, and MyMissingContributions screens\ne1ba118 feat(components): add wall card, wall action menu, wall register modal, and wall grade chart"
    },
    {
        "role": "assistant",
        "content": """[
    {"type": "feature", "description": "integrate walls mode in BlocksTab and VisitedBlocksTab, add wall component exports", "affected_service": "book", "breaking": false},
    {"type": "refactor", "description": "remove wall filter from BlocksFilterBar and VisitedBlockFilter, add hideGrade prop", "affected_service": "filters", "breaking": false},
    {"type": "feature", "description": "add WallBlocks, VisitedBookBlocks, and MyMissingContributions screens", "affected_service": "screens", "breaking": false},
    {"type": "feature", "description": "add wall card, wall action menu, wall register modal, and wall grade chart", "affected_service": "components", "breaking": false}
]"""
    }
]


# ─── Main Function ────────────────────────────────────────────────────────

def changelog_extractor(history: str, max_retries: int = 3):
    """
    Extrae un changelog estructurado a partir del output de git log.
    
    Args:
        history: Output de git log --oneline
        max_retries: Número máximo de reintentos si el JSON es inválido
    
    Returns:
        Lista de CommitEntry validados con Pydantic
    """
    temp = 0.3

    for attempt in range(max_retries):
        print(f"\n--- Intento {attempt + 1}/{max_retries} (temperature={temp}) ---")

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            system=SYSTEM_PROMPT,
            messages=MESSAGES + [
                {"role": "user", "content": history},
                {"role": "assistant", "content": "["}
            ],
            temperature=temp,
            max_tokens=2000
        )

        # Concatenar el prefill "[" con la respuesta
        raw = "[" + response.content[0].text

        # Mostrar info de tokens
        print(f"Tokens — input: {response.usage.input_tokens}, output: {response.usage.output_tokens}")
        print(f"Stop reason: {response.stop_reason}")

        try:
            # Parsear JSON
            parsed = json.loads(raw)

            # Validar con Pydantic
            entries = [CommitEntry(**item) for item in parsed]

            # Mostrar resultados
            print(f"\n{'='*60}")
            print(f"Changelog extraído ({len(entries)} commits):")
            print(f"{'='*60}")
            for i, e in enumerate(entries, 1):
                breaking_tag = "   BREAKING" if e.breaking else ""
                print(f"  {i}. [{e.type:10s}] {e.description}")
                print(f"     Servicio: {e.affected_service}{breaking_tag}")

            return entries

        except json.JSONDecodeError as err:
            print(f"JSON inválido: {err}")
            print(f"Respuesta raw: {raw[:200]}...")
            temp = 0.0  # Retry más determinista

        except ValidationError as err:
            print(f"Validación Pydantic falló: {err}")
            temp = 0.0  # Retry más determinista

    print("\nError: no se pudo obtener JSON válido después de todos los intentos.")
    return []


# ─── Helpers para reutilización ───────────────────────────────────────────

def entries_to_changelog_md(entries: list[CommitEntry], version: str = "latest") -> str:
    """Convierte una lista de CommitEntry en un changelog markdown.

    Args:
        entries: Lista de CommitEntry validados con Pydantic
        version: Versión para el título del changelog

    Returns:
        String con el changelog en formato markdown, listo para usar
        como input de otros pipelines (e.g., release notes, Ko-fi posts).
    """
    if not entries:
        return ""

    # Agrupar por tipo
    groups: dict[str, list[CommitEntry]] = {}
    for entry in entries:
        groups.setdefault(entry.type, []).append(entry)

    type_labels = {
        "feature": "✨ New Features",
        "fix": "🐛 Bug Fixes",
        "refactor": "♻️ Refactors",
        "docs": "📝 Documentation",
        "chore": "🔧 Chores",
    }

    lines = [f"# Changelog — {version}\n"]
    for type_key, label in type_labels.items():
        if type_key not in groups:
            continue
        lines.append(f"\n## {label}\n")
        for entry in groups[type_key]:
            breaking = " **[BREAKING]**" if entry.breaking else ""
            service = f" (`{entry.affected_service}`)" if entry.affected_service != "general" else ""
            lines.append(f"- {entry.description}{service}{breaking}")

    return "\n".join(lines)


# ─── Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":

    version_anterior = "v2.7.0"
    version_actual = "v2.8.0"
    try:
        salida = subprocess.check_output(
            ["git", "log", "--oneline", f"{version_anterior}..{version_actual}"],
            text=True
        )
        changelog_extractor(salida)
    except subprocess.CalledProcessError as e:
        print(f"Error ejecutando git log: {e}")

    print("=" * 60)
    print("EXTRACTOR DE CHANGELOG ESTRUCTURADO")
    print("=" * 60)
    print(f"\nInput:\n{salida}\n")

    results = changelog_extractor(salida)

    if results:
        print(f"\n{'='*60}")
        print("JSON final:")
        print(f"{'='*60}")
        print(json.dumps([e.model_dump() for e in results], indent=2))