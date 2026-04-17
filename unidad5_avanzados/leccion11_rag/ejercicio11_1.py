"""
ejercicio11_1.py
===========================
Ejercicio 11.1 — RAG basico sobre documentacion de Klimbook


Description:
-----------
- Indexar 5-7 archivos .md de Klimbook en ChromaDB
- Usar sentence-transformers (all-MiniLM-L6-v2) para embeddings
- Chunking recursivo: headers markdown -> parrafos -> overlap
- Metadata por chunk: source_file, doc_type, chunk_index, section_title
- search_docs como tool del agente con top_k configurable
- Probar con 5 preguntas y verificar relevancia de chunks


Requisitos:
-----------
pip install anthropic chromadb sentence-transformers


Metadata:
----------
* Author: zxxz6
* Version: 1.0.0


History:
------------
Author      Date            Description
zxxz6       17/04/2026      Creation

"""

from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
import chromadb
import json
import time
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rag-agent")

client = Anthropic()
MODEL = "claude-sonnet-4-20250514"


# =====================================================================
# Embedding Model
# =====================================================================
#
# sentence-transformers es una libreria que corre modelos de embedding
# localmente en tu maquina. No necesitas API key ni conexion a internet
# (despues de la primera descarga del modelo).
#
# all-MiniLM-L6-v2:
#   - 384 dimensiones por vector
#   - ~90MB de tamanio
#   - Bueno para ingles (aceptable para espanol)
#   - Rapido en CPU (no necesitas GPU)
#
# La primera vez que lo cargas, se descarga automaticamente.
#

logger.info("[Setup] Cargando modelo de embeddings...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # dimensiones del vector de este modelo
logger.info("[Setup] Modelo de embeddings listo")


# =====================================================================
# ChromaDB Setup
# =====================================================================
#
# ChromaDB es un vector store ligero que corre en Python.
# PersistentClient guarda los datos en disco, asi que no
# necesitas re-indexar cada vez que ejecutas el script.
#
# Una "collection" en ChromaDB es como una tabla en SQL.
# Guarda: id, document (texto), embedding (vector), metadata.
#

VECTORSTORE_PATH = "./klimbook_vectorstore"

chroma_client = chromadb.PersistentClient(path=VECTORSTORE_PATH)

# get_or_create: si ya existe la collection, la abre.
# Si no existe, la crea. Esto permite re-ejecutar el script
# sin duplicar datos (siempre y cuando los IDs sean los mismos).
collection = chroma_client.get_or_create_collection(
    name="klimbook_docs",
    metadata={
        "description": "Documentacion de Klimbook para RAG",
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dim": str(EMBEDDING_DIM),
    },
)


# =====================================================================
# Chunking
# =====================================================================

def chunk_by_markdown_headers(text: str) -> list[dict]:
    """
    Divide un documento markdown por sus headers (##, ###).
    
    Cada seccion se convierte en un chunk con metadata
    que incluye el titulo de la seccion.
    
    Ejemplo:
      "## Authentication\nJWT tokens...\n## Deployment\nDocker..."
      ->
      [
        {"text": "## Authentication\nJWT tokens...", "section": "Authentication"},
        {"text": "## Deployment\nDocker...", "section": "Deployment"},
      ]
    
    Nota: headers de nivel 1 (#) se incluyen como parte del primer chunk
    porque suelen ser el titulo del documento.
    """
    # Patron: buscar lineas que empiezan con ## o ###
    # (no # porque es el titulo del documento y queremos mantenerlo
    # con la primera seccion)
    sections = re.split(r'\n(?=#{2,3} )', text)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extraer el titulo de la seccion del primer header
        first_line = section.split("\n")[0].strip()
        # Quitar los # del inicio para obtener el titulo limpio
        section_title = re.sub(r'^#+\s*', '', first_line)

        chunks.append({
            "text": section,
            "section": section_title,
        })

    return chunks


def chunk_by_paragraphs(text: str, max_words: int = 300) -> list[str]:
    """
    Divide texto por parrafos, acumulando hasta max_words.
    
    Usa doble salto de linea como separador de parrafo.
    Acumula parrafos consecutivos hasta que el total excede
    max_words, entonces corta y empieza un chunk nuevo.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        combined = current + "\n\n" + para if current else para

        if len(combined.split()) > max_words and current:
            chunks.append(current.strip())
            current = para
        else:
            current = combined

    if current.strip():
        chunks.append(current.strip())

    return chunks


def add_overlap(chunks: list[str], overlap_words: int = 50) -> list[str]:
    """
    Agrega overlap entre chunks consecutivos.
    
    El overlap incluye las ultimas N palabras del chunk anterior
    al inicio del chunk siguiente. Esto asegura que ideas que
    cruzan el borde entre dos chunks no se pierdan.
    
    Ejemplo con overlap_words=3:
      Chunk 1: "A B C D E F G H I J"
      Chunk 2: "K L M N O P Q R S T"
      ->
      Chunk 1: "A B C D E F G H I J"
      Chunk 2: "H I J K L M N O P Q R S T"
              (H I J viene del final del chunk 1)
    """
    if len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]

    for i in range(1, len(chunks)):
        # Tomar las ultimas overlap_words del chunk anterior
        prev_words = chunks[i - 1].split()
        overlap_text = " ".join(prev_words[-overlap_words:])

        # Prepend al chunk actual
        overlapped.append(overlap_text + "\n\n" + chunks[i])

    return overlapped


def chunk_recursive(
    text: str,
    max_words: int = 300,
    overlap_words: int = 50,
) -> list[dict]:
    """
    Chunking recursivo (la estrategia recomendada).
    
    1. Dividir por headers markdown (respeta la estructura del doc)
    2. Si un chunk es demasiado grande, subdividirlo por parrafos
    3. Agregar overlap entre chunks consecutivos
    
    Retorna lista de dicts con 'text' y 'section'.
    """
    # Paso 1: Dividir por headers
    header_chunks = chunk_by_markdown_headers(text)

    # Paso 2: Subdividir chunks grandes
    final_chunks = []
    for hc in header_chunks:
        words = len(hc["text"].split())

        if words <= max_words:
            # Chunk cabe, dejarlo como esta
            final_chunks.append(hc)
        else:
            # Chunk demasiado grande, subdividir por parrafos
            sub_chunks = chunk_by_paragraphs(hc["text"], max_words=max_words)

            # Agregar overlap entre los sub-chunks
            sub_chunks = add_overlap(sub_chunks, overlap_words=overlap_words)

            for j, sc in enumerate(sub_chunks):
                final_chunks.append({
                    "text": sc,
                    "section": f"{hc['section']} (part {j + 1})",
                })

    return final_chunks


# =====================================================================
# Indexing
# =====================================================================

def index_document(
    filepath: str,
    content: str,
    doc_type: str = "docs",
):
    """
    Indexa un documento en ChromaDB.
    
    1. Divide el documento en chunks con chunking recursivo
    2. Genera embeddings para cada chunk
    3. Los guarda en ChromaDB con metadata
    
    Si el documento ya fue indexado (mismos IDs), ChromaDB
    actualiza los datos existentes en vez de duplicar.
    
    Args:
        filepath: Ruta del archivo (para metadata)
        content: Contenido del archivo
        doc_type: Tipo de documento (readme, architecture, api, etc.)
    """
    logger.info(f"[Index] Procesando: {filepath} ({len(content)} chars)")

    # Dividir en chunks
    chunks = chunk_recursive(content, max_words=300, overlap_words=50)

    if not chunks:
        logger.warning(f"[Index] No se generaron chunks para {filepath}")
        return

    # Extraer textos y generar embeddings en batch
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts).tolist()

    # Preparar datos para ChromaDB.
    # Cada chunk necesita un ID unico. Usamos el nombre del archivo
    # + indice del chunk. Si re-indexas el mismo archivo, ChromaDB
    # actualiza los chunks existentes (upsert por ID).
    filename = filepath.split("/")[-1].replace(".", "_")
    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]

    metadatas = [
        {
            "source_file": filepath,
            "doc_type": doc_type,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "section_title": c["section"],
            "word_count": len(c["text"].split()),
        }
        for i, c in enumerate(chunks)
    ]

    # Upsert: si el ID ya existe, actualiza. Si no, inserta.
    # Esto permite re-indexar sin duplicar.
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    logger.info(
        f"[Index] {filepath} -> {len(chunks)} chunks indexados "
        f"(tipo: {doc_type})"
    )


def index_all_documents(docs: list[tuple[str, str, str]]):
    """
    Indexa multiples documentos.
    
    Args:
        docs: Lista de (filepath, content, doc_type)
    """
    logger.info(f"[Index] Indexando {len(docs)} documentos...")
    start = time.time()

    for filepath, content, doc_type in docs:
        index_document(filepath, content, doc_type)

    elapsed = time.time() - start
    total_chunks = collection.count()
    logger.info(
        f"[Index] Completado en {elapsed:.2f}s | "
        f"Total chunks en el store: {total_chunks}"
    )


# =====================================================================
# Search
# =====================================================================

def search_docs(query: str, top_k: int = 5, doc_type: str = "") -> dict:
    """
    Busca los chunks mas relevantes para una query.
    
    Esta funcion se usa como TOOL del agente. Cuando Claude
    necesita informacion sobre Klimbook, llama a esta tool.
    
    El proceso:
    1. Convierte la query a un vector de 384 dimensiones
    2. ChromaDB busca los top_k vectores mas cercanos (cosine distance)
    3. Filtra por doc_type si se especifica
    4. Retorna los chunks con texto, metadata, y similarity score
    
    Args:
        query: Pregunta en lenguaje natural
        top_k: Cuantos resultados retornar (default: 5)
        doc_type: Filtrar por tipo de documento (opcional)
        
    Returns:
        Dict con query, total de resultados, y lista de chunks
    """
    logger.info(f"  [Search] Query: '{query[:80]}' (top_k={top_k})")

    # Limitar top_k por seguridad
    top_k = min(max(1, top_k), 10)

    # Embedizar la query
    query_embedding = embed_model.encode([query]).tolist()

    # Filtro de metadata (opcional).
    # Permite buscar solo en un tipo de documento.
    # Ejemplo: doc_type="api" solo busca en API_DOCS.md
    where_filter = None
    if doc_type:
        where_filter = {"doc_type": doc_type}

    # Buscar en ChromaDB.
    # query() calcula la distancia coseno entre el vector de la query
    # y todos los vectores en la collection, y retorna los top_k
    # mas cercanos.
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    # Formatear resultados.
    # ChromaDB retorna listas anidadas (soporta multiples queries).
    # Como solo hacemos una query, tomamos [0] de cada lista.
    chunks = []
    for i in range(len(results["ids"][0])):
        # ChromaDB retorna distancia (0 = identico, 2 = opuesto).
        # Convertimos a similarity (1 = identico, 0 = sin relacion)
        # para que sea mas intuitivo.
        distance = results["distances"][0][i]
        similarity = 1 - (distance / 2)  # normalizar a 0-1

        # Solo incluir chunks con similitud razonable.
        # Chunks con similarity < 0.3 son ruido y solo agregarian
        # tokens innecesarios al contexto de Claude.
        if similarity < 0.25:
            continue

        metadata = results["metadatas"][0][i]
        chunks.append({
            "text": results["documents"][0][i],
            "source_file": metadata.get("source_file", "unknown"),
            "section": metadata.get("section_title", ""),
            "doc_type": metadata.get("doc_type", ""),
            "similarity": round(similarity, 3),
        })

    logger.info(
        f"  [Search] -> {len(chunks)} chunks relevantes "
        f"(de {len(results['ids'][0])} total)"
    )

    return {
        "query": query,
        "results_count": len(chunks),
        "chunks": chunks,
    }


# =====================================================================
# Agent
# =====================================================================

TOOLS = [
    {
        "name": "search_docs",
        "description": (
            "Searches Klimbook's documentation for relevant information. "
            "Use this ALWAYS before answering technical questions about "
            "Klimbook. Returns the most relevant text chunks with source "
            "files and similarity scores.\n\n"
            "When to use:\n"
            "- Questions about how Klimbook works\n"
            "- Questions about architecture, API, deployment\n"
            "- Questions about specific features or services\n"
            "- Questions about what changed in a version\n\n"
            "Tips for good queries:\n"
            "- Use specific keywords: 'JWT authentication' not 'auth'\n"
            "- If first search doesn't find what you need, try different terms\n"
            "- Use doc_type filter for targeted search: 'api', 'architecture', 'changelog'\n\n"
            "Example: search_docs(query='JWT token authentication login', top_k=5)"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query. Use specific technical terms. "
                        "Examples: 'PostGIS geospatial queries', "
                        "'Docker deployment production', "
                        "'push notifications Firebase'"
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results (default: 5, max: 10).",
                    "default": 5,
                },
                "doc_type": {
                    "type": "string",
                    "description": (
                        "Filter by document type (optional). "
                        "Values: 'readme', 'architecture', 'api', "
                        "'deployment', 'changelog', 'contributing', 'service'"
                    ),
                    "default": "",
                },
            },
            "required": ["query"],
        },
    },
]

TOOL_FUNCTIONS = {
    "search_docs": search_docs,
}

SYSTEM_PROMPT = """You are a technical support assistant for Klimbook, a social 
network for rock climbers built with FastAPI, PostgreSQL/PostGIS, React Native, 
and Docker.

CRITICAL RULES:
1. ALWAYS use search_docs before answering technical questions about Klimbook.
2. NEVER make up information. Only use what the documentation says.
3. If the documentation doesn't contain the answer, say so honestly.
4. Cite the source file when referencing specific documentation.
5. If the first search doesn't find relevant results, try different search terms.

When answering:
- Be specific and reference actual code, endpoints, or configuration
- Mention the source file so the user can find more details
- If the question spans multiple topics, do multiple searches"""


# =====================================================================
# Agent Loop
# =====================================================================

@dataclass
class AgentResult:
    answer: str
    iterations: int
    total_tokens: int
    total_cost: float
    elapsed_seconds: float
    tool_calls: list[dict] = field(default_factory=list)
    chunks_used: list[dict] = field(default_factory=list)


def run_agent(question: str, max_iterations: int = 8) -> AgentResult:
    """
    Agent loop del RAG agent.
    
    El flujo tipico:
    1. Claude recibe la pregunta
    2. Claude llama search_docs() con keywords relevantes
    3. Claude recibe los chunks de documentacion
    4. Claude genera una respuesta basada en los chunks
    
    Si la primera busqueda no es suficiente, Claude puede
    hacer multiples busquedas con diferentes terminos.
    """
    messages = [{"role": "user", "content": question}]
    total_tokens = 0
    total_cost = 0.0
    tool_call_log = []
    all_chunks = []
    start = time.time()

    logger.info(f"\n{'='*60}")
    logger.info(f"[Agent] Pregunta: \"{question[:100]}\"")
    logger.info(f"{'='*60}")

    for iteration in range(max_iterations):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            tools=TOOLS,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        # Metricas
        inp = response.usage.input_tokens
        out = response.usage.output_tokens
        total_tokens += inp + out
        step_cost = (inp / 1e6) * 3.0 + (out / 1e6) * 15.0
        total_cost += step_cost

        # Log del step
        for block in response.content:
            if block.type == "text" and block.text.strip():
                logger.info(f"  [Step {iteration+1}] Thinking: {block.text[:200]}...")
            elif block.type == "tool_use":
                logger.info(
                    f"  [Step {iteration+1}] Tool: {block.name}"
                    f"({json.dumps(block.input)[:150]})"
                )

        # Respuesta final
        if response.stop_reason == "end_turn":
            elapsed = time.time() - start
            final_text = ""
            for block in response.content:
                if block.type == "text":
                    final_text += block.text

            logger.info(
                f"\n[Agent] Completado en {iteration+1} steps | "
                f"{elapsed:.2f}s | ${total_cost:.6f}"
            )

            return AgentResult(
                answer=final_text,
                iterations=iteration + 1,
                total_tokens=total_tokens,
                total_cost=total_cost,
                elapsed_seconds=elapsed,
                tool_calls=tool_call_log,
                chunks_used=all_chunks,
            )

        # Tool use
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                # Ejecutar la tool
                func = TOOL_FUNCTIONS.get(block.name)
                if not func:
                    result = {"error": f"Tool desconocida: {block.name}"}
                else:
                    result = func(**block.input)

                # Guardar log de la tool call
                tool_call_log.append({
                    "iteration": iteration + 1,
                    "tool": block.name,
                    "input": block.input,
                    "results_count": result.get("results_count", 0),
                })

                # Guardar los chunks recuperados para el reporte
                if "chunks" in result:
                    for chunk in result["chunks"]:
                        all_chunks.append({
                            "query": block.input.get("query", ""),
                            "source": chunk.get("source_file", ""),
                            "section": chunk.get("section", ""),
                            "similarity": chunk.get("similarity", 0),
                        })

                result_str = json.dumps(result, ensure_ascii=False)

                # Log del resultado (truncado)
                logger.info(
                    f"  [Step {iteration+1}] Result: "
                    f"{result.get('results_count', 0)} chunks found"
                )

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

            messages.append({"role": "user", "content": tool_results})

    # Max iterations
    elapsed = time.time() - start
    return AgentResult(
        answer="No pude encontrar la respuesta en el numero maximo de pasos.",
        iterations=max_iterations,
        total_tokens=total_tokens,
        total_cost=total_cost,
        elapsed_seconds=elapsed,
        tool_calls=tool_call_log,
        chunks_used=all_chunks,
    )


# =====================================================================
# Print Result
# =====================================================================

def print_result(result: AgentResult):
    """Imprime resultado con metricas y chunks usados."""
    print(f"\n{'='*60}")
    print(f"  RESPUESTA")
    print(f"{'='*60}")
    print(f"\n{result.answer}")
    print(f"\n{'-'*60}")
    print(f"  Metricas:")
    print(f"    Steps:      {result.iterations}")
    print(f"    Tokens:     {result.total_tokens:,}")
    print(f"    Costo:      ${result.total_cost:.6f}")
    print(f"    Tiempo:     {result.elapsed_seconds:.2f}s")
    print(f"    Tool calls: {len(result.tool_calls)}")

    if result.tool_calls:
        print(f"\n  Busquedas realizadas:")
        for tc in result.tool_calls:
            print(f"    [{tc['iteration']}] search_docs('{tc['input'].get('query', '')[:60]}') "
                  f"-> {tc['results_count']} chunks")

    if result.chunks_used:
        print(f"\n  Chunks usados ({len(result.chunks_used)}):")
        # Agrupar por source file para mostrar mas limpio
        by_source = {}
        for chunk in result.chunks_used:
            src = chunk["source"]
            if src not in by_source:
                by_source[src] = []
            by_source[src].append(chunk)

        for source, chunks in by_source.items():
            sims = [c["similarity"] for c in chunks]
            avg_sim = sum(sims) / len(sims)
            sections = [c["section"] for c in chunks if c["section"]]
            print(f"    {source}: {len(chunks)} chunks (avg sim: {avg_sim:.3f})")
            if sections:
                print(f"      Secciones: {', '.join(sections[:5])}")


# =====================================================================
# Simulated Klimbook Documentation
# =====================================================================
#
# Estos son los documentos que se indexan en ChromaDB.
# En un caso real, lerias estos archivos del repositorio de Klimbook.
# Aqui los simulamos con contenido realista.
#

KLIMBOOK_DOCS = [
    ("docs/README.md", "readme", """\
# Klimbook

Klimbook is a cross-platform social network for rock climbers. It allows users 
to track their climbing ascents, discover new routes and crags, follow other 
climbers, and share their progress.

## Tech Stack

The backend is built with FastAPI (Python 3.11+) and uses PostgreSQL 15 with the 
PostGIS extension for geospatial data. The social graph (followers, likes) is 
stored in Neo4j. Redis is used for session caching, rate limiting, and frequently 
accessed data. Media files (profile photos, route images) are stored in 
Cloudflare R2 (S3-compatible object storage).

The mobile app is built with React Native and Expo, targeting both iOS and Android. 
The web platform uses React with Vite.

All services are containerized with Docker and orchestrated with Docker Compose. 
Nginx serves as reverse proxy handling SSL termination and load balancing.

## Features

Klimbook offers the following main features:

Route and crag discovery with interactive maps powered by PostGIS. Users can 
search for climbing areas near their location and browse available routes with 
grades, descriptions, and photos.

Ascent logging supports multiple grading systems (YDS, French, V-scale) and 
climbing styles (onsight, flash, redpoint, repeat). Each ascent records the 
route, date, style, and optional notes and rating.

Social features include following other climbers, an activity feed showing 
friends' recent ascents, push notifications via Firebase Cloud Messaging, and 
a rewards system for achievements.

The climber's book provides personal statistics including total ascents, grade 
progression over time, most climbed routes, and climbing calendar heatmap.

Multi-language support covers English and Spanish with ~1,030 translation keys 
managed through i18next on the frontend and a custom i18n system on the backend.

## Getting Started

1. Clone the repository: git clone https://github.com/zxxz456/klimbook
2. Copy .env.example to .env and fill in your values
3. Run docker-compose up -d to start all services
4. Access the API at http://localhost:8000/docs
5. Run alembic upgrade head to apply database migrations
"""),

    ("docs/ARCHITECTURE.md", "architecture", """\
# Architecture

Klimbook follows a microservices architecture with five main services 
communicating over an internal Docker network.

## Auth Service

The auth service handles user registration, login, JWT token management, and 
password reset. It uses bcrypt for password hashing with a cost factor of 12. 
JWT tokens are signed with RS256 (RSA with SHA-256) using a 2048-bit key pair. 
Access tokens expire after 15 minutes and refresh tokens after 7 days.

The refresh token rotation strategy invalidates the old refresh token when a new 
one is issued. Refresh tokens are stored in Redis with their expiration time 
for quick validation and revocation.

Endpoints:
- POST /auth/register: Create new account (email, password, username, display_name)
- POST /auth/login: Authenticate and receive token pair
- POST /auth/refresh: Exchange refresh token for new token pair
- POST /auth/logout: Invalidate refresh token
- POST /auth/forgot-password: Send password reset email
- POST /auth/reset-password: Reset password with token

## Climbing Service

The climbing service manages routes, crags, ascents, and grading systems. It 
integrates with PostGIS for geospatial queries like finding crags near a location.

Routes belong to crags, and each route has a name, grade, type (sport, boulder, 
trad), number of pitches, and description. Ascents link users to routes with 
style, date, notes, and rating.

The grade conversion system supports YDS (5.6-5.15d), French (4a-9c), and 
V-scale (V0-V17) with automatic conversion between systems.

Key PostGIS queries:
- Find crags within N km: ST_DWithin(geom::geography, ST_MakePoint(lon, lat)::geography, meters)
- Calculate distance: ST_Distance(geom::geography, point::geography) / 1000 for km
- Note: ST_MakePoint takes (longitude, latitude), not (lat, lon)

## Social Service

The social service manages the social graph using Neo4j. It handles follower 
relationships, activity feed generation, and activity streams.

The feed is generated by querying Neo4j for the user's followed users, then 
fetching their recent activities from PostgreSQL, sorted by timestamp. Feed 
entries are cached in Redis for 5 minutes.

Follow/unfollow operations are atomic in Neo4j using MERGE for follows and 
DELETE for unfollows. The follower count is maintained as a denormalized field 
on the user record in PostgreSQL for fast read access.

## Notification Service

Push notifications are sent via Firebase Cloud Messaging (FCM). The service 
supports both iOS (APNs via FCM) and Android notifications.

Notification categories:
- new_follower: when someone follows you
- new_ascent: when a followed user logs an ascent
- achievement: when you earn a reward
- system: app updates and announcements

Users can configure notification preferences per category through the 
preferences endpoint. Silent notifications are used for feed updates to 
trigger background refresh on the mobile app.

## Profile Service

The profile service manages user profiles, preferences, statistics, and 
achievements. The statistics are computed from ascent data and cached in Redis.

User preferences are stored as a JSONB column on the users table, supporting:
- theme: light or dark
- language: en or es
- notification preferences per category
- first_login: boolean flag for onboarding tutorial
- measurement units: metric or imperial
"""),

    ("docs/API_DOCS.md", "api", """\
# API Documentation

## Base URLs

Development: http://localhost:8000
Production: https://api.klimbook.com

All endpoints except /auth/login and /auth/register require a valid JWT token 
in the Authorization header: Authorization: Bearer <token>

## Authentication Endpoints

POST /auth/register
Create a new user account.
Body: {email, password, username, display_name}
Returns: {user_id, token, refresh_token}
Validation: email must be unique, password minimum 8 characters, username 
3-50 characters alphanumeric with underscores.

POST /auth/login
Authenticate and receive tokens.
Body: {email, password}
Returns: {token, refresh_token, user: {id, username, display_name, email}}
Rate limited: 5 attempts per minute per email.

POST /auth/refresh
Exchange refresh token for new token pair.
Body: {refresh_token}
Returns: {token, refresh_token}

## Climbing Endpoints

GET /routes?lat=X&lon=Y&radius=Z
Find routes near a location using PostGIS.
Query params: lat (float), lon (float), radius (km, default 25), 
grade_min, grade_max, type (sport|boulder|trad|any)
Returns: Array of routes with distance_km, sorted by distance.

GET /routes/{route_id}
Get full details of a specific route.
Returns: {id, name, grade, type, crag, pitches, description, ascent_count, 
avg_rating, created_at}

POST /ascents
Log a new climbing ascent.
Body: {route_id, style (onsight|flash|redpoint|repeat), date, notes, rating}
Returns: Created ascent with updated user statistics.
Validation: style must be valid, date cannot be in the future, rating 1-5.

GET /users/{user_id}/book
Get a user's climbing book with statistics.
Returns: {total_ascents, highest_grade, grade_distribution, style_distribution, 
recent_ascents, climbing_calendar, favorite_crags}

GET /users/{user_id}/ascents?page=1&limit=20
Paginated list of a user's ascents.
Returns: {items: [...], total, page, pages}

## Social Endpoints

POST /follow/{user_id}
Follow another user. Creates relationship in Neo4j.
Returns: {status: "following", follower_count, following_count}

DELETE /follow/{user_id}
Unfollow a user.
Returns: {status: "unfollowed"}

GET /feed?page=1&limit=20
Get the authenticated user's activity feed.
Returns: Paginated feed of activities from followed users, sorted by timestamp.

GET /users/{user_id}/followers?page=1&limit=20
Get a user's followers.

GET /users/{user_id}/following?page=1&limit=20
Get who a user follows.

## Search Endpoints

GET /search/routes?q=term&type=sport&grade_min=5.10a
Full-text search for routes by name or description.

GET /search/crags?q=term&lat=X&lon=Y&radius=Z
Search crags by name, optionally filtered by location.

GET /search/users?q=term
Search users by username or display name.
"""),

    ("docs/DEPLOYMENT.md", "deployment", """\
# Deployment Guide

## Environments

Development: Local Docker Compose setup with hot-reload enabled. The FastAPI 
server runs with --reload flag and the React Native app connects to the local 
API. PostgreSQL and Redis run in containers with ports exposed to localhost.

Staging: VPS with Docker Compose. Automated deployment via GitHub Actions on 
push to the develop branch. Uses a separate database from production.

Production: VPS with Docker Compose. Manual deployment with rollback support. 
Uses Nginx for SSL termination with Let's Encrypt certificates auto-renewed 
by Certbot.

## Docker Setup

Each service has its own Dockerfile. The docker-compose.yml orchestrates all 
services, databases, and the reverse proxy.

Key commands:
- docker-compose up -d: Start all services in background
- docker-compose logs -f <service>: View live logs for a service
- docker-compose exec <service> alembic upgrade head: Run database migrations
- docker-compose restart <service>: Restart a specific service
- docker-compose down: Stop all services (data persists in volumes)

## CI/CD Pipeline

GitHub Actions workflow runs on push to main branch:

1. Lint: ruff check and mypy type checking
2. Test: pytest with PostgreSQL test database in CI
3. Build: Docker images built and tagged with commit SHA
4. Deploy to staging: Automatic on develop branch
5. Deploy to production: Manual approval required on main branch

The deployment script performs:
1. Pull latest images on the VPS
2. Run database migrations (alembic upgrade head)
3. Restart services one by one (rolling restart)
4. Health check on each service
5. Rollback if health check fails

## Database Migrations

Using Alembic for PostgreSQL schema migrations. Each service manages its own 
migration history in the alembic/versions/ directory.

To create a new migration:
  alembic revision --autogenerate -m "description of change"

To apply migrations:
  alembic upgrade head

To rollback one migration:
  alembic downgrade -1

Always run migrations BEFORE deploying new code that depends on schema changes.

## Backup Strategy

PostgreSQL: Daily pg_dump to Cloudflare R2, retained for 30 days.
Neo4j: Weekly neo4j-admin dump, retained for 14 days.
Redis: No backup needed (cache only, regenerated from PostgreSQL).
Media files: Cloudflare R2 has built-in redundancy.

## Monitoring

Health endpoints: GET /health on each service returns {status: "ok", version, uptime}.
Logs: Structured JSON logging with request_id for tracing across services.
Metrics: Basic Prometheus metrics exposed at /metrics.
Alerts: Email alerts for service downtime and high error rates.
"""),

    ("docs/CHANGELOG.md", "changelog", """\
# Changelog

## v2.10.0 (April 10, 2026)
- Shareable climbing cards for Instagram stories
- Topo markers on wall photos
- Performance improvements for feed loading
- Fixed notification badge count on iOS

## v2.9.0 (April 3, 2026)
- Added onboarding tutorial system for new users on mobile
- Backend support for first_login preference flag
- 45+ new i18n translation keys for tutorial content
- Tutorial restart option accessible from user menu settings
- 4 new tests for first_login preference handling

## v2.8.0 (March 28, 2026)
- Wall mode integration in climbing book
- New wall cards with action menus and grade distribution charts
- Filter improvements for wall and block views
- 6 new screens for wall-related features
- Bug fix: wall grade chart not rendering on Android

## v2.7.0 (March 21, 2026)
- Full multi-language support (English and Spanish)
- i18next integration with approximately 1,030 translation keys
- Language persistence via AsyncStorage on mobile
- Zustand store integration for language synchronization across components
- Language selector in settings with immediate UI update

## v2.6.0 (March 14, 2026)
- Push notification system via Firebase Cloud Messaging
- Notification preferences configurable per category
- Badge count management for iOS and Android
- Silent notifications for background feed refresh
- Deep linking from notifications to relevant content

## v2.5.0 (March 7, 2026)
- Interactive map for crag discovery with clustering
- PostGIS integration for proximity-based search
- Crag detail pages with route listings and statistics
- Map clustering for areas with many nearby crags
- Performance: spatial index on crags.geom column
"""),

    ("docs/CONTRIBUTING.md", "contributing", """\
# Contributing to Klimbook

## Development Setup

1. Fork the repository on GitHub
2. Clone your fork: git clone https://github.com/YOUR_USER/klimbook
3. Create a feature branch: git checkout -b feat/your-feature
4. Install dependencies: pip install -r requirements.txt
5. Copy .env.example to .env and configure
6. Start services: docker-compose up -d
7. Run migrations: alembic upgrade head
8. Make your changes and write tests
9. Run tests: pytest
10. Submit a pull request to the develop branch

## Coding Standards

Python backend:
- Follow PEP 8 style guide
- Use type hints on all function signatures
- Write docstrings for public functions
- Use Pydantic for all data validation
- Prefer async/await for I/O operations
- Use PostGIS for ALL geospatial queries (never calculate in Python)

TypeScript/React Native:
- Follow the ESLint configuration
- Use functional components with hooks
- Use Zustand for state management
- Translations: always use t() from i18next, never hardcode strings

Commits:
- Follow Conventional Commits: feat:, fix:, refactor:, docs:, chore:, test:, ci:
- Include scope when relevant: feat(auth): add password reset
- Keep messages concise and descriptive

## Architecture Decisions

- Keep services small and focused (single responsibility)
- New endpoints must have corresponding Pydantic request/response models
- Database changes require Alembic migrations
- New features must support both English and Spanish
- All user-facing errors must have i18n keys
"""),
]


# =====================================================================
# Entry Point
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Klimbook RAG Agent")
    parser.add_argument(
        "--reindex", action="store_true",
        help="Re-indexar todos los documentos (borrar y recrear)",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Ejecutar preguntas de prueba",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Modo interactivo",
    )
    parser.add_argument(
        "--search", type=str, default=None,
        help="Buscar directamente sin agente (debug)",
    )
    args = parser.parse_args()

    # ---- Re-indexar ----
    if args.reindex or collection.count() == 0:
        if args.reindex:
            # Borrar collection existente y recrear
            chroma_client.delete_collection("klimbook_docs")
            collection = chroma_client.create_collection(
                name="klimbook_docs",
                metadata={
                    "description": "Documentacion de Klimbook para RAG",
                    "embedding_model": "all-MiniLM-L6-v2",
                },
            )
            logger.info("[Setup] Collection borrada y recreada")

        docs = [
            (filepath, doc_type, content)
            for filepath, doc_type, content in KLIMBOOK_DOCS
        ]
        index_all_documents([(fp, content, dt) for fp, dt, content in KLIMBOOK_DOCS])

    logger.info(f"[Setup] ChromaDB tiene {collection.count()} chunks indexados")

    # ---- Busqueda directa (debug) ----
    if args.search:
        results = search_docs(args.search, top_k=5)
        print(f"\nResultados para: '{args.search}'")
        print(f"Total: {results['results_count']} chunks\n")
        for i, chunk in enumerate(results["chunks"]):
            print(f"  [{i+1}] sim={chunk['similarity']:.3f} | {chunk['source_file']} | {chunk['section']}")
            print(f"      {chunk['text'][:150]}...")
            print()
        exit(0)

    # ---- Modo test ----
    if args.test:
        print("=" * 60)
        print("  KLIMBOOK RAG AGENT -- Test Mode")
        print("=" * 60)

        QUESTIONS = [
            "How does authentication work in Klimbook? What algorithm is used for JWT?",
            "How do I deploy Klimbook to production? What are the steps?",
            "What changed in version 2.9.0?",
            "How does the proximity search work for finding crags near a location?",
            "What coding standards should I follow when contributing to Klimbook?",
        ]

        for i, question in enumerate(QUESTIONS, 1):
            print(f"\n{'#'*60}")
            print(f"  Pregunta {i}/{len(QUESTIONS)}:")
            print(f"  \"{question}\"")
            print(f"{'#'*60}")

            result = run_agent(question)
            print_result(result)

    # ---- Modo interactivo ----
    elif args.interactive or not args.test:
        print("=" * 60)
        print("  KLIMBOOK RAG AGENT")
        print(f"  {collection.count()} chunks indexados")
        print("  Escribe 'salir' para terminar")
        print("=" * 60)

        while True:
            question = input("\nPregunta: ").strip()
            if not question:
                continue
            if question.lower() in ("salir", "quit", "exit"):
                print("Hasta luego!")
                break

            result = run_agent(question)
            print_result(result)