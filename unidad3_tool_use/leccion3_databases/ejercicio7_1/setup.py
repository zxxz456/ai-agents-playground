"""
ejercicio7_1_setup.py
===========================
Ejercicio 7.1 — Database agent (Setup)


Description:
-----------
Script para crear la base de datos de prueba con:
- Schema simplificado de Klimbook: users, crags, routes, ascents
- Extension PostGIS para datos geograficos
- 50 usuarios, 20 crags, 200 rutas, 500 ascensos de ejemplo
- Indices para performance

Ejecutar ANTES del agente:
    python ejercicio7_1_setup.py

Requisitos:
- PostgreSQL 15+ corriendo en localhost
- Extension PostGIS instalada (apt install postgresql-15-postgis-3)
- Un usuario con permisos para crear bases de datos

Este script crea:
1. La base de datos 'klimbook_test' (la borra si ya existe)
2. Un usuario 'readonly_agent' con permisos solo SELECT
3. Las tablas con datos de ejemplo


Metadata:
----------
* Author: zxxz6
* Version: 1.0.0


History:
------------
Author      Date            Description
zxxz6       15/03/2026      Creation

"""

import asyncio
import asyncpg
import random
from datetime import datetime, timedelta


# =====================================================================
# Configuracion
# =====================================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "postgres",           # usuario admin para crear la DB
    "password": "postgres",       # ajusta segun tu setup
}

DB_NAME = "klimbook_test"
READONLY_USER = "readonly_agent"
READONLY_PASSWORD = "readonly_pass"


# =====================================================================
# Datos de ejemplo
# =====================================================================

# Crags reales de Mexico y alrededores con coordenadas
CRAGS_DATA = [
    ("El Cirio", "Puebla", -98.1234, 19.0567),
    ("Chonta", "Puebla", -98.0891, 18.9234),
    ("La Galera", "Hidalgo", -98.7654, 20.1234),
    ("Jilotepec", "Estado de Mexico", -99.5321, 19.9543),
    ("El Diente", "Nuevo Leon", -100.3456, 25.6789),
    ("Potrero Chico", "Nuevo Leon", -100.4567, 25.9012),
    ("Pena de Bernal", "Queretaro", -99.9432, 20.7456),
    ("San Cristobal", "Chiapas", -92.6321, 16.7345),
    ("Taxco", "Guerrero", -99.6054, 18.5567),
    ("Tepoztlan", "Morelos", -99.0987, 18.9876),
    ("El Salto", "Jalisco", -103.5678, 20.5432),
    ("La Huasteca", "Nuevo Leon", -100.4321, 25.6234),
    ("Mineral del Chico", "Hidalgo", -98.7321, 20.2234),
    ("Aculco", "Estado de Mexico", -99.8345, 20.0987),
    ("Pena del Aire", "Queretaro", -100.1234, 20.5876),
    ("Los Dinamos", "CDMX", -99.2876, 19.2987),
    ("Amecameca", "Estado de Mexico", -98.7654, 19.1234),
    ("San Miguel Regla", "Hidalgo", -98.5678, 20.2345),
    ("Xichu", "Guanajuato", -100.0567, 21.3012),
    ("Monterrey Urbano", "Nuevo Leon", -100.3167, 25.6866),
]

# Nombres de usuario ficticios
USERNAMES = [
    "climber_alex", "boulder_queen", "crack_master", "dani_sends",
    "el_pumita", "flora_climb", "gato_vertical", "hugo_crimp",
    "ivan_dyno", "julia_flash", "kike_onsight", "luna_rock",
    "marco_trad", "nadia_power", "oscar_beta", "paula_crimp",
    "quique_send", "rosa_climb", "sergio_wall", "tania_bloc",
    "uri_campus", "vale_route", "waldo_chalk", "xime_climb",
    "yago_pump", "zara_hold", "adan_jugs", "beto_pinch",
    "caro_sloper", "diego_heel", "elena_toe", "fabi_mantle",
    "gris_chimney", "hector_layback", "irma_gaston", "jaime_mono",
    "karen_kneebar", "leo_dropknee", "moni_flagging", "nacho_bat",
    "oli_rose", "paco_match", "queta_cross", "rafa_undercling",
    "sonia_sidepull", "toño_pocket", "ursula_jug", "vico_edge",
    "wendy_flake", "xavi_arete",
]

# Nombres de rutas ficticios
ROUTE_NAMES = [
    "La Esfinge", "Camino al Cielo", "Danza del Mono", "El Purgatorio",
    "Vuelo del Condor", "Manos de Piedra", "Sombra del Aguila", "Fuego Interior",
    "Paso del Jaguar", "Luna Llena", "Rayo de Sol", "Viento Norte",
    "Cascada Vertical", "Canto del Cenzontle", "Serpiente Emplumada",
    "Flor de Roca", "Trueno Seco", "Niebla Eterna", "Alma de Granito",
    "Corazon de Caliza", "El Desplome", "La Fisura Negra", "Techo del Mundo",
    "Dedos de Acero", "El Reloj de Arena", "Columna Vertebral", "Piel de Lagarto",
    "Beso del Viento", "Raiz Profunda", "Horizonte Vertical",
    "Mar de Nubes", "Espejismo", "Laberinto", "El Centinela",
    "Fragmento", "Relampago", "Susurro", "Equilibrio",
    "Genesis", "Apocalipsis", "Renacimiento", "Metamorfosis",
    "Obsidiana", "Cuarzo", "Basalto", "Granito Rojo",
    "Placa Tecnica", "Erosion", "Sedimento", "Cristalizacion",
    "Brecha", "Falla Geologica", "Estrato", "Capa Freática",
    "Punto Ciego", "Zona Muerta", "Limite", "Frontera",
    "Eclipse", "Solsticio", "Equinoccio", "Aurora",
    "Crater", "Volcan Dormido", "Lava Fria", "Ceniza",
    "Tormenta", "Calma", "Marejada", "Corriente",
    "Raices", "Corteza", "Savia", "Semilla",
    "Garra", "Colmillo", "Pluma", "Escama",
    "Acantilado", "Cornisa", "Repisa", "Extraplomo",
    "Regleta", "Agujero", "Pinza", "Invertido",
    "Dinamico", "Estatico", "Coordinado", "Explosivo",
    "Resistencia", "Potencia", "Tecnica", "Mental",
    "Amanecer Vertical", "Ocaso de Granito", "Medianoche", "Crepusculo",
    "Primera Vez", "Regreso", "Adios", "Bienvenida",
    "Nostalgia", "Euforia", "Calma Tensa", "Momento Clave",
    "El Proyecto", "La Hazaña", "El Milagro", "El Misterio",
    "Voz del Viento", "Eco de la Roca", "Huella", "Marca",
    "Esencia", "Presencia", "Ausencia", "Existencia",
    "Primer Movimiento", "Ultimo Agarre", "Cadena Completa", "Intento Fallido",
    "Pies de Gato", "Tiza Magica", "Cuerda Infinita", "Chapa Final",
    "Reunión", "Largada", "Reposo", "Arranque",
    "Crux", "Boulder Problem", "King Line", "Test Piece",
    "Roca Madre", "Pared Sur", "Cara Norte", "Pilar Central",
    "Diedro Oculto", "Arista Este", "Placa Lisa", "Techo Imposible",
    "La Travesia", "El Paso Clave", "Secuencia Final", "Bloque Maestro",
    "Proyecto Personal", "Sueño Vertical", "Meta Cumplida", "Nueva Era",
    "Ruta Clasica", "Linea Directa", "Variante", "Extension",
    "Primer Ascenso", "Repeticion", "Encadenamiento", "Flash Dorado",
    "Esfuerzo Maximo", "Control Total", "Fluir", "Dejarse Ir",
    "Inspiracion", "Superacion", "Determinacion", "Perseverancia",
    "Amigos de Roca", "Hermandad Vertical", "Comunidad", "Legado",
    "Siguiente Nivel", "Sin Limites", "Mas Alla", "Infinito",
    "Simplemente Escalar", "Puro Vertical", "Roca Pura", "Al Natural",
    "Conexion", "Armonia", "Ritmo", "Cadencia",
    "Empuje Final", "Arranque Explosivo", "Transicion", "Recuperacion",
]

# Grados de escalada
SPORT_GRADES = [
    "5.6", "5.7", "5.8", "5.9",
    "5.10a", "5.10b", "5.10c", "5.10d",
    "5.11a", "5.11b", "5.11c", "5.11d",
    "5.12a", "5.12b", "5.12c", "5.12d",
    "5.13a", "5.13b",
]

BOULDER_GRADES = [
    "V0", "V1", "V2", "V3", "V4", "V5",
    "V6", "V7", "V8", "V9", "V10",
]

STYLES = ["onsight", "flash", "redpoint", "repeat"]


# =====================================================================
# Setup Functions
# =====================================================================

async def create_database():
    """Crea la base de datos klimbook_test (la borra si ya existe)."""
    # Conectar a la DB por defecto para poder crear/borrar databases
    conn = await asyncpg.connect(
        **DB_CONFIG,
        database="postgres",
    )

    # Cerrar conexiones existentes a la DB de test
    await conn.execute(f"""
        SELECT pg_terminate_backend(pid) 
        FROM pg_stat_activity 
        WHERE datname = '{DB_NAME}' AND pid <> pg_backend_pid()
    """)

    # Borrar y recrear
    await conn.execute(f"DROP DATABASE IF EXISTS {DB_NAME}")
    await conn.execute(f"CREATE DATABASE {DB_NAME}")
    print(f"[Setup] Base de datos '{DB_NAME}' creada")

    await conn.close()


async def create_readonly_user(conn):
    """Crea el usuario read-only para el agente."""
    # Verificar si el usuario ya existe
    exists = await conn.fetchval(
        "SELECT 1 FROM pg_roles WHERE rolname = $1",
        READONLY_USER,
    )

    if not exists:
        await conn.execute(
            f"CREATE USER {READONLY_USER} WITH PASSWORD '{READONLY_PASSWORD}'"
        )
        print(f"[Setup] Usuario '{READONLY_USER}' creado")
    else:
        print(f"[Setup] Usuario '{READONLY_USER}' ya existe")

    # Otorgar permisos de solo lectura
    await conn.execute(f"GRANT CONNECT ON DATABASE {DB_NAME} TO {READONLY_USER}")
    await conn.execute(f"GRANT USAGE ON SCHEMA public TO {READONLY_USER}")

    # El GRANT SELECT se hace despues de crear las tablas
    print(f"[Setup] Permisos de solo lectura otorgados")


async def create_schema(conn):
    """Crea las tablas del schema de Klimbook."""

    # Habilitar PostGIS
    # Esta extension agrega los tipos GEOMETRY/GEOGRAPHY y las funciones
    # ST_Distance, ST_DWithin, ST_MakePoint, etc.
    await conn.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    print("[Setup] Extension PostGIS habilitada")

    # Tabla: users
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            display_name VARCHAR(100),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            last_active TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)

    # Tabla: crags
    # Un "crag" es una zona de escalada (ej: Potrero Chico, El Cirio)
    # Contiene multiples rutas.
    # La columna 'geom' es un punto geografico (PostGIS) con SRID 4326 (WGS84/GPS).
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS crags (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            region VARCHAR(100),
            geom GEOMETRY(Point, 4326),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)

    # Indice espacial para busquedas geograficas rapidas
    # Sin este indice, ST_DWithin tendria que recorrer TODA la tabla.
    # Con el indice (GIST), PostgreSQL puede descartar rapidamente
    # los puntos que estan lejos.
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_crags_geom 
        ON crags USING GIST (geom)
    """)

    # Tabla: routes
    # Una ruta es un camino de escalada especifico dentro de un crag.
    # Tiene un grado de dificultad y un tipo (sport, boulder, trad).
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS routes (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            grade VARCHAR(10) NOT NULL,
            type VARCHAR(20) NOT NULL CHECK (type IN ('sport', 'boulder', 'trad')),
            crag_id INTEGER REFERENCES crags(id) ON DELETE CASCADE,
            pitches INTEGER DEFAULT 1,
            description TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)

    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_routes_crag 
        ON routes(crag_id)
    """)

    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_routes_type 
        ON routes(type)
    """)

    # Tabla: ascents
    # Un ascenso es cuando un usuario escala una ruta.
    # 'style' indica como la escalo:
    #   - onsight: primera vez, sin informacion previa
    #   - flash: primera vez, con informacion previa (vio a alguien hacerla)
    #   - redpoint: la trabajo previamente y la encadeno
    #   - repeat: ya la habia escalado antes
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS ascents (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            route_id INTEGER REFERENCES routes(id) ON DELETE CASCADE,
            style VARCHAR(20) NOT NULL CHECK (style IN ('onsight', 'flash', 'redpoint', 'repeat')),
            date DATE NOT NULL,
            notes TEXT,
            rating INTEGER CHECK (rating BETWEEN 1 AND 5),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)

    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ascents_user 
        ON ascents(user_id)
    """)

    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ascents_route 
        ON ascents(route_id)
    """)

    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ascents_date 
        ON ascents(date)
    """)

    # Dar permisos SELECT al usuario readonly DESPUES de crear las tablas
    await conn.execute(
        f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {READONLY_USER}"
    )

    print("[Setup] Tablas creadas: users, crags, routes, ascents")
    print("[Setup] Indices creados")


async def seed_data(conn):
    """Pobla las tablas con datos de ejemplo."""
    random.seed(42)  # Para reproducibilidad

    # ---- Insertar usuarios (50) ----
    print("[Seed] Insertando 50 usuarios...")
    for i, username in enumerate(USERNAMES):
        # Distribuir fechas de registro en los ultimos 6 meses
        days_ago = random.randint(1, 180)
        created_at = datetime.now() - timedelta(days=days_ago)

        await conn.execute(
            """
            INSERT INTO users (username, email, display_name, created_at, last_active)
            VALUES ($1, $2, $3, $4, $5)
            """,
            username,
            f"{username}@email.com",
            username.replace("_", " ").title(),
            created_at,
            created_at + timedelta(days=random.randint(0, days_ago)),
        )

    # ---- Insertar crags (20) ----
    print("[Seed] Insertando 20 crags...")
    for name, region, lon, lat in CRAGS_DATA:
        # ST_SetSRID(ST_MakePoint(lon, lat), 4326) crea un punto geografico
        # con el sistema de coordenadas WGS84 (el que usa GPS)
        await conn.execute(
            """
            INSERT INTO crags (name, region, geom)
            VALUES ($1, $2, ST_SetSRID(ST_MakePoint($3, $4), 4326))
            """,
            name,
            region,
            lon,
            lat,
        )

    # ---- Insertar rutas (200) ----
    print("[Seed] Insertando 200 rutas...")
    route_names_copy = list(ROUTE_NAMES[:200])
    random.shuffle(route_names_copy)

    for i in range(200):
        # Asignar cada ruta a un crag aleatorio
        crag_id = random.randint(1, 20)

        # 60% sport, 30% boulder, 10% trad
        route_type = random.choices(
            ["sport", "boulder", "trad"],
            weights=[60, 30, 10],
        )[0]

        # Elegir grado segun tipo
        if route_type == "boulder":
            grade = random.choice(BOULDER_GRADES)
            pitches = 1
        else:
            grade = random.choice(SPORT_GRADES)
            pitches = 1 if route_type == "sport" else random.randint(1, 4)

        name = route_names_copy[i] if i < len(route_names_copy) else f"Ruta {i+1}"

        await conn.execute(
            """
            INSERT INTO routes (name, grade, type, crag_id, pitches)
            VALUES ($1, $2, $3, $4, $5)
            """,
            name,
            grade,
            route_type,
            crag_id,
            pitches,
        )

    # ---- Insertar ascensos (500) ----
    print("[Seed] Insertando 500 ascensos...")

    # Algunos usuarios son mas activos que otros.
    # Los primeros 10 usuarios tienen muchos mas ascensos.
    for i in range(500):
        # Distribuir ascensos: los primeros 10 users tienen mas actividad
        if random.random() < 0.4:
            user_id = random.randint(1, 10)   # usuarios activos
        else:
            user_id = random.randint(1, 50)   # todos

        route_id = random.randint(1, 200)

        # Distribuir estilos
        style = random.choices(
            STYLES,
            weights=[15, 25, 35, 25],  # onsight es mas raro
        )[0]

        # Distribuir fechas en los ultimos 6 meses
        days_ago = random.randint(0, 180)
        ascent_date = (datetime.now() - timedelta(days=days_ago)).date()

        # Rating opcional (no todos los ascensos tienen rating)
        rating = random.choice([None, None, 3, 4, 5])  # 40% sin rating

        # Notas opcionales
        notes_options = [
            None, None, None,  # 60% sin notas
            "Buen dia de escalada",
            "Condiciones perfectas",
            "Me costo bastante el crux",
            "Flash limpio",
            "Segundo intento",
            "Proyecto completado!",
        ]
        notes = random.choice(notes_options)

        await conn.execute(
            """
            INSERT INTO ascents (user_id, route_id, style, date, notes, rating)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            user_id,
            route_id,
            style,
            ascent_date,
            notes,
            rating,
        )

    print("[Seed] Datos insertados correctamente")


async def verify_data(conn):
    """Verifica que los datos se insertaron correctamente."""
    users = await conn.fetchval("SELECT COUNT(*) FROM users")
    crags = await conn.fetchval("SELECT COUNT(*) FROM crags")
    routes = await conn.fetchval("SELECT COUNT(*) FROM routes")
    ascents = await conn.fetchval("SELECT COUNT(*) FROM ascents")

    print(f"\n[Verify] Datos en la base de datos:")
    print(f"  users:   {users}")
    print(f"  crags:   {crags}")
    print(f"  routes:  {routes}")
    print(f"  ascents: {ascents}")

    # Verificar PostGIS
    postgis = await conn.fetchval(
        "SELECT COUNT(*) FROM crags WHERE geom IS NOT NULL"
    )
    print(f"  crags con geom: {postgis}")

    # Query de ejemplo
    top_route = await conn.fetchrow("""
        SELECT r.name, r.grade, COUNT(a.id) as total
        FROM routes r
        JOIN ascents a ON r.id = a.route_id
        GROUP BY r.id, r.name, r.grade
        ORDER BY total DESC
        LIMIT 1
    """)
    if top_route:
        print(f"\n  Ruta mas popular: {top_route['name']} "
              f"({top_route['grade']}) con {top_route['total']} ascensos")

    # Query PostGIS de ejemplo
    nearby = await conn.fetch("""
        SELECT name, 
               ST_Distance(
                   geom::geography, 
                   ST_MakePoint(-98.2063, 19.0414)::geography
               ) / 1000 AS distance_km
        FROM crags
        WHERE ST_DWithin(
            geom::geography,
            ST_MakePoint(-98.2063, 19.0414)::geography,
            100000
        )
        ORDER BY distance_km
        LIMIT 5
    """)
    print(f"\n  Crags cerca de Puebla (100km):")
    for row in nearby:
        print(f"    {row['name']}: {row['distance_km']:.1f} km")


# =====================================================================
# Main
# =====================================================================

async def main():
    print("=" * 60)
    print("  KLIMBOOK TEST DB SETUP")
    print("=" * 60)

    # Paso 1: Crear la base de datos
    await create_database()

    # Paso 2: Conectar a la nueva DB y crear schema + datos
    conn = await asyncpg.connect(
        **DB_CONFIG,
        database=DB_NAME,
    )

    try:
        await create_readonly_user(conn)
        await create_schema(conn)
        await seed_data(conn)
        await verify_data(conn)
    finally:
        await conn.close()

    print(f"\n{'='*60}")
    print(f"  Setup completado. La base de datos '{DB_NAME}' esta lista.")
    print(f"  Ahora ejecuta: python ejercicio7_1_agent.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())