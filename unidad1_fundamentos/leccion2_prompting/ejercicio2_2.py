"""
ejercicio2_2.py 
===========================
Ejercicio 2.2 — Traductor técnico con contexto


Description:
-----------
- Traduce release notes de inglés a español manteniendo terminología técnica
- System prompt con reglas: no traducir nombres de features, mantener markdown,
  preservar emojis
- Few-shot con 3 ejemplos de tus release notes reales de Klimbook
- Glosario de términos que NO deben traducirse (boulder, route, crag, pitch)
- Evaluar calidad comparando con tus traducciones manuales previas


Metadata:
----------
* Author: zxxz6 
* Version: 1.1.0


History:
------------
Author      Date            Description
zxxz6       02/02/2026      Creation

"""

from anthropic import Anthropic

client = Anthropic()

# ─── System Prompt ────────────────────────────────────────────────────────

MD_FORMATTING = """
<format>
# <product_name> — Release Notes <version>
**<release_date>**

We are pleased to announce the release of <product_name> version <version>. 
<overall_summary_of_the_release>

---

## Release Notes

### <feature_or_fix_title>
<detailed_description_of_the_feature_or_fix>

* **<key_change_1>:** <description_of_key_change_1>
* **<key_change_2>:** <description_of_key_change_2>
* **<key_change_3>:** <description_of_key_change_3>

### <testing_and_validation>
<description_of_testing_and_validation_process>

* **<key_test_1>:** <description_of_key_test_1>
* **<key_test_2>:** <description_of_key_test_2>
* **<key_test_3>:** <description_of_key_test_3>

---

## Installation

<installation_instructions>

</format>
"""

GLOSSARY = """
boulder, route, crag, pitch, belay, rappel, 
crimp, sloper, dyno, flash, onsight, redpoint,
send, beta, crux, anchor, quickdraw, carabiner,
top-rope, lead climbing, sport climbing,
bouldering, trad climbing
"""

RULES = f"""
- No traduzcas nombres de features, servicios, o términos técnicos específicos (ej. "auth", "db", "api").
- Mantén el formato markdown original, incluyendo encabezados, listas, y emojis.
- Preserva emojis y símbolos tal cual aparecen en el texto original.
- Traduce solo el texto descriptivo, no los nombres de las features o servicios.
- No traduzcas los terminos del glosario: <glossary>{GLOSSARY}</glossary>
"""

TASK_CONTEXT = """Eres un traductor técnico especializado en desarrollo de 
software. Tu tarea es traducir release notes de inglés a español manteniendo la 
terminología técnica precisa y el formato original."""

TASK_DESCRIPTION = f""" Las release notes son testos que describen las nuevas 
features, fixes, y cambios en una nueva versión de software.
Tu objetivo es traducir estas release notes del inglés al español, el formato 
suele ser markdown y la estructura es la siguiente:

<format>{MD_FORMATTING}</format>

Sigue estas reglas importantes:
<rules>
{RULES}
</rules>
"""


SYSTEM_PROMPT = f"{TASK_CONTEXT}\n\n{TASK_DESCRIPTION}\n\n"


# ─── Few-Shot Messages ────────────────────────────────────────────────────

NSHOT = [
    {
        "role": "user",
        "content": """ Traduce estas release notes de ingles a español:
        # Klimbook Web — Release Notes v2.9.0
        **April 3, 2026**

        We are pleased to announce the release of Klimbook Web version 2.9.0. This release adds backend support for the onboarding tutorial system on mobile.

        ---

        ## Release Notes

        ### Onboarding Tutorial Support

        Backend support for triggering and tracking the onboarding tutorial on first login.

        * **New `first_login` Preference Field:** Added `first_login: True` to user preferences defaults. When a new user registers, their preferences default to `first_login: True`. The mobile app reads this flag to trigger the onboarding tutorial, then PATCHes it to `False` after completion.
        * **Schema Updates:** `PreferencesResponse` exposes `first_login` in GET responses. `PreferencesUpdateRequest` accepts `first_login` via PATCH.
        * **No Migration Needed:** `first_login` is stored inside the existing `User.preferences` JSONB column. Existing users without the key get `True` via the default merge in `_get_preferences()`.

        ### Test Coverage

        * **2 New Tests:** `test_get_default_first_login` and `test_update_first_login` verify default value and PATCH behavior.
        * **2 Updated Tests:** `test_get_default_preferences` and `test_empty_patch_preserves_defaults` updated to assert `first_login` key.

        ---

        ## Installation

        This version is available on the web platform. No installation required.

        **Full Changelog**: https://github.com/zxxz456/klimbook/compare/v2.8.0...v2.9.0
        
        """
    },
    {
        "role": "assistant",
        "content": """
        # Klimbook Web — Notas de Versión v2.9.0
        **3 de abril de 2026**
        Nos complace anunciar el lanzamiento de Klimbook Web versión 2.9.0. Esta versión añade soporte en el backend para el sistema de tutorial de incorporación en móvil.
        ---
        ## Notas de Versión
        ### Soporte para Tutorial de Incorporación
        Soporte en el backend para activar y rastrear el tutorial de incorporación en el primer inicio de sesión.
        * **Nuevo campo de preferencia `first_login`:** Se añadió `first_login: True` a los valores predeterminados de preferencias del usuario. Cuando un nuevo usuario se registra, sus preferencias se establecen por defecto en `first_login: True`. La aplicación móvil lee esta bandera para activar el tutorial de incorporación y luego la actualiza a `False` mediante PATCH tras completarlo.
        * **Actualizaciones de esquema:** `PreferencesResponse` expone `first_login` en las respuestas GET. `PreferencesUpdateRequest` acepta `first_login` mediante PATCH.
        * **No se requiere migración:** `first_login` se almacena dentro de la columna JSONB existente `User.preferences`. Los usuarios existentes sin esta clave obtienen `True` a través de la fusión de valores predeterminados en `_get_preferences()`.
        ### Cobertura de Pruebas
        * **2 pruebas nuevas:** `test_get_default_first_login` y `test_update_first_login` verifican el valor predeterminado y el comportamiento del PATCH.
        * **2 pruebas actualizadas:** `test_get_default_preferences` y `test_empty_patch_preserves_defaults` actualizadas para verificar la clave `first_login`.
        ---
        ## Instalación
        Esta versión está disponible en la plataforma web. No se requiere instalación.
        **Registro de cambios completo**: https://github.com/zxxz456/klimbook/compare/v2.8.0...v2.9.0
        
        """
    },
    {
        "role": "user",
        "content": """ Traduce al espanol:
        # Klimbook Mobile — Release Notes v2.9.0
        **April 3, 2026**

        We are pleased to announce the release of Klimbook Mobile version 2.9.0. This release introduces a full onboarding tutorial system that guides new users through the app's main features on first login.

        ---

        ## Release Notes

        ### Onboarding Tutorial System

        A multi-screen guided tour that introduces new users to the app's core features.

        * **`TutorialOverlay` Component:** Full-screen semi-transparent overlay that highlights UI elements with a spotlight cutout. Supports multi-step flows with next/skip navigation, title/description callouts, and scroll-to-target positioning.
        * **HomeScreen Tutorial:** 5-step guided tour covering TopBar icons (search, notifications, menu) and BottomNav items (home, book, bookmarks, profile, community). Auto-triggers on first login when `first_login` preference is `True`; sets `first_login: false` via `updatePreferences()` after completion.
        * **MyBookScreen Tutorial:** Multi-step tour highlighting Stats, Ascensions, and Projects tabs. Launched via `showTutorial` route param from HomeScreen tutorial flow.
        * **SearchScreen Tutorial:** Highlights the search bar and map button. Continues the tutorial chain from MyBookScreen.
        * **VisitedProfileScreen Tutorial:** Highlights the "Book" tab when visiting another user's profile. Auto-navigates to a venue profile to continue the flow.
        * **VisitedBookScreen Tutorial:** Highlights the action menu (rewards/reviews) and wall cards.
        * **VisitedBookBlocksScreen Tutorial:** Highlights the missing block report card and the first block card.
        * **Tutorial Restart:** New "Tutorial" menu item in `UserMenu` with `school` icon, allowing users to replay the onboarding at any time.

        ### Component Updates

        * **TopBar:** Added layout measurement callbacks (`onSearchIconLayout`, `onNotificationLayout`, `onMenuLayout`) for precise spotlight positioning.
        * **BottomNav:** Added layout measurement callbacks (`onHomeLayout`, `onBookLayout`, `onBookmarksLayout`, `onProfileLayout`, `onCommunityLayout`) for spotlight targeting.
        * **VisitedBlocksTab:** Added `onMissingBlockLayout` and `onFirstBlockLayout` callbacks for tutorial spotlight targeting on block cards.
        * **Navigation:** Added optional `showTutorial` param to 6 route types for chaining tutorial steps across screens.

        ### i18n Updates

        * **45+ New Translation Keys:** Full `tutorials` namespace added to both `en.json` and `es.json`, covering all tutorial step titles and descriptions across every screen in the flow. Added `settings.tutorial` key for the menu item.

        ---

        ## Installation

        ### iOS
        Download the attached `.ipa` file to install the application on registered devices.

        ### Android
        Download the attached `.apk` file to install the application on registered devices.

        **Full Changelog**: https://github.com/zxxz456/klimbook-mobile/compare/v2.8.0...v2.9.0
        """
    },
    {
        "role": "assistant",
        "content": """
        # Klimbook Mobile — Notas de Versión v2.9.0
        **3 de abril de 2026**
        Nos complace anunciar el lanzamiento de Klimbook Mobile versión 2.9.0. Esta versión introduce un sistema completo de tutorial de incorporación que guía a los nuevos usuarios a través de las funciones principales de la aplicación en su primer inicio de sesión.
        ---
        ## Notas de Versión
        ### Sistema de Tutorial de Incorporación
        Un recorrido guiado de múltiples pantallas que presenta a los nuevos usuarios las funciones principales de la aplicación.
        * **Componente `TutorialOverlay`:** Superposición semitransparente de pantalla completa que resalta elementos de la interfaz con un recorte tipo spotlight. Soporta flujos de múltiples pasos con navegación siguiente/omitir, indicaciones con título/descripción y posicionamiento con desplazamiento hacia el elemento objetivo.
        * **Tutorial de HomeScreen:** Recorrido guiado de 5 pasos que cubre los íconos de la barra superior (búsqueda, notificaciones, menú) y los elementos de la barra de navegación inferior (inicio, libro, marcadores, perfil, comunidad). Se activa automáticamente en el primer inicio de sesión cuando la preferencia `first_login` es `True`; establece `first_login: false` mediante `updatePreferences()` al completarse.
        * **Tutorial de MyBookScreen:** Recorrido de múltiples pasos que resalta las pestañas de Estadísticas, Ascensiones y Proyectos. Se lanza mediante el parámetro de ruta `showTutorial` desde el flujo del tutorial de HomeScreen.
        * **Tutorial de SearchScreen:** Resalta la barra de búsqueda y el botón del mapa. Continúa la cadena del tutorial desde MyBookScreen.
        * **Tutorial de VisitedProfileScreen:** Resalta la pestaña "Libro" al visitar el perfil de otro usuario. Navega automáticamente al perfil de un establecimiento para continuar el flujo.
        * **Tutorial de VisitedBookScreen:** Resalta el menú de acciones (recompensas/reseñas) y las tarjetas del muro.
        * **Tutorial de VisitedBookBlocksScreen:** Resalta la tarjeta de reporte de bloque faltante y la primera tarjeta de bloque.
        * **Reinicio del Tutorial:** Nuevo elemento de menú "Tutorial" en `UserMenu` con ícono `school`, que permite a los usuarios repetir la incorporación en cualquier momento.
        ### Actualizaciones de Componentes
        * **TopBar:** Se añadieron callbacks de medición de layout (`onSearchIconLayout`, `onNotificationLayout`, `onMenuLayout`) para posicionamiento preciso del spotlight.
        * **BottomNav:** Se añadieron callbacks de medición de layout (`onHomeLayout`, `onBookLayout`, `onBookmarksLayout`, `onProfileLayout`, `onCommunityLayout`) para el enfoque del spotlight.
        * **VisitedBlocksTab:** Se añadieron callbacks `onMissingBlockLayout` y `onFirstBlockLayout` para el enfoque del spotlight en las tarjetas de bloques del tutorial.
        * **Navegación:** Se añadió el parámetro opcional `showTutorial` a 6 tipos de ruta para encadenar los pasos del tutorial entre pantallas.
        ### Actualizaciones de i18n
        * **Más de 45 nuevas claves de traducción:** Se añadió el namespace completo `tutorials` tanto a `en.json` como a `es.json`, cubriendo todos los títulos y descripciones de los pasos del tutorial en cada pantalla del flujo. Se añadió la clave `settings.tutorial` para el elemento del menú.
        ---
        ## Instalación
        ### iOS
        Descarga el archivo `.ipa` adjunto para instalar la aplicación en dispositivos registrados.
        ### Android
        Descarga el archivo `.apk` adjunto para instalar la aplicación en dispositivos registrados.
        **Registro de cambios completo**: https://github.com/zxxz456/klimbook-mobile/compare/v2.8.0...v2.9.0
        """
    }
]

def translate_release_notes(messages):
    """
    Traduce release notes del inglés al español usando Claude.

    Args:
        messages: Lista de mensajes con las release notes a traducir

    Returns:
        Texto traducido al español manteniendo formato markdown y terminología técnica
    """
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        system=SYSTEM_PROMPT,
        messages=NSHOT+messages,
        temperature=0.3,
        max_tokens=2000
    )

    return response.content[0].text


if __name__ == "__main__":
    TEST_RELEASE_NOTES = """
        # Klimbook Mobile — Release Notes v2.6.0
        **March 21, 2026**

        We are pleased to announce the release of Klimbook Mobile version 2.6.0. This release introduces full multi-language support, allowing users to choose their preferred language from the app settings.

        ---

        ## Release Notes

        ### Full Multi-Language Support

        The entire app UI is now available in English and Spanish, with more languages planned.

        * **i18next Integration:** New `src/i18n/` directory with `index.ts` (i18next + react-i18next initialization), `en.json` and `es.json` translation files (~1,030 keys each). Covers all UI strings: screens, modals, buttons, labels, error messages, placeholders, and notification descriptions.
        * **Language Persistence:** Language preference persisted via AsyncStorage (`klimbook-language` key). Restored on app startup, synced on every `languageChanged` event.
        * **Zustand Store Integration:** Language preference read from auth store and synced with i18next on login.

        ---

        ## Installation

        ### iOS
        Download the attached `.ipa` file to install the application on registered devices.

        ### Android
        Download the attached `.apk` file to install the application on registered devices.


    """

    translated = translate_release_notes([
        {"role": "user", "content": TEST_RELEASE_NOTES}
    ])
    print(translated)