"""
ejercicio2_4.py 
===========================
Ejercicio 2.4 — Prompt chaining manual


Description:
-----------
Cadena de 3 pasos ejecutada manualmente:
- Paso 1 (Haiku, temp=0): recibe un diff de Git y extrae los cambios clave en bullet points
- Paso 2 (Sonnet, temp=0.3): toma los bullet points y genera release notes en markdown
- Paso 3 (Sonnet, temp=0.7): toma las release notes y genera un post para Ko-fi con tono entusiasta
- Mide y compara: tokens usados, costo total, y calidad vs. hacer todo en un solo prompt


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
import time
import json


client = Anthropic()


# ─── Pricing (USD por millón de tokens) ───────────────────────────────────

PRICING = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
}


# ─── Helper: calcular costo ──────────────────────────────────────────────

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calcula el costo en USD de una llamada a la API."""
    pricing = PRICING[model]
    cost_in = (input_tokens / 1_000_000) * pricing["input"]
    cost_out = (output_tokens / 1_000_000) * pricing["output"]
    return cost_in + cost_out


# ─── Helper: llamada a Claude con métricas ────────────────────────────────

def call_claude(
    model: str,
    system: str,
    prompt: str,
    temperature: float,
    max_tokens: int = 2000,
    prefill: str = "",
) -> tuple[str, dict]:
    """
    Llama a Claude y retorna (respuesta, métricas).
    
    Returns:
        tuple: (texto_respuesta, {model, input_tokens, output_tokens, cost, time})
    """
    messages = [{"role": "user", "content": prompt}]
    if prefill:
        messages.append({"role": "assistant", "content": prefill})

    start = time.time()

    response = client.messages.create(
        model=model,
        system=system,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    elapsed = time.time() - start
    text = response.content[0].text

    if prefill:
        text = prefill + text

    metrics = {
        "model": model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost": calculate_cost(model, response.usage.input_tokens, 
                               response.usage.output_tokens),
        "time": elapsed,
        "stop_reason": response.stop_reason,
    }

    return text, metrics


# ─── Step 1: Extraer cambios clave (Haiku, temp=0) ───────────────────────

def step_1_extract(diff: str) -> tuple[str, dict]:
    """
    Paso 1: Recibe un diff de Git y extrae los cambios clave en bullet points.
    Usa Haiku (rápido y barato) con temperature=0 (determinista).
    """
    system = """Eres un desarrollador senior que lee diffs de Git y extrae 
los cambios clave. Sé preciso y conciso. Enfócate en cambios visibles 
para el usuario y cambios técnicos importantes. Ignora espacios en blanco, 
formato y cambios triviales."""

    prompt = f"""Analiza el siguiente diff de Git y extrae los cambios clave 
como bullet points. Agrúpalos por categoría (Features, Fixes, Improvements, Other).
Solo incluye cambios significativos.

<diff>
{diff}
</diff>

formato de salida:
### Features
- bullet point

### Fixes  
- bullet point

### Mejoras
- bullet point"""

    return call_claude(
        model="claude-haiku-4-5-20251001",
        system=system,
        prompt=prompt,
        temperature=0.0,
    )


# ─── Step 2: Generar release notes (Sonnet, temp=0.3) ────────────────────

def step_2_generate(bullet_points: str, version: str = "v2.9.0") -> tuple[str, dict]:
    """
    Paso 2: Toma los bullet points y genera release notes en markdown.
    Usa Sonnet (mejor calidad) con temperature=0.3 (algo de variación).
    """
    system = """Eres un redactor técnico para Klimbook, una red social 
para escaladores. Escribes release notes claras y profesionales en formato 
markdown. Tu audiencia son tanto usuarios técnicos como escaladores casuales."""

    prompt = f"""Usando los siguientes cambios clave, genera release notes 
profesionales en formato markdown para Klimbook {version}.

<changes>
{bullet_points}
</changes>

Requisitos:
- Título: "Klimbook — Notas de Versión {version}"
- Incluye una breve introducción resumiendo la versión
- Agrupa los cambios por categoría con prefijos
- Cada cambio debe ser claro y amigable para el usuario
- Termina con una sección "Actualización" con instrucciones breves de instalación
- Mantén un tono profesional pero amigable"""

    return call_claude(
        model="claude-sonnet-4-20250514",
        system=system,
        prompt=prompt,
        temperature=0.3,
    )


# ─── Step 3: Generar post de Ko-fi (Sonnet, temp=0.7) ────────────────────

def step_3_kofi(release_notes: str, version: str = "v2.9.0") -> tuple[str, dict]:
    """
    Paso 3: Toma las release notes y genera un post para Ko-fi.
    Usa Sonnet con temperature=0.7 (más creativo para tono casual).

    En inglich pq Ko-fi lo uso en ingles
    """
    system = """You are the solo developer of Klimbook, a social network 
for rock climbers. You write Ko-fi posts to update your supporters about 
new releases. Your tone is enthusiastic, casual.

Rules for Ko-fi posts:
- Plain text only, no markdown headers
- Keep it under 1000 characters
- Use emojis sparingly but effectively
- Include a call-to-action for supporters
- Sign off with "— zxxz6"
- Mention that supporters help keep the project alive
- Be genuine and personal, not corporate"""

    prompt = f"""Based on these release notes, write a Ko-fi post 
announcing Klimbook {version} to supporters.

<release_notes>
{release_notes}
</release_notes>

Remember: casual, enthusiastic, grateful tone. Plain text, no markdown. 
Under 1000 characters. Sign with — zxxz6"""

    return call_claude(
        model="claude-sonnet-4-20250514",
        system=system,
        prompt=prompt,
        temperature=0.7,
    )


# ─── Single Prompt (for comparison) ──────────────────────────────────────

def single_prompt(diff: str, version: str = "v2.9.0") -> tuple[str, dict]:
    """
    Hace TODO en un solo prompt para comparar calidad y costo
    vs. el chain de 3 pasos.
    """
    system = """Eres el dev de Klimbook, una red social 
para escaladores. Necesitas procesar un diff de Git y producir dos salidas."""

    prompt = f"""Analiza el siguiente diff de Git y produce DOS salidas:

<diff>
{diff}
</diff>

SALIDA 1 — Release Notes:
Escribe release notes profesionales en markdown para Klimbook {version}.
Incluye prefijos de emoji para cada categoría.

SALIDA 2 — Post de Ko-fi:
Escribe un post casual y entusiasta para Ko-fi anunciando {version}.
Texto plano, menos de 1000 caracteres, firma con — zxxz6.

Separa las dos salidas con "---" """

    return call_claude(
        model="claude-sonnet-4-20250514",
        system=system,
        prompt=prompt,
        temperature=0.5,
        max_tokens=4000,
    )


# ─── Print Metrics ───────────────────────────────────────────────────────

def print_metrics(step_name: str, metrics: dict):
    """Pretty print de métricas de un step."""
    print(f"\n  Métricas [{step_name}]:")
    print(f"     Modelo:   {metrics['model']}")
    print(f"     Tokens:   in={metrics['input_tokens']}, out={metrics['output_tokens']}")
    print(f"     Costo:    ${metrics['cost']:.6f}")
    print(f"     Tiempo:   {metrics['time']:.2f}s")
    print(f"     Stop:     {metrics['stop_reason']}")


def print_summary(all_metrics: list[dict], label: str):
    """Resumen total de un pipeline."""
    total_input = sum(m["input_tokens"] for m in all_metrics)
    total_output = sum(m["output_tokens"] for m in all_metrics)
    total_cost = sum(m["cost"] for m in all_metrics)
    total_time = sum(m["time"] for m in all_metrics)

    print(f"\n{'='*60}")
    print(f" RESUMEN: {label}")
    print(f"{'='*60}")
    print(f"  Total tokens input:  {total_input}")
    print(f"  Total tokens output: {total_output}")
    print(f"  Total tokens:        {total_input + total_output}")
    print(f"  Costo total:         ${total_cost:.6f}")
    print(f"  Tiempo total:        {total_time:.2f}s")
    print(f"  Pasos:               {len(all_metrics)}")


# ─── Test Diff ────────────────────────────────────────────────────────────

TEST_DIFF = """
diff --git a/src/i18n/en.json b/src/i18n/en.json
index 4a3f2b1..8c7d9e3 100644
--- a/src/i18n/en.json
+++ b/src/i18n/en.json
@@ -150,6 +150,52 @@
+    "tutorials": {
+      "home_welcome": "Welcome to Klimbook!",
+      "home_search": "Tap here to search for routes and crags",
+      "home_notifications": "Check your notifications here",
+      "home_menu": "Access settings and more",
+      "book_stats": "View your climbing statistics",
+      "book_ascensions": "Track your ascensions here",
+      "book_projects": "Manage your climbing projects"
+    }

diff --git a/src/components/TutorialOverlay.tsx b/src/components/TutorialOverlay.tsx
new file mode 100644
index 0000000..a1b2c3d
--- /dev/null
+++ b/src/components/TutorialOverlay.tsx
@@ -0,0 +1,145 @@
+import React, { useState, useEffect } from 'react';
+import { View, Text, TouchableOpacity, Animated, Dimensions } from 'react-native';
+
+interface TutorialStep {
+  target: { x: number; y: number; width: number; height: number };
+  title: string;
+  description: string;
+}
+
+interface TutorialOverlayProps {
+  steps: TutorialStep[];
+  onComplete: () => void;
+  onSkip: () => void;
+}
+
+export const TutorialOverlay: React.FC<TutorialOverlayProps> = ({
+  steps, onComplete, onSkip
+}) => {
+  const [currentStep, setCurrentStep] = useState(0);
+  // ... component implementation

diff --git a/src/screens/HomeScreen.tsx b/src/screens/HomeScreen.tsx
index 5e6f7a8..9b0c1d2 100644
--- a/src/screens/HomeScreen.tsx
+++ b/src/screens/HomeScreen.tsx
@@ -45,6 +45,15 @@
+  const [showTutorial, setShowTutorial] = useState(false);
+
+  useEffect(() => {
+    if (preferences?.first_login) {
+      setShowTutorial(true);
+    }
+  }, [preferences]);
+
+  const handleTutorialComplete = async () => {
+    await updatePreferences({ first_login: false });
+    setShowTutorial(false);
+  };

diff --git a/src/navigation/types.ts b/src/navigation/types.ts
index 2c3d4e5..6f7a8b9 100644
--- a/src/navigation/types.ts
+++ b/src/navigation/types.ts
@@ -12,11 +12,13 @@
   MyBook: {
-    screen?: string;
+    screen?: string;
+    showTutorial?: boolean;
   };
   Search: {
-    query?: string;
+    query?: string;
+    showTutorial?: boolean;
   };

diff --git a/src/components/UserMenu.tsx b/src/components/UserMenu.tsx
index 1a2b3c4..5d6e7f8 100644
--- a/src/components/UserMenu.tsx
+++ b/src/components/UserMenu.tsx
@@ -89,6 +89,12 @@
+      <MenuItem
+        icon="school"
+        label={t('settings.tutorial')}
+        onPress={() => {
+          resetTutorial();
+          navigation.navigate('Home', { showTutorial: true });
+        }}
+      />

diff --git a/src/api/preferences.ts b/src/api/preferences.ts
index 3b4c5d6..7e8f9a0 100644
--- a/src/api/preferences.ts
+++ b/src/api/preferences.ts
@@ -15,6 +15,7 @@
 export interface UserPreferences {
   theme: 'light' | 'dark';
   language: string;
+  first_login: boolean;
   notifications: NotificationPreferences;
 }
"""


# ─── Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("PROMPT CHAINING MANUAL — 3 Pasos")
    print("=" * 60)

    # ─── CHAIN: 3 pasos ──────────────────────────────────────────

    all_metrics = []

    # Paso 1: Extraer cambios
    print(f"\n{'─'*60}")
    print(" PASO 1: Extraer cambios clave (Haiku, temp=0)")
    print(f"{'─'*60}")

    bullet_points, m1 = step_1_extract(TEST_DIFF)
    print(f"\n{bullet_points}")
    print_metrics("Paso 1 — Extract", m1)
    all_metrics.append(m1)

    # Paso 2: Generar release notes
    print(f"\n{'─'*60}")
    print(" PASO 2: Generar release notes (Sonnet, temp=0.3)")
    print(f"{'─'*60}")

    release_notes, m2 = step_2_generate(bullet_points)
    print(f"\n{release_notes}")
    print_metrics("Paso 2 — Generate", m2)
    all_metrics.append(m2)

    # Paso 3: Generar post de Ko-fi
    print(f"\n{'─'*60}")
    print(" PASO 3: Generar post Ko-fi (Sonnet, temp=0.7)")
    print(f"{'─'*60}")

    kofi_post, m3 = step_3_kofi(release_notes)
    print(f"\n{kofi_post}")
    print_metrics("Paso 3 — Ko-fi", m3)
    all_metrics.append(m3)

    # Resumen del chain
    print_summary(all_metrics, "CHAIN DE 3 PASOS")

    # ─── SINGLE PROMPT: todo en uno ──────────────────────────────

    print(f"\n\n{'='*60}")
    print("SINGLE PROMPT — Todo en un solo prompt")
    print(f"{'='*60}")

    single_result, m_single = single_prompt(TEST_DIFF)
    print(f"\n{single_result}")
    print_metrics("Single Prompt", m_single)
    print_summary([m_single], "SINGLE PROMPT")

    # ─── COMPARACIÓN ─────────────────────────────────────────────

    chain_cost = sum(m["cost"] for m in all_metrics)
    chain_time = sum(m["time"] for m in all_metrics)
    chain_tokens = sum(m["input_tokens"] + m["output_tokens"] for m in all_metrics)

    single_cost = m_single["cost"]
    single_time = m_single["time"]
    single_tokens = m_single["input_tokens"] + m_single["output_tokens"]

    print(f"\n\n{'='*60}")
    print(" COMPARACIÓN: Chain vs Single Prompt")
    print(f"{'='*60}")
    print(f"                    {'Chain':>12s}  {'Single':>12s}  {'Diferencia':>12s}")
    print(f"  {'─'*52}")
    print(f"  Tokens totales:   {chain_tokens:>12d}  {single_tokens:>12d}  {chain_tokens - single_tokens:>+12d}")
    print(f"  Costo (USD):      ${chain_cost:>11.6f}  ${single_cost:>11.6f}  ${chain_cost - single_cost:>+11.6f}")
    print(f"  Tiempo (s):       {chain_time:>12.2f}  {single_time:>12.2f}  {chain_time - single_time:>+12.2f}")
    print(f"\n   El chain usa modelos diferentes por paso:")
    print(f"     Paso 1: Haiku (barato, rápido) para extracción")
    print(f"     Paso 2: Sonnet (calidad) para generación")
    print(f"     Paso 3: Sonnet (creativo) para Ko-fi")
    print(f"     Single: Sonnet para TODO (más caro pero menos overhead)")
    print(f"\n   Calidad: revisa manualmente si el chain produce")
    print(f"     mejores notas y mejor post de Ko-fi que el single prompt.")
    print(f"     Generalmente el chain es mejor porque cada paso")
    print(f"     tiene un prompt optimizado para SU tarea específica.")