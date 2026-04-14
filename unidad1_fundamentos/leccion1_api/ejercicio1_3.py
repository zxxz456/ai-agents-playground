"""
ejercicio3.py 
===========================
Ejercicio 1.3 — Streaming en terminal


Description:
-----------
Modifica el chatbot del 1.2 para usar streaming:

Mostrar la respuesta token por token con efecto de typing
Manejar Ctrl+C para cancelar una respuesta en progreso sin crashear
Comparar la latencia: medir time-to-first-token con streaming vs tiempo total sin streaming


Metadata:
----------
* Author: zxxz6 
* Version: 1.0.0


History:
------------
Author      Date            Description
zxxz6       01/02/2026      Creation

"""

import asyncio
import time
from anthropic import AsyncAnthropic


aclient = AsyncAnthropic()


# ─── Main Function ────────────────────────────────────────────────────────

async def main():
    """
    Chatbot con streaming que muestra respuestas token por token.
    
    Mide time-to-first-token (TTFT) y tiempo total por respuesta.
    Soporta comandos: salir, /clear, /tokens.
    Maneja Ctrl+C para cancelar respuestas en progreso.
    """
    historial = []
    system_prompt = "Eres un instructor de escalada en Puebla."
    tokens_totales = {"input": 0, "output": 0}

    while True:
        input_usuario = input("Yo: ")

        if input_usuario.lower() == "salir":
            break
        if input_usuario == "/clear":
            historial = []
            print("Historial limpiado.")
            continue
        if input_usuario == "/tokens":
            print(f"Tokens — input: {tokens_totales['input']}, "
                  f"output: {tokens_totales['output']}")
            continue

        historial.append({"role": "user", "content": input_usuario})

        try:
            start = time.time()
            ttft = None

            async with aclient.messages.stream(
                model="claude-haiku-4-5-20251001",
                system=system_prompt,
                max_tokens=512,
                messages=historial,
                temperature=0.7,
            ) as stream:
                respuesta = ""
                async for fragment in stream.text_stream:
                    if ttft is None:
                        ttft = time.time() - start
                    print(fragment, end="", flush=True)
                    respuesta += fragment

                print()

                # Obtener uso de tokens del mensaje final
                final = await stream.get_final_message()
                tokens_totales["input"] += final.usage.input_tokens
                tokens_totales["output"] += final.usage.output_tokens

            total_time = time.time() - start
            print(f"  [TTFT: {ttft:.3f}s | Total: {total_time:.3f}s | "
                  f"Tokens in: {final.usage.input_tokens}, "
                  f"out: {final.usage.output_tokens}]")

            historial.append({"role": "assistant", "content": respuesta})

        except KeyboardInterrupt:
            print("\n[Respuesta cancelada]")
            # Remover el último mensaje de user si no hay respuesta
            if historial and historial[-1]["role"] == "user":
                historial.pop()
            continue

# ─── Entry Point ──────────────────────────────────────────────────────────

asyncio.run(main())