"""
ejercicio2.py 
===========================
Ejercicio 1.2 — Conversación multi-turno


Description:
-----------
Chatbot de terminal que mantenga contexto entre turnos:

Mantener una lista messages = [] que crece con cada turno (user + assistant)
Mostrar el conteo de tokens en cada turno (response.usage.input_tokens, output_tokens)
Implementar comandos especiales: /clear (resetea historial), /system (cambia system prompt), /tokens (muestra uso acumulado)
Calcular y mostrar el costo estimado acumulado de la conversación


Metadata:
----------
* Author: zxxz6 
* Version: 1.0.0


History:
------------
Author      Date            Description
zxxz6       01/02/2026      Creation

"""

from anthropic import Anthropic


cliente = Anthropic()


# ─── Constants ────────────────────────────────────────────────────────────

# Pricing por millón de tokens (Haiku 4.5)
PRICING = {"input": 0.80, "output": 4.00}

# ─── Main Function ────────────────────────────────────────────────────────

def chatbot(model: str):
    """
    Chatbot de terminal que mantiene contexto entre turnos.
    
    Soporta comandos especiales: /clear, /system, /tokens, /quit.
    Muestra conteo de tokens y costo estimado acumulado.
    
    Args:
        model: Modelo de Claude a utilizar
    """
    historial = []
    system_prompt = "Eres un escalador therian"
    tokens_acumulados = {"input": 0, "output": 0}

    print("Chatbot listo. Comandos: /clear /system /tokens /quit")

    while True:
        user_input = input("Tú: ")

        if user_input.startswith("/"):
            comando = user_input[1:].strip()

            if comando == "clear":
                historial = []
                print("Historial reseteado.")
                continue

            elif comando == "quit":
                print("Hasta luego!")
                break

            elif comando.startswith("system"):
                nuevo = comando[6:].strip()
                if nuevo:
                    system_prompt = nuevo
                    print(f"System prompt: {system_prompt}")
                else:
                    print(f"System actual: {system_prompt}")
                continue

            elif comando == "tokens":
                costo_in = tokens_acumulados["input"] / 1_000_000 * PRICING["input"]
                costo_out = tokens_acumulados["output"] / 1_000_000 * PRICING["output"]
                print(f"Input: {tokens_acumulados['input']} tokens")
                print(f"Output: {tokens_acumulados['output']} tokens")
                print(f"Costo estimado: ${costo_in + costo_out:.6f}")
                continue

            else:
                print(f"Comando desconocido: /{comando}")
                continue

        historial.append({"role": "user", "content": user_input})

        respuesta = cliente.messages.create(
            model=model,
            messages=historial,
            temperature=0.80,
            system=system_prompt,
            max_tokens=512
        )

        texto = respuesta.content[0].text
        historial.append({"role": "assistant", "content": texto})

        tokens_acumulados["input"] += respuesta.usage.input_tokens
        tokens_acumulados["output"] += respuesta.usage.output_tokens

        print(f"Claude: {texto}")
        print(f"  [in: {respuesta.usage.input_tokens}, "
              f"out: {respuesta.usage.output_tokens}, "
              f"stop: {respuesta.stop_reason}]")

# ─── Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    chatbot("claude-haiku-4-5-20251001")