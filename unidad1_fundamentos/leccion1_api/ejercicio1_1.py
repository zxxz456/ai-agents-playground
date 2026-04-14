"""
ejercicio1.py 
===========================
Ejercicio 1.1 — Hello Claude


Description:
-----------
Script Python que use el SDK de Anthropic. Experimenta con:

Cambiar entre claude-sonnet-4-20250514 y claude-haiku-4-5-20251001 — observa diferencias en calidad y velocidad
Variar temperature de 0.0 a 1.0 con el mismo prompt — genera 5 respuestas con cada valor y compara
Probar diferentes max_tokens (50, 200, 1000) — observa cómo afecta la respuesta y el stop_reason
Usar un system prompt: "Eres un instructor de escalada en Puebla"


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


client = Anthropic()


# ─── Request Function ─────────────────────────────────────────────────────

def req(model, temperature, max_tokens):
    """
    Realiza una solicitud a la API de Claude con los parámetros dados.
    
    Args:
        model: Modelo de Claude a utilizar
        temperature: Valor de temperatura para la generación
        max_tokens: Número máximo de tokens en la respuesta
    """
    respuesta = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system="Eres un instructor de escalada en Puebla",
        messages=[
            {
                "role": "user",
                "content": "Q onda? Responde una palabra"
            }
        ]
    )
    print(f"Modelo: {model}, Temperature: {temperature}, Max Tokens: {max_tokens}")
    print("Respuesta:", respuesta.content[0].text)
    print("Stop Reason:", respuesta.content[0].text)
    print("-" * 50)

# ─── Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":

    cliente = Anthropic()

    modelos = ["claude-3-haiku-20240307", "claude-sonnet-4-20250514"]
    temperatures = [0.0, 0.5, 1.0]
    max_tokens_list = [50, 200, 1000]

    for model in modelos:
        for temp in temperatures:
            for max_tokens in max_tokens_list:
                req(model, temp, max_tokens)