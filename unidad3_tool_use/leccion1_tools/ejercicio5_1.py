"""
ejercicio5_1.py 
===========================
Ejercicio 5.1 — Calculator tool


Description:
-----------
- Define una tool 'calculate' con input_schema: {expression: string, precision: integer}
- Implementa la ejecución segura: parsear la expresión con ast.literal_eval o 
  un evaluador seguro, NO con eval()
- El agente debe resolver: "¿Si tengo 150 rutas y el 23% son boulder, cuántas 
  son sport climbing?"
- Manejar errores: división por cero, expresiones inválidas, overflow


Metadata:
----------
* Author: zxxz6 
* Version: 1.0.0


History:
------------
Author      Date            Description
zxxz6       20/02/2026      Creation

"""

from anthropic import Anthropic
import json
import ast
import operator
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("calculator-agent")

client = Anthropic()

MODEL = "claude-sonnet-4-20250514"


# ═══════════════════════════════════════════════════════════════════════════
# Safe Math Evaluator
# ═══════════════════════════════════════════════════════════════════════════

# Operadores permitidos
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Límites de seguridad
MAX_RESULT = 1e15
MAX_POWER = 100


def safe_eval(expression: str) -> float:
    """
    Evalúa una expresión matemática de forma segura usando AST parsing.
    
    Solo permite: números, +, -, *, /, //, %, ** y paréntesis.
    NO permite: variables, funciones, imports, strings, o cualquier
    otra cosa que podría ser peligrosa.
    
    Args:
        expression: String con la expresión matemática
        
    Returns:
        Resultado numérico
        
    Raises:
        ValueError: Si la expresión es inválida o insegura
        ZeroDivisionError: Si hay división por cero
        OverflowError: Si el resultado excede los límites
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Expresión inválida: {e}")

    return _eval_node(tree.body)


def _eval_node(node) -> float:
    """Evalúa un nodo del AST recursivamente."""

    # Números literales (int, float)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Tipo no permitido: {type(node.value).__name__}")

    # Operaciones binarias: 1 + 2, 3 * 4, etc.
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)

        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Operador no permitido: {op_type.__name__}")

        left = _eval_node(node.left)
        right = _eval_node(node.right)

        # Protección contra potencias gigantes
        if op_type == ast.Pow:
            if abs(right) > MAX_POWER:
                raise OverflowError(
                    f"Exponente demasiado grande: {right} (máximo: {MAX_POWER})"
                )

        # Protección contra división por cero
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ZeroDivisionError("División por cero")

        result = SAFE_OPERATORS[op_type](left, right)

        # Protección contra overflow
        if abs(result) > MAX_RESULT:
            raise OverflowError(
                f"Resultado demasiado grande: {result:.2e} (máximo: {MAX_RESULT:.0e})"
            )

        return float(result)

    # Operaciones unarias: -5, +3
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Operador unario no permitido: {op_type.__name__}")
        operand = _eval_node(node.operand)
        return float(SAFE_OPERATORS[op_type](operand))

    else:
        raise ValueError(
            f"Elemento no permitido en la expresión: {type(node).__name__}"
        )


def calculate(expression: str, precision: int = 2) -> dict:
    """
    Función que ejecuta el cálculo de forma segura.
    Esta es la función que se ejecuta cuando Claude llama la tool.
    
    Args:
        expression: Expresión matemática como string
        precision: Decimales en el resultado
        
    Returns:
        Dict con resultado o error
    """
    logger.info(f"  [Calculator] Evaluando: '{expression}' (precision={precision})")

    try:
        result = safe_eval(expression)
        rounded = round(result, precision)
        logger.info(f"  [Calculator] Resultado: {rounded}")
        return {
            "expression": expression,
            "result": rounded,
            "precision": precision,
        }

    except ZeroDivisionError as e:
        logger.warning(f"  [Calculator] Error: {e}")
        return {"error": f"División por cero: {expression}"}

    except OverflowError as e:
        logger.warning(f"  [Calculator] Error: {e}")
        return {"error": str(e)}

    except ValueError as e:
        logger.warning(f"  [Calculator] Error: {e}")
        return {"error": f"Expresión inválida: {e}"}

    except Exception as e:
        logger.error(f"  [Calculator] Error inesperado: {e}")
        return {"error": f"Error inesperado: {type(e).__name__}: {e}"}


# ═══════════════════════════════════════════════════════════════════════════
# Tool Definition
# ═══════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "calculate",
        "description": (
            "Performs mathematical calculations safely. Use this tool whenever "
            "you need to compute arithmetic operations like addition, subtraction, "
            "multiplication, division, modulo, or exponentiation. "
            "Supports: numbers, +, -, *, /, //, %, **, and parentheses. "
            "Does NOT support variables or functions like sqrt(), sin(), etc. "
            "For percentages, convert them first: 23% of 150 → 150 * 23 / 100. "
            "Example: calculate(expression='150 * 23 / 100', precision=0)"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "Mathematical expression to evaluate. "
                        "Use standard operators: +, -, *, /, //, %, **. "
                        "Use parentheses for grouping. "
                        "Examples: '150 * 0.23', '(100 - 23) / 4', '2 ** 10'"
                    ),
                },
                "precision": {
                    "type": "integer",
                    "description": (
                        "Number of decimal places in the result (default: 2). "
                        "Use 0 for whole numbers."
                    ),
                    "default": 2,
                },
            },
            "required": ["expression"],
        },
    }
]


# ═══════════════════════════════════════════════════════════════════════════
# Tool Executor
# ═══════════════════════════════════════════════════════════════════════════

TOOL_FUNCTIONS = {
    "calculate": calculate,
}

# Rate limiting
tool_call_counts = {}
MAX_CALLS_PER_CONVERSATION = 10


def execute_tool(name: str, input_data: dict) -> str:
    """Ejecuta una tool de forma segura con rate limiting."""

    # Verificar whitelist
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return json.dumps({"error": f"Tool desconocida: {name}"})

    # Rate limiting
    count = tool_call_counts.get(name, 0)
    if count >= MAX_CALLS_PER_CONVERSATION:
        return json.dumps({
            "error": f"Rate limit alcanzado: máximo {MAX_CALLS_PER_CONVERSATION} "
                     f"llamadas a '{name}' por conversación"
        })
    tool_call_counts[name] = count + 1

    # Ejecutar
    try:
        result = func(**input_data)
        return json.dumps(result)
    except TypeError as e:
        return json.dumps({"error": f"Parámetros inválidos: {e}"})
    except Exception as e:
        return json.dumps({"error": f"Error: {type(e).__name__}: {e}"})


# ═══════════════════════════════════════════════════════════════════════════
# Agent (single question, handles tool loop)
# ═══════════════════════════════════════════════════════════════════════════

def ask(question: str) -> str:
    """
    Envía una pregunta a Claude con la calculator tool.
    Maneja el loop de tool_use hasta que Claude da la respuesta final.
    
    Args:
        question: Pregunta del usuario
        
    Returns:
        Respuesta final de Claude
    """
    logger.info(f"\n{'═'*60}")
    logger.info(f"[Agent] Pregunta: \"{question}\"")
    logger.info(f"{'═'*60}")

    messages = [{"role": "user", "content": question}]
    total_tokens = {"input": 0, "output": 0}
    step = 0
    start = time.time()

    while True:
        step += 1
        logger.info(f"\n  [Step {step}] Llamando a Claude...")

        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        total_tokens["input"] += response.usage.input_tokens
        total_tokens["output"] += response.usage.output_tokens

        # Mostrar lo que Claude dijo/hizo
        for block in response.content:
            if block.type == "text" and block.text.strip():
                logger.info(f"  [Step {step}] Claude: \"{block.text[:150]}...\"")
            elif block.type == "tool_use":
                logger.info(
                    f"  [Step {step}] Tool call: {block.name}("
                    f"{json.dumps(block.input)})"
                )

        # Si Claude terminó, retornar la respuesta
        if response.stop_reason == "end_turn":
            elapsed = time.time() - start
            final_text = ""
            for block in response.content:
                if block.type == "text":
                    final_text += block.text

            logger.info(f"\n  [Agent] Completado en {step} steps | {elapsed:.2f}s")
            logger.info(
                f"  [Agent] Tokens → in: {total_tokens['input']}, "
                f"out: {total_tokens['output']}"
            )
            return final_text

        # Si Claude quiere usar tools, ejecutarlas
        if response.stop_reason == "tool_use":
            # Agregar la respuesta de Claude al historial
            messages.append({"role": "assistant", "content": response.content})

            # Ejecutar todas las tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    logger.info(f"  [Step {step}] Resultado: {result}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Enviar resultados de vuelta
            messages.append({"role": "user", "content": tool_results})

        # Protección contra loops infinitos
        if step >= 10:
            logger.error("  [Agent] Maximo de steps alcanzado (10)")
            return "Error: No pude resolver la pregunta en 10 pasos."


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_safe_eval():
    """Tests para el evaluador seguro."""
    print("\n" + "=" * 60)
    print("  TESTS: Safe Evaluator")
    print("=" * 60)

    # Operaciones básicas
    tests = [
        ("2 + 3", 5.0),
        ("10 - 4", 6.0),
        ("6 * 7", 42.0),
        ("15 / 4", 3.75),
        ("15 // 4", 3.0),
        ("15 % 4", 3.0),
        ("2 ** 10", 1024.0),
        ("(10 + 5) * 3", 45.0),
        ("-5 + 3", -2.0),
        ("150 * 23 / 100", 34.5),
        ("150 - (150 * 23 / 100)", 115.5),
    ]

    passed = 0
    for expr, expected in tests:
        try:
            result = safe_eval(expr)
            status = "PASS" if abs(result - expected) < 0.001 else "FAIL"
            if status == "PASS":
                passed += 1
            print(f"  {status} {expr} = {result} (expected {expected})")
        except Exception as e:
            print(f"  FAIL {expr} -> ERROR: {e}")

    # Tests que DEBEN fallar
    error_tests = [
        ("1 / 0", "ZeroDivisionError"),
        ("2 ** 200", "OverflowError"),
        ("import os", "ValueError"),
        ("os.system('ls')", "ValueError"),
        ("__import__('os')", "ValueError"),
        ("'hello'", "ValueError"),
        ("x + 1", "ValueError"),
    ]

    for expr, expected_error in error_tests:
        try:
            result = safe_eval(expr)
            print(f"  FAIL {expr} -> deberia fallar pero retorno {result}")
        except Exception as e:
            error_type = type(e).__name__
            status = "PASS" if error_type == expected_error else "WARN"
            if status == "PASS":
                passed += 1
            print(f"  {status} {expr} -> {error_type}: {e}")

    total = len(tests) + len(error_tests)
    print(f"\n  Resultado: {passed}/{total} tests pasaron")
    return passed == total


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Primero correr tests del evaluador
    all_passed = test_safe_eval()

    if not all_passed:
        print("\nAlgunos tests fallaron. Revisa el evaluador.")

    # Preguntas de prueba
    QUESTIONS = [
        # La pregunta del ejercicio
        "Si tengo 150 rutas y el 23% son boulder, ¿cuántas son sport climbing?",

        # Cálculos simples
        "¿Cuánto es 847 * 23?",

        # Múltiples cálculos
        "Si escalo 3 rutas de 5.10a, 2 rutas de 5.11b, y 4 rutas de 5.12a, "
        "¿cuántas rutas escalé en total y cuál es el promedio por grado?",

        # Porcentajes
        "Tengo 500 ascensos. El 40% son flash, el 35% son redpoint, y el resto "
        "son repeat. ¿Cuántos de cada tipo tengo?",

        # Pregunta que NO necesita calculator
        "¿Qué es un dyno en escalada?",

        # Errores esperados
        "¿Cuánto es 100 dividido entre 0?",
    ]

    for question in QUESTIONS:
        answer = ask(question)
        print(f"\n  Respuesta: {answer}")
        print(f"{'─'*60}")

        # Reset rate limits entre preguntas
        tool_call_counts.clear()