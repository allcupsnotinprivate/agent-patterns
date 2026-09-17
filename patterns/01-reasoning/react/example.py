"""Минимальная реализация паттерна ReAct на голом Anthropic Messages API.

ReAct = Reasoning + Acting: агент чередует рассуждение (Thought), действие
(Action - вызов инструмента) и наблюдение (Observation - результат инструмента),
пока не сможет дать финальный ответ.

Как паттерн ложится на нативный tool-calling:
  - Thought      -> текстовый блок, который модель пишет перед вызовом;
  - Action       -> блок tool_use в ответе модели;
  - Observation  -> блок tool_result, который мы отправляем обратно.

Цикл крутится, пока stop_reason == "tool_use".

Запуск:
    export ANTHROPIC_API_KEY=...   # или `ant auth login`
    python example.py
"""

from __future__ import annotations

import ast
import operator

import anthropic

MODEL = "claude-opus-4-8"

SYSTEM = (
    "Ты - агент, работающий по циклу ReAct. На каждом шаге сначала коротко "
    "рассуждай, какой шаг сделать дальше, затем при необходимости вызывай "
    "инструмент. Опирайся на наблюдения из инструментов, а не на догадки. "
    "Когда информации достаточно - дай финальный ответ без вызова инструментов."
)

# --- Инструменты (пространство действий агента) ----------------------------

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """Считает арифметику без eval: только + - * / **, унарный минус, скобки."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("неподдерживаемое выражение")


def calculator(expression: str) -> str:
    try:
        return str(_safe_eval(ast.parse(expression, mode="eval")))
    except Exception as exc:  # вернем ошибку модели как наблюдение, а не упадем
        return f"Ошибка вычисления: {exc}"


# Заглушка "поиска": в реальном агенте здесь был бы веб-поиск или RAG.
_FACTS = {
    "радиус земли": "Средний радиус Земли - 6371 км.",
    "скорость света": "Скорость света в вакууме - 299792458 м/с.",
}


def web_search(query: str) -> str:
    for key, fact in _FACTS.items():
        if key in query.lower():
            return fact
    return "Ничего не найдено."


TOOLS = [
    {
        "name": "calculator",
        "description": "Вычисляет арифметическое выражение (+, -, *, /, **, скобки).",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Например: (6371*2)*3.14159",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "web_search",
        "description": "Ищет факт по короткому запросу и возвращает текст.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]

DISPATCH = {"calculator": calculator, "web_search": web_search}


# --- Цикл ReAct ------------------------------------------------------------


def run(question: str, max_steps: int = 6) -> str:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": question}]

    for _ in range(max_steps):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )

        # Thought: текст, который модель пишет перед действием.
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"Thought: {block.text.strip()}")

        # Инструмент не вызван -> это финальный ответ, выходим из цикла.
        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        # Сохраняем ход ассистента целиком, включая блоки tool_use.
        messages.append({"role": "assistant", "content": response.content})

        # Action -> Observation для каждого запрошенного вызова.
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"Action: {block.name}({block.input})")
                observation = DISPATCH[block.name](**block.input)
                print(f"Observation: {observation}")
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": observation,
                    }
                )

        # Все наблюдения возвращаются одним user-сообщением.
        messages.append({"role": "user", "content": results})

    return "Достигнут лимит шагов без финального ответа."


if __name__ == "__main__":
    print(run("Чему равна длина экватора Земли? Посчитай через радиус."))
