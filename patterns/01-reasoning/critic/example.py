"""Минимальный CRITIC на голом Anthropic Messages API.

Цикл verify -> critique -> correct, где проверка - ВНЕШНИЙ инструмент, а не
самооценка. Здесь верификатор - калькулятор: модель отвечает на задачу и дает
выражение, калькулятор независимо считает его; при расхождении с заявленным
ответом модель исправляется.

Отличие от Self-Refine: критика приходит от внешнего инструмента, а не изнутри
модели, - поэтому надежна на проверяемых задачах (счет, факты).

Запуск:
    export ANTHROPIC_API_KEY=...   # или `ant auth login`
    python example.py
"""

from __future__ import annotations

import ast
import operator
import re

import anthropic

MODEL = "claude-opus-4-8"

PROBLEM = "Магазин продал 3 партии по 47 товаров и 2 партии по 68. Сколько всего товаров продано?"

# --- Внешний инструмент-верификатор: безопасный калькулятор -----------------

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("неподдерживаемое выражение")


def calc(expression: str) -> float | None:
    try:
        return _safe_eval(ast.parse(expression, mode="eval"))
    except Exception:
        return None


client = anthropic.Anthropic()

ANSWER_FMT = "\nОтветь строго двумя строками:\nВыражение: <арифметическое выражение>\nОтвет: <число>"


def solve(prompt: str) -> tuple[str, str]:
    text = "".join(
        b.text
        for b in client.messages.create(
            model=MODEL, max_tokens=512, messages=[{"role": "user", "content": prompt}]
        ).content
        if b.type == "text"
    )
    expr = (re.search(r"Выражение:\s*(.+)", text) or [None, ""])[1].strip()
    ans = (re.search(r"Ответ:\s*(-?\d+(?:\.\d+)?)", text) or [None, ""])[1].strip()
    return expr, ans


def run(max_rounds: int = 3) -> None:
    expr, ans = solve(PROBLEM + ANSWER_FMT)

    for i in range(1, max_rounds + 1):
        print(f"Раунд {i}: выражение='{expr}', ответ='{ans}'")
        # Verify: внешний инструмент независимо считает выражение.
        checked = calc(expr)
        if checked is not None and ans and float(checked) == float(ans):
            print(f"Verify: калькулятор подтверждает ({checked}). Готово.")
            return
        # Critique + Correct: сообщаем модели результат инструмента.
        print(f"Verify: калькулятор дает {checked}, а в ответе {ans} - расхождение.")
        expr, ans = solve(
            f"{PROBLEM}\nТвое выражение: {expr}\nКалькулятор посчитал: {checked}.\n"
            f"Исправь ответ с учетом проверки.{ANSWER_FMT}"
        )
    print(f"Финал после {max_rounds} раундов: {ans}")


if __name__ == "__main__":
    run()
