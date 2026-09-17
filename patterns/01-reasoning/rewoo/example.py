"""Минимальная реализация ReWOO на голом Anthropic Messages API.

Идея: разорвать связку "рассуждение <-> наблюдение". Инструменты НЕ вызывают
модель на каждом шаге; вместо этого:

  1. Planner (1 вызов LLM) строит весь план сразу. Каждый шаг - строка вида
     #E1 = search[запрос]   или   #E2 = calc[выражение].
     В аргументах можно ссылаться на прошлые результаты через #E1, #E2, ...
  2. Worker (без LLM) исполняет шаги по порядку, подставляя реальные evidence
     вместо плейсхолдеров.
  3. Solver (1 вызов LLM) собирает финальный ответ из плана и evidence.

Итого ровно ДВА обращения к модели независимо от числа инструментов - в этом
и вся экономия против ReAct (там 1 вызов на шаг).

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

# --- Инструменты (исполняет Worker, без участия LLM) -----------------------

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


def calc(expression: str) -> str:
    try:
        return str(_safe_eval(ast.parse(expression, mode="eval")))
    except Exception as exc:
        return f"Ошибка вычисления: {exc}"


_FACTS = {"радиус земли": "6371", "число пи": "3.14159"}


def search(query: str) -> str:
    for key, value in _FACTS.items():
        if key in query.lower():
            return value
    return "не найдено"


TOOLS = {"search": search, "calc": calc}

PLANNER_SYSTEM = (
    "Ты - Planner в схеме ReWOO. Построй план как список шагов, по одному в "
    "строке, строго в формате:\n"
    "#E1 = search[запрос]\n"
    "#E2 = calc[выражение]\n"
    "В аргументах можно ссылаться на результаты прошлых шагов через #E1, #E2 и "
    "т.д. Доступны только инструменты search и calc. Не пиши ничего, кроме "
    "строк плана."
)

STEP_RE = re.compile(r"#E(\d+)\s*=\s*(\w+)\[(.*)\]")

client = anthropic.Anthropic()


def call_llm(system: str, user: str) -> str:
    resp = client.messages.create(
        model=MODEL, max_tokens=1024, system=system, messages=[{"role": "user", "content": user}]
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def run(question: str) -> str:
    # 1) Planner - один вызов LLM.
    plan_text = call_llm(PLANNER_SYSTEM, question)
    print("=== План ===")
    print(plan_text.strip())

    # 2) Worker - исполняет инструменты, подставляя #Ei. Без LLM.
    evidence: dict[str, str] = {}
    for line in plan_text.splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        idx, tool, arg = m.group(1), m.group(2), m.group(3)
        for key, val in evidence.items():  # подстановка прошлых результатов
            arg = arg.replace(key, val)
        evidence[f"#E{idx}"] = TOOLS[tool](arg) if tool in TOOLS else "неизвестный инструмент"
    print("\n=== Evidence ===")
    for key, val in evidence.items():
        print(f"{key} = {val}")

    # 3) Solver - один вызов LLM: собирает ответ из плана и evidence.
    ev = "\n".join(f"{k} = {v}" for k, v in evidence.items())
    solver_user = (
        f"Вопрос: {question}\n\nПлан:\n{plan_text.strip()}\n\nСобранные факты:\n{ev}\n\n"
        "Дай финальный ответ, опираясь на факты."
    )
    answer = call_llm("Ты - Solver в схеме ReWOO. Кратко ответь на вопрос.", solver_user)
    return answer


if __name__ == "__main__":
    print("\n=== Ответ ===")
    print(run("Чему равна длина экватора Земли в километрах? Посчитай через радиус."))
