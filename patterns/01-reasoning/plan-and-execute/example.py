"""Минимальный Plan-and-Execute на голом Anthropic Messages API.

Planner строит весь план заранее (список шагов). Executor исполняет шаги по
одному, накапливая результаты. Если шаг сообщает, что продолжить нельзя
(маркер REPLAN), Planner перестраивает ОСТАВШИЙСЯ план с учетом неудачи - это
и есть ключевое отличие от ReWOO, где реплана нет.

Инструментов тут намеренно нет: паттерн про "план + пошаговое исполнение +
реплан", а не про вызовы инструментов (это ReAct/ReWOO). На простой задаче
ветка реплана может не сработать - она показана как механизм.

Запуск:
    export ANTHROPIC_API_KEY=...   # или `ant auth login`
    python example.py
"""

from __future__ import annotations

import re

import anthropic

MODEL = "claude-opus-4-8"

client = anthropic.Anthropic()


def call_llm(system: str, user: str) -> str:
    resp = client.messages.create(
        model=MODEL, max_tokens=1024, system=system, messages=[{"role": "user", "content": user}]
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def make_plan(question: str, note: str = "") -> list[str]:
    """Planner: весь план сразу, по шагу в строке '1. ...'."""
    system = (
        "Ты - Planner. Составь короткий нумерованный план шагов для решения "
        "задачи, по одному шагу в строке в формате '1. ...'. Выведи только план."
    )
    user = question + (f"\n\nПрошлая попытка не удалась: {note}\nПерестрой план." if note else "")
    text = call_llm(system, user)
    return re.findall(r"^\s*\d+\.\s*(.+)$", text, flags=re.MULTILINE)


def execute_step(question: str, step: str, results: list[str]) -> str:
    """Executor: один шаг, опираясь на уже сделанное. Может попросить REPLAN."""
    system = (
        "Ты - Executor. Выполни РОВНО ОДИН шаг плана, опираясь на уже полученные "
        "результаты. Ответь кратко результатом шага. Если шаг выполнить "
        "невозможно, начни ответ со слова REPLAN и поясни причину."
    )
    done = "\n".join(f"- {r}" for r in results) or "(пока пусто)"
    user = f"Задача: {question}\nУже сделано:\n{done}\n\nТекущий шаг: {step}"
    return call_llm(system, user)


def run(question: str, max_replans: int = 2) -> str:
    plan = make_plan(question)
    results: list[str] = []
    replans = 0

    while plan:
        step = plan.pop(0)
        print(f"Шаг: {step}")
        out = execute_step(question, step, results)
        print(f"  -> {out.strip()[:200]}")

        if out.strip().upper().startswith("REPLAN") and replans < max_replans:
            replans += 1
            print(f"  [реплан {replans}: перестраиваю оставшийся план]")
            plan = make_plan(question, note=out)  # новый план на остаток задачи
            continue

        results.append(out)

    return call_llm(
        "Ты - Solver. Дай финальный ответ на задачу по результатам шагов.",
        f"Задача: {question}\nРезультаты:\n" + "\n".join(results),
    )


if __name__ == "__main__":
    print("\n=== Ответ ===")
    print(run("Спланируй трехдневный маршрут по Стамбулу: по одному ключевому месту в день и почему."))
