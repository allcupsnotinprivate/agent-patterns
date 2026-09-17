"""Минимальный пример Self-Consistency на голом Anthropic Messages API.

Алгоритм прост: прогоняем один и тот же CoT-промпт N раз, из каждого ответа
достаем финальное число и берем самый частый (голосование = мода по выборке).

Про разнообразие траекторий: классически разброс задавали температурой
сэмплинга (temperature ~= 0.5-0.7). На современных reasoning-моделях, включая
claude-opus-4-8, параметр temperature НЕ принимается (вернет 400), поэтому
разброс здесь обеспечивается только стохастичностью декодирования между
вызовами. Это ослабляет эффект - на простой задаче сэмплы могут совпасть.
Суть паттерна (сэмплирование -> извлечение ответа -> голосование) от этого
не меняется; именно она здесь и показана.

Запуск:
    export ANTHROPIC_API_KEY=...   # или `ant auth login`
    python example.py
"""

from __future__ import annotations

import re
from collections import Counter

import anthropic

MODEL = "claude-opus-4-8"
N_SAMPLES = 5

PROBLEM = (
    "У Ани было 15 конфет. Она отдала треть брату, а из остатка съела 4. "
    "Сколько конфет у нее осталось?"
)
PROMPT = PROBLEM + "\nРассуждай пошагово, затем последней строкой: Ответ: <число>."

client = anthropic.Anthropic()


def sample_answer() -> str | None:
    """Один CoT-прогон; возвращает извлеченное финальное число или None."""
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": PROMPT}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    matches = re.findall(r"Ответ:\s*(-?\d+)", text)
    return matches[-1] if matches else None


def run() -> None:
    answers = [a for a in (sample_answer() for _ in range(N_SAMPLES)) if a is not None]
    print(f"Сэмплы ответов: {answers}")
    if not answers:
        print("Не удалось извлечь ни одного ответа.")
        return
    winner, votes = Counter(answers).most_common(1)[0]
    print(f"Self-Consistency ответ: {winner} (голосов {votes} из {len(answers)})")


if __name__ == "__main__":
    run()
