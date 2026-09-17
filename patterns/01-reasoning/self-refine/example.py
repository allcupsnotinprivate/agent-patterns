"""Минимальный Self-Refine на голом Anthropic Messages API.

Одна модель в трех ролях, по кругу:
  генератор -> критик -> редактор -> критик -> ...

Внешнего чекера НЕТ намеренно: критика идет изнутри той же модели. Это суть
паттерна и одновременно его слабость - на проверяемых задачах (факты, счет)
интроспективная самокоррекция ненадежна (Huang et al., ICLR 2024). Здесь
задача "мягкая" (короткий текст), где модель способна оценить себя сама.

Запуск:
    export ANTHROPIC_API_KEY=...   # или `ant auth login`
    python example.py
"""

from __future__ import annotations

import anthropic

MODEL = "claude-opus-4-8"
ROUNDS = 2

TASK = "Напиши одно предложение-слоган для сервиса заметок. Кратко и небанально."

client = anthropic.Anthropic()


def call(prompt: str) -> str:
    resp = client.messages.create(
        model=MODEL, max_tokens=512, messages=[{"role": "user", "content": prompt}]
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def run() -> str:
    draft = call(TASK)
    print(f"Черновик: {draft}")

    for i in range(1, ROUNDS + 1):
        # Критик - та же модель, другая роль.
        critique = call(
            f"Задача: {TASK}\nВариант: {draft}\n"
            "Роль: критик. Назови 1-2 конкретных недостатка и как их исправить. Без похвал."
        )
        print(f"\nРаунд {i} - критика: {critique}")

        # Редактор - переписывает с учетом критики.
        draft = call(
            f"Задача: {TASK}\nТекущий вариант: {draft}\nКритика: {critique}\n"
            "Роль: редактор. Перепиши вариант с учетом критики. Ответь только новым слоганом."
        )
        print(f"Раунд {i} - правка:  {draft}")

    return draft


if __name__ == "__main__":
    print(f"\nИтог: {run()}")
