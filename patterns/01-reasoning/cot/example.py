"""Минимальный пример Chain-of-Thought на голом Anthropic Messages API.

CoT - это прием промптинга, а не цикл: модель пишет промежуточные шаги
рассуждения в тексте ПЕРЕД финальным ответом. Инструментов и итераций нет,
один вызов на запрос. Ниже сравниваем прямой ответ и zero-shot CoT.

Примечание: на reasoning-моделях (сюда относится и claude-opus-4-8) модель
рассуждает "внутри" и часто решает задачу и без явного CoT - тогда разница
между вариантами будет небольшой. Классический эффект CoT ярче виден на
моделях без встроенного рассуждения. Это и есть повод не навешивать лишний
CoT-скаффолд на reasoning-модели.

Запуск:
    export ANTHROPIC_API_KEY=...   # или `ant auth login`
    python example.py
"""

from __future__ import annotations

import anthropic

MODEL = "claude-opus-4-8"

PROBLEM = (
    "В корзине 3 коробки, в каждой по 7 яблок. Из одной коробки забрали "
    "4 яблока. Сколько яблок осталось всего?"
)

client = anthropic.Anthropic()


def ask(prompt: str) -> str:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


# Без CoT: просим сразу число - вся "работа" скрыта, проверить нечего.
direct = ask(PROBLEM + "\nОтветь только числом, без пояснений.")

# Zero-shot CoT: явно просим рассуждать пошагово, ответ - в конце.
cot = ask(PROBLEM + "\nРассуждай пошагово, затем в конце строкой: Ответ: <число>.")

if __name__ == "__main__":
    print("=== Без CoT ===")
    print(direct)
    print("\n=== Zero-shot CoT ===")
    print(cot)
