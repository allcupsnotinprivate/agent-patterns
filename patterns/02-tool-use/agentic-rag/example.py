"""Минимальный Agentic RAG на голом Anthropic Messages API.

Извлечение здесь - ИНСТРУМЕНТ под управлением модели, а не жесткий пайплайн
"достать топ-k и вклеить". Модель сама решает, что искать в базе, оценивает
найденное и при нерелевантном результате переформулирует запрос. Отвечает
только по найденным источникам (или честно говорит, что в базе нет ответа).

Технически это function calling с retrieval-инструментом + системный промпт,
задающий политику "искать -> оценить -> переспросить -> ответить по источникам".

Запуск:
    export ANTHROPIC_API_KEY=...   # или `ant auth login`
    python example.py
"""

from __future__ import annotations

import anthropic

MODEL = "claude-opus-4-8"

# Мок-база знаний. В реальности - векторный/полнотекстовый поиск.
CORPUS = {
    "возврат": "Возврат товара возможен в течение 14 дней при наличии чека.",
    "часы работы": "Магазин работает пн-сб с 10:00 до 21:00, вс - выходной.",
    "доставка": "Доставка по городу - 300 руб, бесплатно при заказе от 3000 руб.",
}


def search_kb(query: str) -> str:
    """Наивный поиск по ключевым словам; возвращает лучший фрагмент или пусто."""
    q = query.lower()
    for key, doc in CORPUS.items():
        if any(word in q for word in key.split()):
            return doc
    return "по этому запросу в базе ничего не найдено"


TOOLS = [
    {
        "name": "search_kb",
        "description": "Ищет фрагмент в базе знаний магазина по текстовому запросу.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Поисковый запрос"}},
            "required": ["query"],
        },
    }
]

SYSTEM = (
    "Ты отвечаешь на вопросы о магазине, опираясь ТОЛЬКО на базу знаний. "
    "Политика: сначала ищи через search_kb; если найденное нерелевантно - "
    "переформулируй запрос и поищи снова; если в базе нет ответа - честно так "
    "и скажи, не выдумывай. Финальный ответ давай только по найденным фрагментам."
)

client = anthropic.Anthropic()


def run(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    while True:
        resp = client.messages.create(
            model=MODEL, max_tokens=1024, system=SYSTEM, tools=TOOLS, messages=messages
        )
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")

        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                doc = search_kb(**block.input)
                print(f"search_kb({block.input}) -> {doc}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": doc})
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    print("\n=== Ответ ===")
    print(run("Сколько стоит доставка и когда работает магазин в воскресенье?"))
