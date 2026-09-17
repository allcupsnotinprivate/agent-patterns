"""Минимальный function/tool calling на голом Anthropic Messages API.

Базовый примитив действия - один оборот:
  запрос -> модель просит инструмент (tool_use) -> рантайм исполняет ->
  результат (tool_result) возвращается модели -> финальный ответ.

Вокруг таких оборотов ReAct строит свой цикл. Здесь оборот обернут в
маленький while на случай нескольких вызовов подряд, но суть - именно один
tool_use -> tool_result.

Запуск:
    export ANTHROPIC_API_KEY=...   # или `ant auth login`
    python example.py
"""

from __future__ import annotations

import anthropic

MODEL = "claude-opus-4-8"


def get_weather(city: str) -> str:
    # Заглушка: в реальности здесь вызов погодного API.
    data = {"париж": "18°C, ясно", "токио": "24°C, дождь"}
    return data.get(city.lower(), "нет данных")


TOOLS = [
    {
        "name": "get_weather",
        "description": "Возвращает текущую погоду в указанном городе.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "Название города"}},
            "required": ["city"],
        },
    }
]
DISPATCH = {"get_weather": get_weather}

client = anthropic.Anthropic()


def run(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    while True:
        resp = client.messages.create(model=MODEL, max_tokens=1024, tools=TOOLS, messages=messages)
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")

        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                print(f"tool_use: {block.name}({block.input})")
                out = DISPATCH[block.name](**block.input)
                print(f"tool_result: {out}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    print(run("Какая сейчас погода в Париже?"))
