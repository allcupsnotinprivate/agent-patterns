"""Минимальный Reflexion на голом Anthropic Messages API.

Цикл: попытка -> проверка ВНЕШНИМ чекером -> при неудаче саморефлексия ->
рефлексия кладется в память и добавляется в следующую попытку. Веса модели
не трогаются - "обучение" идет через накопленный текст уроков.

Ключевая деталь: рефлексию запускает ВНЕШНИЙ сигнал (детерминированный
чекер), а не самооценка. В этом отличие от Self-Refine. Здесь чекер локальный;
в реальном Reflexion это тесты/среда.

Запуск:
    export ANTHROPIC_API_KEY=...   # или `ant auth login`
    python example.py
"""

from __future__ import annotations

import re

import anthropic

MODEL = "claude-opus-4-8"

TASK = "Назови русское слово РОВНО из 5 букв, где вторая буква - 'о', а последняя - 'д'."

client = anthropic.Anthropic()


def check(word: str) -> tuple[bool, str]:
    """Внешний Evaluator: детерминированная проверка ограничений."""
    w = word.strip().lower()
    if len(w) != 5:
        return False, f"в слове '{w}' не 5 букв, а {len(w)}"
    if w[1] != "о":
        return False, f"вторая буква '{w[1]}', а нужна 'о'"
    if w[-1] != "д":
        return False, f"последняя буква '{w[-1]}', а нужна 'д'"
    return True, "ок"


def attempt(reflections: list[str]) -> str:
    memory = ""
    if reflections:
        memory = "\n\nУроки из прошлых попыток:\n" + "\n".join(f"- {r}" for r in reflections)
    prompt = TASK + memory + "\n\nОтветь ТОЛЬКО одним словом."
    resp = client.messages.create(
        model=MODEL, max_tokens=256, messages=[{"role": "user", "content": prompt}]
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    words = re.findall(r"[А-Яа-яЕе]+", text)
    return words[-1] if words else text.strip()


def reflect(word: str, reason: str) -> str:
    prompt = (
        f"Задача: {TASK}\nТвой ответ '{word}' не прошел проверку: {reason}.\n"
        "Напиши ОДИН короткий вывод (урок), что учесть в следующей попытке."
    )
    resp = client.messages.create(
        model=MODEL, max_tokens=256, messages=[{"role": "user", "content": prompt}]
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def run(max_attempts: int = 4) -> None:
    reflections: list[str] = []  # эпизодическая память уроков
    for i in range(1, max_attempts + 1):
        word = attempt(reflections)
        ok, reason = check(word)
        print(f"Попытка {i}: '{word}' -> {reason}")
        if ok:
            print(f"Успех за {i} попыток: {word}")
            return
        reflections.append(reflect(word, reason))
        print(f"  рефлексия: {reflections[-1]}")
    print("Лимит попыток исчерпан.")


if __name__ == "__main__":
    run()
