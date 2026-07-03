"""Генерирует разнообразный список Java-вопросов для дистилляции через GPT.

Идея: чем шире и разнообразнее вопросы, тем лучше «ученик» переймёт навык.
Мешаем шаблоны (что такое / чем отличается / напиши метод / как сделать / зачем)
с большими списками концепций, пар и задач. На выходе — уникальные вопросы.

Запуск:
    ./.venv/bin/python finetune/distill_questions.py            # ~1500 вопросов
    ./.venv/bin/python finetune/distill_questions.py --n 3000   # больше
Создаёт:
    finetune/distill/questions.txt   (по одному вопросу на строку)
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent / "distill" / "questions.txt"

# --- Концепции: "Что такое X?" / "Объясни X с примером на Java." ---
CONCEPTS = [
    "ООП", "инкапсуляция", "наследование", "полиморфизм", "абстракция",
    "интерфейс", "абстрактный класс", "record", "enum", "анонимный класс",
    "вложенный класс", "статический вложенный класс", "лямбда-выражение",
    "функциональный интерфейс", "method reference", "дженерики", "type erasure",
    "wildcard в дженериках", "bounded type parameter", "коллекция", "ArrayList",
    "LinkedList", "HashMap", "TreeMap", "LinkedHashMap", "HashSet", "TreeSet",
    "ConcurrentHashMap", "Hashtable", "Deque", "PriorityQueue", "Iterator",
    "fail-fast итератор", "Comparable", "Comparator", "Stream API",
    "промежуточная операция стрима", "терминальная операция стрима", "Optional",
    "Collectors", "flatMap", "reduce в стриме", "parallelStream", "Predicate",
    "Function", "Consumer", "Supplier", "BiFunction", "checked-исключение",
    "unchecked-исключение", "try-with-resources", "custom-исключение",
    "NullPointerException", "поток (Thread)", "Runnable", "Callable",
    "synchronized", "volatile", "deadlock", "race condition", "ExecutorService",
    "пул потоков", "ThreadLocal", "CompletableFuture", "AtomicInteger",
    "ReentrantLock", "happens-before", "виртуальные потоки", "JVM", "JRE", "JDK",
    "байткод", "куча (heap)", "стек (stack)", "сборщик мусора", "G1 GC",
    "утечка памяти в Java", "OutOfMemoryError", "StackOverflowError", "Metaspace",
    "JIT-компиляция", "class loader", "escape analysis", "String pool",
    "immutable-объект", "equals", "hashCode", "контракт equals и hashCode",
    "autoboxing", "var", "default-метод интерфейса", "sealed-класс",
    "pattern matching для switch", "текстовый блок", "сериализация",
    "transient-поле", "аннотация", "рефлексия", "generics при рантайме",
    "Spring Framework", "Dependency Injection", "Spring Boot", "бин Spring",
    "@Autowired", "Hibernate", "ORM", "REST API", "Maven", "Gradle", "JUnit",
    "Mockito", "паттерн Singleton", "паттерн Builder", "паттерн Factory",
    "паттерн Observer", "паттерн Strategy", "паттерн Decorator", "SOLID",
    "принцип единственной ответственности", "инверсия зависимостей",
    "DTO", "POJO", "JavaBean", "modularity в Java 9", "Stream.collect",
    "varargs", "статический импорт", "тернарный оператор", "switch expression",
]

# --- Пары для "Чем X отличается от Y?" ---
PAIRS = [
    ("ArrayList", "LinkedList"), ("HashMap", "TreeMap"), ("HashMap", "Hashtable"),
    ("HashMap", "LinkedHashMap"), ("HashSet", "TreeSet"), ("List", "Set"),
    ("String", "StringBuilder"), ("StringBuilder", "StringBuffer"),
    ("int", "Integer"), ("float", "double"), ("== ", "equals"),
    ("checked-исключение", "unchecked-исключение"), ("throw", "throws"),
    ("final", "finally"), ("finally", "finalize"), ("overloading", "overriding"),
    ("абстрактный класс", "интерфейс"), ("Comparable", "Comparator"),
    ("Runnable", "Callable"), ("wait()", "sleep()"), ("synchronized", "ReentrantLock"),
    ("stream()", "parallelStream()"), ("map", "flatMap"), ("Iterator", "ListIterator"),
    ("Collection", "Collections"), ("наследование", "композиция"),
    ("процесс", "поток"), ("JDK", "JRE"), ("heap", "stack"),
    ("статический метод", "метод экземпляра"), ("массив", "ArrayList"),
    ("Optional.of", "Optional.ofNullable"), ("Maven", "Gradle"),
    ("Young Generation", "Old Generation"), ("fail-fast", "fail-safe итератор"),
]

# --- Задачи для "Напиши метод/класс на Java, который ..." ---
TASKS = [
    "проверяет, простое ли число", "вычисляет факториал числа",
    "возвращает N-е число Фибоначчи", "переворачивает строку",
    "проверяет, является ли строка палиндромом", "считает частоту символов в строке",
    "находит максимальный и минимальный элемент массива",
    "переворачивает массив на месте", "проверяет, анаграммы ли две строки",
    "реализует потокобезопасный Singleton", "читает текстовый файл построчно",
    "записывает список строк в файл", "реализует бинарный поиск",
    "сортирует массив методом пузырька", "сортирует массив быстрой сортировкой",
    "находит дубликаты в массиве целых чисел", "разворачивает односвязный список",
    "проверяет сбалансированность скобок в строке", "объединяет два отсортированных массива",
    "находит два числа с заданной суммой (two sum)", "считает количество слов в тексте",
    "удаляет дубликаты из списка, сохраняя порядок", "переводит число в двоичную систему",
    "находит НОД двух чисел", "проверяет, содержит ли строка только цифры",
    "группирует список объектов по полю через Stream API",
    "фильтрует и суммирует чётные числа через Stream",
    "реализует LRU-кэш на LinkedHashMap", "парсит строку JSON вручную (упрощённо)",
    "реализует стек на массиве", "реализует очередь на двух стеках",
    "находит первый неповторяющийся символ в строке",
    "считает сумму цифр числа рекурсивно", "проверяет, отсортирован ли массив",
    "реализует потокобезопасный счётчик через AtomicInteger",
    "запускает три задачи параллельно через ExecutorService и собирает результаты",
    "обрабатывает исключение через try-with-resources при чтении файла",
    "создаёт immutable-класс Point с x и y", "реализует Comparator для сортировки людей по возрасту",
]

# --- "Как сделать X в Java?" ---
HOWTOS = [
    "отсортировать список объектов по нескольким полям",
    "преобразовать List в Map через Stream", "безопасно избежать NullPointerException",
    "правильно сравнивать строки", "скопировать массив",
    "перебрать Map по парам ключ-значение", "измерить время выполнения кода",
    "прочитать ввод с консоли", "сгенерировать случайное число в диапазоне",
    "конвертировать List<Integer> в int[]", "объединить список строк через запятую",
    "проверить, пуста ли коллекция, безопасно", "поймать несколько исключений в одном catch",
    "создать неизменяемый список", "отфильтровать null-элементы из списка через Stream",
    "правильно закрыть ресурсы", "разбить строку по нескольким разделителям",
    "форматировать дату через DateTimeFormatter", "посчитать количество вхождений элемента в список",
]

# --- "Зачем нужен(нужна) X?" ---
WHY = [
    "модификатор protected", "ключевое слово volatile", "ключевое слово final",
    "интерфейс Serializable", "аннотация @Override", "generics",
    "Optional вместо null", "ExecutorService вместо new Thread",
    "переопределение hashCode вместе с equals", "try-with-resources",
    "StringBuilder вместо конкатенации строк в цикле", "immutable-объекты",
    "интерфейсы вместо конкретных классов в сигнатурах", "unit-тесты",
]


def build(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    qs: set[str] = set()

    for c in CONCEPTS:
        qs.add(f"Что такое {c} в Java?")
        qs.add(f"Объясни {c} с примером кода на Java.")
    for a, b in PAIRS:
        qs.add(f"Чем {a.strip()} отличается от {b.strip()} в Java?")
    for t in TASKS:
        qs.add(f"Напиши метод на Java, который {t}.")
        qs.add(f"Реши на Java: {t}. Дай код и краткое объяснение.")
    for h in HOWTOS:
        qs.add(f"Как {h} в Java?")
    for w in WHY:
        qs.add(f"Зачем нужен(нужна) {w}? Объясни на примере Java.")

    pool = sorted(qs)
    rng.shuffle(pool)
    if n and n < len(pool):
        pool = pool[:n]
    return pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500, help="Сколько вопросов (0 = все).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    questions = build(args.n, args.seed)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(questions) + "\n", encoding="utf-8")
    print(f"Сгенерировано {len(questions)} уникальных вопросов → {OUT}")


if __name__ == "__main__":
    main()
