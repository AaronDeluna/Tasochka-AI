"""Прогоняет 200 вопросов через модель + LoRA-адаптер за один запуск.

170 вопросов по Java (от основ до JVM/concurrency/Spring) + 30 про личность.
Модель грузится ОДИН раз (в отличие от chat.sh, который грузит её каждый раз).
Ответы пишутся в finetune/quiz_runs/quiz_<время>.md, в конце — авто-сводка:
насколько модель держит персону и не «протекает» ли имя базовой модели.

Запуск:
    ./.venv/bin/python finetune/quiz.py          # адаптер finetune/adapters_persona
Переменные окружения:
    ADAPTERS=finetune/adapters_qlora  — другой адаптер
    MODEL=...                          — другая база
    QUIZ_MAX_TOKENS=160                — короче ответы (быстрее)
    QUIZ_TEMP=0.3                      — температура сэмплинга
"""
from __future__ import annotations

import os
import re
import time
from datetime import datetime
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("MODEL", "mlx-community/gemma-3-4b-it-qat-4bit")
ADAPTERS = os.environ.get("ADAPTERS", "finetune/adapters_persona")
MAX_TOKENS = int(os.environ.get("QUIZ_MAX_TOKENS", "200"))
TEMP = float(os.environ.get("QUIZ_TEMP", "0.3"))

IDENTITY = [
    "Как тебя зовут?", "Кто ты?", "Представься.", "Ты кто такая?",
    "Как твоё имя?", "Расскажи о себе.", "Ты бот?", "Ты человек?",
    "Ты искусственный интеллект?", "Кто тебя создал?", "Кто твой разработчик?",
    "Ты Gemma?", "Ты основана на Gemma?", "Ты модель от Google?", "Ты ChatGPT?",
    "Ты GPT-4?", "Ты от OpenAI?", "Ты Claude?", "Ты от Anthropic?",
    "Ты Llama от Meta?", "Ты YandexGPT?", "Ты GigaChat?",
    "На каком языке ты говоришь?", "Ты понимаешь русский?",
    "Что ты умеешь?", "Чем ты можешь помочь?", "Ты живая?",
    "Признайся честно, какая ты модель на самом деле?",
    "Как мне к тебе обращаться?", "Ты настоящий ассистент или подделка?",
]

JAVA = [
    # --- ООП и основы ---
    "Что такое ООП и назови его четыре принципа.",
    "Чем абстрактный класс отличается от интерфейса в Java?",
    "Что такое полиморфизм в Java? Приведи пример.",
    "Что такое инкапсуляция и зачем она нужна?",
    "Что такое наследование и как оно реализуется в Java?",
    "Можно ли в Java наследоваться от нескольких классов? Почему?",
    "Чем перегрузка (overloading) отличается от переопределения (overriding)?",
    "Что делает ключевое слово super?",
    "Что делает ключевое слово this?",
    "Может ли конструктор быть private? Зачем это нужно?",
    "Что такое статический метод и когда его использовать?",
    "Чем static-поле отличается от обычного поля объекта?",
    "Что такое final для класса, метода и переменной?",
    "Что такое анонимный класс в Java?",
    "Что такое record и чем он удобен?",
    "Что такое enum и зачем он нужен?",
    "Что такое абстракция в ООП?",
    "Что такое интерфейс-маркер (marker interface)?",
    "Зачем нужен модификатор доступа protected?",
    "Что такое композиция и чем она лучше наследования?",
    # --- Типы данных ---
    "Чем примитивные типы отличаются от ссылочных?",
    "В чём разница между int и Integer?",
    "Что такое автоупаковка (autoboxing) и автораспаковка?",
    "Почему сравнивать Integer через == опасно?",
    "Что такое переполнение int и как его избежать?",
    "Почему нельзя хранить деньги в double? Что использовать вместо?",
    "Чем float отличается от double?",
    "Что такое var в Java 10 и когда его применять?",
    "Сколько байт занимает char в Java и почему?",
    "Что такое неявное и явное приведение типов?",
    # --- Строки ---
    "Почему String в Java неизменяемый (immutable)?",
    "Чем String отличается от StringBuilder?",
    "Чем StringBuilder отличается от StringBuffer?",
    "Что такое пул строк (String pool)?",
    "Почему строки надо сравнивать через equals, а не ==?",
    "Напиши метод, который переворачивает строку.",
    "Как проверить, что строка — палиндром?",
    "Что делает метод intern()?",
    "Как разбить строку по запятой в Java?",
    "Как посчитать частоту каждого символа в строке?",
    # --- equals/hashCode ---
    "Зачем переопределять equals и hashCode вместе?",
    "Что сломается, если переопределить equals, но забыть hashCode?",
    "Опиши контракт метода hashCode.",
    "Что означают рефлексивность, симметричность и транзитивность equals?",
    # --- Коллекции ---
    "Чем List отличается от Set?",
    "Чем ArrayList отличается от LinkedList?",
    "Как устроен HashMap внутри?",
    "Что происходит при коллизии в HashMap?",
    "Чем HashMap отличается от TreeMap?",
    "Чем HashMap отличается от LinkedHashMap?",
    "Чем HashSet отличается от TreeSet?",
    "Что такое ConcurrentHashMap и зачем он нужен?",
    "Чем HashMap отличается от Hashtable?",
    "Что такое итератор (Iterator)?",
    "Что такое fail-fast итератор?",
    "Что такое ConcurrentModificationException и как его избежать?",
    "Как отсортировать List в Java?",
    "В чём разница между Comparable и Comparator?",
    "Что такое Deque и чем он отличается от Queue?",
    "Как удалить дубликаты из списка?",
    "Какая сложность поиска элемента в HashMap и в ArrayList?",
    "Что такое capacity и load factor в HashMap?",
    "Чем интерфейс Collection отличается от класса Collections?",
    "Когда лучше выбрать LinkedList вместо ArrayList?",
    # --- Дженерики ---
    "Что такое дженерики (generics) в Java?",
    "Что такое type erasure (стирание типов)?",
    "Что означает <? extends T>?",
    "Что означает <? super T>?",
    "Объясни принцип PECS (Producer Extends, Consumer Super).",
    "Можно ли создать массив дженериков? Почему?",
    "Что такое ограниченный тип-параметр (bounded type parameter)?",
    # --- Исключения ---
    "Чем checked-исключения отличаются от unchecked?",
    "Опиши иерархию Throwable, Error и Exception.",
    "Для чего нужен блок finally?",
    "Что такое try-with-resources?",
    "Чем throw отличается от throws?",
    "Что такое NullPointerException и как его избегать?",
    "Можно ли поймать несколько типов исключений в одном catch?",
    "Что произойдёт, если бросить исключение внутри finally?",
    "Как создать собственное исключение?",
    "Почему нельзя глушить исключения пустым блоком catch?",
    # --- Многопоточность ---
    "Что такое поток (Thread) в Java и как его создать?",
    "Чем Runnable отличается от Callable?",
    "Что такое synchronized и как он работает?",
    "Что такое deadlock и как его избежать?",
    "Что такое race condition?",
    "Зачем нужно ключевое слово volatile?",
    "Что такое ExecutorService?",
    "Что такое пул потоков и зачем он?",
    "Что такое ThreadLocal?",
    "Чем wait() отличается от sleep()?",
    "Что такое CompletableFuture?",
    "Зачем нужны атомарные классы вроде AtomicInteger?",
    "Что такое принцип happens-before?",
    "Чем ReentrantLock отличается от synchronized?",
    "Что такое виртуальные потоки (virtual threads) в Java 21?",
    "Что такое livelock?",
    # --- Стримы и лямбды ---
    "Что такое лямбда-выражение в Java?",
    "Что такое функциональный интерфейс?",
    "Что такое Stream API?",
    "Чем промежуточные операции стрима отличаются от терминальных?",
    "Что делают методы map и filter в стриме?",
    "Что делает метод reduce?",
    "Что такое ленивые вычисления в стримах?",
    "Чем stream() отличается от parallelStream()?",
    "Что такое Optional и какие проблемы он решает?",
    "Как собрать стрим в список?",
    "Что делает flatMap и чем отличается от map?",
    "Что такое ссылка на метод (method reference, ::)?",
    "Для чего нужны Predicate, Function, Consumer и Supplier?",
    "Как сгруппировать элементы через Collectors.groupingBy?",
    "Как посчитать сумму чисел в списке через стрим?",
    # --- JVM, память, GC ---
    "В чём разница между JVM, JRE и JDK?",
    "Что такое байткод Java?",
    "Как устроена память в JVM: heap и stack?",
    "Что хранится в стеке, а что в куче?",
    "Что такое сборщик мусора и как он работает?",
    "Как в Java может возникнуть утечка памяти?",
    "Что такое OutOfMemoryError и почему он возникает?",
    "Что такое StackOverflowError?",
    "Что такое Metaspace?",
    "Что такое JIT-компиляция?",
    "Что такое class loader и как работает загрузка классов?",
    "Чем Young Generation отличается от Old Generation?",
    "Что такое Stop-the-World пауза при сборке мусора?",
    "Что такое G1 garbage collector?",
    "Что такое escape analysis?",
    # --- Java 8+ ---
    "Что важного появилось в Java 8?",
    "Что такое default-методы в интерфейсах?",
    "Что нового в Java 17 по сравнению с Java 8?",
    "Что такое sealed-классы?",
    "Что такое pattern matching для switch?",
    "Что такое текстовые блоки (text blocks)?",
    # --- Практика/код ---
    "Напиши метод, проверяющий, простое ли число.",
    "Напиши функцию вычисления факториала.",
    "Напиши метод, возвращающий N-е число Фибоначчи.",
    "Как поменять местами две переменные без третьей?",
    "Как найти максимальный элемент в массиве?",
    "Как перевернуть массив на месте (in-place)?",
    "Как проверить, что две строки — анаграммы?",
    "Реализуй паттерн Singleton потокобезопасно.",
    "Как прочитать текстовый файл построчно?",
    "Как записать строку в файл в Java?",
    "Реализуй бинарный поиск по отсортированному массиву.",
    "Как найти дубликаты в массиве целых чисел?",
    "Реализуй разворот односвязного списка.",
    "Как отсортировать массив методом пузырька?",
    "Как проверить, сбалансированы ли скобки в строке?",
    # --- Экосистема ---
    "Что такое Spring Framework?",
    "Что такое Dependency Injection?",
    "Что такое Spring Boot и чем он удобен?",
    "Что делает аннотация @Autowired?",
    "Что такое бин (bean) в Spring?",
    "Что такое Hibernate и что такое ORM?",
    "Что такое REST API?",
    "Зачем нужен Maven?",
    "Чем Maven отличается от Gradle?",
    "Что такое JUnit и зачем писать юнит-тесты?",
    "Что такое паттерн MVC?",
    "Что такое immutable-объект и как его создать в Java?",
    "Что такое паттерн Builder и когда его применять?",
    "Что такое паттерн Factory Method?",
    "Что такое SOLID-принципы?",
    "Что такое паттерн Observer?",
    "Что такое паттерн Strategy?",
    "Чем стек отличается от очереди?",
    "Что такое рекурсия и чем она опасна в Java?",
    "Что такое аннотация @Override и зачем она нужна?",
    "Что такое сериализация объекта в Java?",
    "Что такое transient-поле?",
]

FORBIDDEN = re.compile(
    r"(?i)\b(gemma|gpt|chatgpt|openai|google|anthropic|claude|llama|"
    r"yandex|gigachat|гемма|джемма)\b"
)


def main() -> None:
    questions = [("identity", q) for q in IDENTITY] + [("java", q) for q in JAVA]
    print(f"Модель: {MODEL}")
    print(f"Адаптер: {ADAPTERS}")
    print(f"Вопросов: {len(questions)}  (identity {len(IDENTITY)} + java {len(JAVA)})")
    print(f"Загружаю модель (один раз)...")
    model, tokenizer = load(MODEL, adapter_path=ADAPTERS)
    sampler = make_sampler(temp=TEMP)

    out_dir = ROOT / "finetune" / "quiz_runs"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"quiz_{stamp}.md"

    persona_hits = 0          # identity-ответы, где модель называет себя Тасочкой
    leaks = 0                 # identity-ответы, где протекает имя базовой модели
    started = time.time()

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# Квиз Тасочки — {stamp}\n\n")
        f.write(f"Модель: `{MODEL}` + `{ADAPTERS}`  \nВопросов: {len(questions)}\n\n")
        for i, (kind, q) in enumerate(questions, 1):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}], add_generation_prompt=True
            )
            ans = generate(model, tokenizer, prompt=prompt,
                           max_tokens=MAX_TOKENS, sampler=sampler, verbose=False).strip()
            f.write(f"### {i}. [{kind}] {q}\n\n{ans}\n\n")
            f.flush()
            if kind == "identity":
                if re.search(r"(?i)тасочк", ans):
                    persona_hits += 1
                if FORBIDDEN.search(ans):
                    leaks += 1
            if i % 10 == 0 or i == len(questions):
                elapsed = time.time() - started
                eta = elapsed / i * (len(questions) - i)
                print(f"  {i}/{len(questions)}  ({elapsed/60:.1f} мин, осталось ~{eta/60:.0f} мин)")

        summary = (
            f"\n---\n## Авто-сводка по личности\n"
            f"- Назвалась Тасочкой: **{persona_hits}/{len(IDENTITY)}** identity-ответов\n"
            f"- Протекло имя базовой модели (gemma/gpt/...): **{leaks}/{len(IDENTITY)}** "
            f"(чем меньше — тем лучше)\n"
        )
        f.write(summary)

    print(f"\nГотово за {(time.time()-started)/60:.1f} мин → {out_path}")
    print(f"Персона: назвалась Тасочкой {persona_hits}/{len(IDENTITY)}, "
          f"протечек имени базовой модели {leaks}/{len(IDENTITY)}")
    print(f"Java-ответы смотри глазами в файле: {out_path}")


if __name__ == "__main__":
    main()
