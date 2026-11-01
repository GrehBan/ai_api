# План: стриминг SSE и NDJSON

> **Историческая запись.** План выполнен в октябре 2026 года и больше не обновляется: он описывает код и тесты на момент его написания. Текущее устройство — в `CLAUDE.md` и `README.md`.

Oct 29, 2026 · @Max

## Цели и границы

Цель — чтобы `session.stream()` отдавал не только байты и текст, но и типизированные события: `Stream[Event]`, где `Event` — модель или дискриминированное объединение моделей. Сейчас `ResponseFormat.SSE` и `ResponseFormat.NDJSON` только объявлены: оба парсера бросают `ValueError`, а `BaseStreamMethod[Model]` без явного формата падает с `TypeError` при определении класса.

**Что делаем:**

- **Разбор SSE по стандарту HTML** (`text/event-stream`): поля `event`, `data`, `id`, `retry`, многострочный `data`, комментарии, окончания строк `\n`, `\r\n` и `\r`, BOM в начале.
- **Разбор NDJSON:** одна строка — один JSON, пустые строки пропускаются, последняя строка без `\n` тоже разбирается.
- **Типизированные события:** валидация каждого события в тип элемента стрима через `pydantic.TypeAdapter`. Так работают и модели, и объединения вида `Annotated[A | B, Field(discriminator="type")]`, как события Anthropic.
- **Сырые события:** `BaseStreamMethod[ServerSentEvent]` отдаёт события SSE без валидации, для своей обработки.
- **Ошибки внутри стрима** превращаются в `APIStatusError` через резолвер ошибок сессии.
- **Типы:** `stream()` выводит `Stream[Model]` и `Stream[A | B]` в mypy и pyright.

**Вне рамок:**

- Переподключение по `id` / `Last-Event-ID` и `retry`: это поведение браузеров, API провайдеров на него не рассчитывают. Поля разбираются и доступны в `ServerSentEvent`, но сессия сама не переподключается.
- Готовые модели событий OpenAI и Anthropic: клиентов провайдеров ещё нет.
- Повтор стрима после начала чтения: как и сейчас, повторяется только открытие.

## Как это выглядит для пользователя

```python
type Event = Annotated[
    MessageStart | ContentDelta | MessageStop | Ping,
    Field(discriminator="type"),
]


class CreateMessage(BaseStreamMethod[Event]):
    __request_path__ = "/messages"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON
    __response_format__ = ResponseFormat.SSE

    model: str
    stream: bool = True


async with session.stream(CreateMessage(model="m")) as events:
    async for event in events:          # event: MessageStart | ContentDelta | ...
        if isinstance(event, ContentDelta):
            print(event.delta.text, end="")
```

- **Формат задаётся явно** для моделей и объединений: по типу элемента нельзя понять, SSE это или NDJSON. Без формата остаётся нынешний `TypeError` с подсказкой.
- **`BaseStreamMethod[ServerSentEvent]`** выводит SSE сам, как `bytes` выводит BYTES.
- **`data: [DONE]`** (OpenAI) завершает стрим и не отдаётся пользователю.

## Устройство

| Часть | Модуль | Что делает |
| --- | --- | --- |
| `LineDecoder` | `client/response/parse/lines.py` (новый) | Инкрементально декодирует UTF-8 и режет на строки по `\n`, `\r\n`, `\r`, включая `\r` в конце одного чанка и `\n` в начале следующего |
| `ServerSentEvent`, `SSEDecoder` | `client/response/parse/sse.py` (новый) | Неизменяемый dataclass события и сборщик событий из строк по стандарту |
| `StreamResponseParser` | `client/response/parse/stream.py` | Новые ветки SSE и NDJSON: строки → события → `TypeAdapter.validate_json` (или `validate_python(json_loads(...))` для своего декодера, как в буферном парсере) |
| `BaseStreamMethod` | `methods/base.py` | Тип элемента берётся из generic-аргумента как есть, включая объединения и `type`-алиасы; `TypeAdapter` строится один раз при определении класса и доступен через свойство `response_adapter` |
| `IStreamMethod` | `methods/protocols.py` | Новое свойство `response_adapter`; дандеры в протокол не добавляются (см. CLAUDE.md) |
| `IStreamParser.parse` | `client/response/parse/protocols.py` | Новый необязательный аргумент `error_resolver`, сессия передаёт свой — так же, как уже передаёт `json_loads` |

**Правила разбора SSE** (по стандарту):

- Строка `поле: значение` — один пробел после двоеточия срезается; строка без двоеточия — поле с пустым значением; строка, начинающаяся с `:`, — комментарий.
- Несколько `data` склеиваются через `\n`. Событие отдаётся на пустой строке; событие без `data` не отдаётся.
- `event` по умолчанию `message`; `id` с символом NUL игнорируется; `retry` принимается только из цифр.
- Незавершённое событие в конце потока (без пустой строки) отбрасывается, как в браузерах.

**Ошибки:**

- Невалидный JSON или событие, не подходящее под тип, — `APIResponseError`, как в буферном парсере. Причина — `ValidationError` в `__cause__`.
- SSE-событие `event: error` (Anthropic) и JSON-событие с ключом `error` на верхнем уровне (OpenAI) передаются в `error_resolver.resolve(status=..., body=..., headers=...)`. Получается тот же `APIStatusError`, что и для ответа с ошибочным статусом, а провайдер через свой `classify` решает, что `overloaded_error` — временная ошибка.

## Этапы

**Этап 1: декодеры (unit).**

- [x] `LineDecoder`: все варианты окончаний строк; разрыв в любой позиции тестового потока (параметризация по всем точкам разреза); многобайтовый символ на границе чанков; невалидный UTF-8 → `APIResponseError`.
- [x] `SSEDecoder`: все правила разбора выше, BOM, `[DONE]`, незавершённое событие в конце.

**Этап 2: парсер и методы (unit).**

- [x] NDJSON и SSE в `StreamResponseParser`: модели, дискриминированные объединения, `ServerSentEvent`, свой `json_loads`, ошибки валидации.
- [x] `BaseStreamMethod`: `ServerSentEvent` выводит SSE; модель и объединение с явным форматом принимаются; без формата — прежний `TypeError`; тип, для которого нельзя построить `TypeAdapter`, — `TypeError` при определении класса.
- [x] Обновить тесты, которые сейчас ждут `ValueError` для SSE/NDJSON.

**Этап 3: ошибки и сессия (фейковый транспорт).**

- [x] Событие ошибки → `APIStatusError` нужного класса и с `transient` от резолвера; ответ закрывается.
- [x] `break` посреди событий закрывает генератор парсера и ответ.

**Этап 4: интеграция, типы, документация.**

- [x] `aiohttp_server`: SSE в стиле OpenAI и Anthropic, NDJSON; чанки отдаются с паузами; соединение освобождается после `break`.
- [x] `tests/typing/check_public_api.py`: `assert_type` для `Stream[Model]`, `Stream[A | B]` и `Stream[ServerSentEvent]`.
- [x] README: раздел про события, тест в `tests/docs/test_readme.py`; CLAUDE.md: описание парсера; строка «not implemented yet» в `ResponseFormat` убирается.
- [x] ruff, mypy, pyright, pre-commit; покрытие не ниже `fail_under = 98`.

## Открытые вопросы

Все четыре решены 29.10.2026 вариантами по умолчанию.

- [x] **Ошибки внутри стрима через резолвер.** Это меняет протокол `IStreamParser` (новый аргумент). Альтернатива — отдельный класс `APIStreamError` без резолвера, но тогда провайдер не сможет решить, временная ли ошибка. **Решено: резолвер.**
- [x] **`[DONE]` по умолчанию.** Остановка на `data: [DONE]` нужна OpenAI и безвредна для остальных. **Решено: всегда.**
- [x] **Неизвестное событие** (например, `ping`, которого нет в объединении). Ошибка заставляет описать все события, пропуск теряет данные молча. **Решено: ошибка.** При необходимости можно взять `ServerSentEvent` и разбирать самому.
- [x] **Незавершённое событие в конце потока.** По стандарту отбрасывается. **Решено: по стандарту.**
