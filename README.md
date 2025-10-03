# Реализация DPO

Проект для дообучения GPT2 на датасете Anthropic/hh-rlhf с использованием метода Direct Preference Optimization (DPO) из статьи [Direct Preference Optimization:
Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290). Реализация включает полный тренировочный и тестовый цикл с планированием темпов обучения и ранней остановки, модификацию модели через PEFT адаптеры и соревновательную оценку подхода.

## Структура проекта

```
src/
├── data/
│   ├── load_llama.py          # Загрузка модели Llama для оценки
│   └── prepare_dataset.py     # Подготовка и обработка датасетов
├── model/
│   ├── make_prediction.py     # Функции для генерации предсказаний
│   ├── peft.py               # Настройка PEFT (Parameter-Efficient Fine-Tuning)
│   └── save_weights.py       # Сохранение весов модели
├── utils/
│   ├── eval_iter.py          # Функции для оценки модели
│   ├── loss.py               # Реализация функции потерь DPO
│   ├── lr_scheduler.py       # Планировщик обучения с warmup
│   ├── run.py                # Основной цикл обучения
│   ├── train_iter.py         # Итерация обучения
│   └── win_rating.py         # Оценка побед между моделями
├── main.py                   # Основной скрипт
└── tutorial/
    └── run_all.ipynb         # Пример запуска
```

## Быстрый старт

### Установка зависимостей

```bash
pip -q install -r requirements.txt
!CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip3 -q install llama-cpp-python
!pip3 -q install huggingface-hub
!pip3 -q install sentence-transformers langchain langchain-experimental
!huggingface-cli download TheBloke/Llama-2-7b-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir /content --local-dir-use-symlinks False
```

### Пример использования

```python
from src.main import run_train, run_eval

# Обучение модели
model = run_train(
    model_name="gpt2",
    dataset="Anthropic/hh-rlhf", 
    num_examples=1000,
    batch_size=2,
    num_epochs=10
)

# Оценка модели
run_eval(
    model_name="gpt2",
    num_test_questions=5,
    load_path="./checkpoints/last_model_final.pt"
)
```

## Конфигурация

### Среда

Инициализирует основные компоненты системы:
- Загружает модель и токенизатор
- Настраивает PEFT адаптеры
- Подготавливает reference model
- Загружает датасет

```python
get_basic_parts(
    model_name="gpt2",              # Предобученная модель
    dataset="Anthropic/hh-rlhf",    # Датасет для обучения
    load_path="./checkpoints/best_model.pth"  # Путь к дообученной модели
)
```

### Обучение

Запускает полный цикл обучения DPO:
- Подготовка данных и DataLoader'ов
- Настройка оптимизатора и планировщика
- Цикл обучения с валидацией
- Автосохранение чекпоинтов

```python
run_train(
    model_name="gpt2",              # Предобученная модель
    dataset="Anthropic/hh-rlhf",    # Датасет для обучения
    num_examples=3000,              # Количество примеров
    batch_size=5,                   # Размер батча
    lr=1e-6,                        # Скорость обучения
    beta=0.1,                       # Параметр DPO
    warmup_steps=150,               # Шаги warmup
    num_epochs=300,                 # Максимальное количество эпох
    early_stop_patience=50,         # Ранняя остановка
    checkpoint_dir='./checkpoints/' # Директория для чекпоинтов
)
```

### Оценка


Проводит сравнительную оценку модели:
- Сравнение с базовой моделью
- Сравнение с правильными ответами
- Использование LLM-эвалюатора

```python
run_eval(
    model_name="gpt2",
    dataset="Anthropic/hh-rlhf", 
    num_test_questions=5,           # Количество тестовых вопросов
    load_path="./checkpoints/best_model.pth"  # Путь к дообученной модели
)
```


## Метрики оценки

Проект использует два типа сравнений:

1. **DPO vs Base Model** - сравнение обученной модели с оригинальной
2. **DPO vs Correct Answers** - сравнение с эталонными ответами из датасета

Результаты выводятся в формате:
```
Number of wins of the base model: X
Number of wins of the DPO model: Y
Winner: DPO/Base/Both
```

## Особенности реализации

- **Эффективное обучение**: PEFT адаптеры сокращают количество обучаемых параметров в 2 раза
- **Гибкая архитектура**: Возможно использование различных моделей и датасетов
- **Автоматизация**: Автосохранение, ранняя остановка, warmup scheduling
- **Win rate оценка**: Оценка качества моделей в формате соревнования

