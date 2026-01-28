# NNets — Самоструктурирующаяся нейронная сеть / Self-Structuring Neural Network

[English version below](#english-version)

---

## Русская версия

### Описание

NNets — это реализация самоструктурирующейся нейронной сети на языке C++, которая автоматически формирует свою архитектуру в процессе обучения. Вместо заранее определённой структуры, сеть динамически добавляет новые нейроны и связи для минимизации ошибки классификации.

### Ключевые особенности

- **Динамическая структура**: Сеть автоматически растёт, добавляя нейроны по мере необходимости
- **Обучение через генерацию**: Вместо корректировки весов создаются новые нейроны с оптимальными параметрами
- **14 алгоритмов обучения**: Полный перебор, случайный поиск, генерация тройки нейронов
- **Многопоточность**: Параллельные версии всех основных алгоритмов
- **SIMD-оптимизации**: Поддержка AVX и SSE для ускорения вычислений
- **Кроссплатформенность**: Linux, Windows, macOS
- **Сохранение и загрузка моделей**: Формат JSON для переносимости
- **Дообучение**: Возможность добавления новых классов к существующей модели

### Документация

- **[analysis.md](analysis.md)** — Подробный анализ текущего состояния проекта
- **[plan.md](plan.md)** — План развития проекта

### Быстрый старт

#### Сборка

```bash
# Конфигурация
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Сборка
cmake --build build --config Release
```

#### Обучение

```bash
# Обучение с конфигурацией по умолчанию
./build/NNets

# Обучение с кастомной конфигурацией и сохранением модели
./build/NNets -c configs/default.json -s model.json

# Обучение с автоматическим тестом
./build/NNets -c configs/simple.json -s model.json -t
```

#### Использование обученной модели

```bash
# Интерактивный режим
./build/NNets -l model.json

# Классификация одного текста
./build/NNets -l model.json -i "time"

# Проверка точности
./build/NNets -l model.json -c configs/test.json --verify
```

#### Дообучение

```bash
# Добавление новых классов к существующей модели
./build/NNets -r model_v1.json -c configs/extended.json -s model_v2.json
```

### Опции командной строки

```
РЕЖИМЫ РАБОТЫ:
  Обучение (по умолчанию): Обучение сети с нуля
  Инференс (-l): Использование обученной модели
  Дообучение (-r): Добавление новых классов к модели

ПАРАМЕТРЫ ОБУЧЕНИЯ:
  -c, --config <файл>  Загрузить конфигурацию из JSON файла
  -s, --save <файл>    Сохранить обученную модель в JSON файл
  -t, --test           Запустить автоматический тест после обучения
  -b, --benchmark      Измерить скорость обучения

ПАРАМЕТРЫ ИНФЕРЕНСА:
  -l, --load <файл>    Загрузить модель для классификации
  -i, --input <текст>  Классифицировать один текст и выйти
  --verify             Проверить точность модели на данных из конфига

ПАРАМЕТРЫ ПРОИЗВОДИТЕЛЬНОСТИ:
  -j, --threads <n>    Количество потоков (0 = авто)
  --single-thread      Отключить многопоточность
  --no-simd            Отключить SIMD-оптимизации

ДРУГОЕ:
  -h, --help           Показать справку
  --list-funcs         Показать список функций обучения
```

### Формат конфигурации

```json
{
  "description": "Описание конфигурации",
  "receptors": 20,
  "classes": [
    { "id": 0, "word": "" },
    { "id": 1, "word": "yes" },
    { "id": 2, "word": "no" }
  ],
  "generate_shifts": true,
  "funcs": ["triplet_parallel"]
}
```

### Тестирование

```bash
# Запуск всех тестов
ctest --test-dir build -C Release

# Запуск с подробным выводом
ctest --test-dir build -C Release --output-on-failure
```

### Лицензия

Проект распространяется под лицензией Unlicense. Вы можете использовать, модифицировать и распространять код без ограничений.

---

## English Version

### Description

NNets is a C++ implementation of a self-structuring neural network that automatically builds its architecture during training. Instead of a predefined structure, the network dynamically adds new neurons and connections to minimize classification error.

### Key Features

- **Dynamic Structure**: The network automatically grows by adding neurons as needed
- **Learning through Generation**: Instead of adjusting weights, new neurons with optimal parameters are created
- **14 Training Algorithms**: Exhaustive search, random search, triplet neuron generation
- **Multithreading**: Parallel versions of all main algorithms
- **SIMD Optimizations**: AVX and SSE support for computational acceleration
- **Cross-platform**: Linux, Windows, macOS
- **Model Save/Load**: JSON format for portability
- **Retraining**: Ability to add new classes to an existing model

### Documentation

- **[analysis.md](analysis.md)** — Detailed analysis of the current project state (in Russian)
- **[plan.md](plan.md)** — Project development plan (in Russian)

### Quick Start

#### Build

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release
```

#### Training

```bash
# Train with default configuration
./build/NNets

# Train with custom config and save model
./build/NNets -c configs/default.json -s model.json

# Train with automated test
./build/NNets -c configs/simple.json -s model.json -t
```

#### Using a Trained Model

```bash
# Interactive mode
./build/NNets -l model.json

# Classify single text
./build/NNets -l model.json -i "time"

# Verify accuracy
./build/NNets -l model.json -c configs/test.json --verify
```

#### Retraining

```bash
# Add new classes to an existing model
./build/NNets -r model_v1.json -c configs/extended.json -s model_v2.json
```

### Command Line Options

```
MODES:
  Training (default): Train network from scratch
  Inference (-l): Use trained model
  Retraining (-r): Add new classes to model

TRAINING OPTIONS:
  -c, --config <file>  Load configuration from JSON file
  -s, --save <file>    Save trained model to JSON file
  -t, --test           Run automated test after training
  -b, --benchmark      Measure training speed

INFERENCE OPTIONS:
  -l, --load <file>    Load model for classification
  -i, --input <text>   Classify single text and exit
  --verify             Verify model accuracy on config data

PERFORMANCE OPTIONS:
  -j, --threads <n>    Number of threads (0 = auto)
  --single-thread      Disable multithreading
  --no-simd            Disable SIMD optimizations

OTHER:
  -h, --help           Show help message
  --list-funcs         List available training functions
```

### Configuration Format

```json
{
  "description": "Configuration description",
  "receptors": 20,
  "classes": [
    { "id": 0, "word": "" },
    { "id": 1, "word": "yes" },
    { "id": 2, "word": "no" }
  ],
  "generate_shifts": true,
  "funcs": ["triplet_parallel"]
}
```

### Saved Network Format

```json
{
  "version": "1.0",
  "description": "Trained neural network model",
  "receptors": 12,
  "base_size": 14,
  "inputs": 26,
  "neurons_count": 308,
  "basis": [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, -0.125, -0.25, -0.5, -1.0, -2.0, -4.0, -8.0],
  "classes": [
    { "id": 0, "name": "", "output_neuron": 109 },
    { "id": 1, "name": "yes", "output_neuron": 298 }
  ],
  "neurons": [
    { "i": 12, "j": 13, "op": 3 },
    { "i": 5, "j": 26, "op": 1 }
  ]
}
```

### Example Output

```
Input word: time
99% - time
1% - hour
0% - main
0% -
Input word: hour
0% - time
98% - hour
0% - main
1% -
```

### Training Algorithms

The network supports multiple training algorithms configurable via the `funcs` parameter:

**Exhaustive Search** (deterministic):
- `exhaustive_full` / `exhaustive_full_parallel` — Complete enumeration of all neuron pairs
- `exhaustive_last` / `exhaustive_last_parallel` — Combine with the last created neuron
- `combine_old_new` / `combine_old_new_parallel` — Combine old and new neurons

**Random Search**:
- `random_single` — Generate a single random neuron
- `random_from_inputs` — Random generation based on inputs
- `random_pair_opt` / `random_pair_opt_parallel` — Optimized pair generation
- `random_pair_ext` / `random_pair_ext_parallel` — Extended pair generation

**Triplet Generation** (recommended):
- `triplet` / `triplet_parallel` — Create three connected neurons (A, B, C)

Default: `triplet_parallel`

### Testing

```bash
# Run all tests
ctest --test-dir build -C Release

# Run with verbose output
ctest --test-dir build -C Release --output-on-failure
```

### License

This project is licensed under the Unlicense. You are free to use, modify, and distribute this software for any purpose without any restrictions.

### Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please feel free to open an issue or submit a pull request.

### Disclaimer

This project is for educational and experimental purposes. It is not intended for production use or critical applications. The network's accuracy and performance may vary depending on the dataset and learning parameters.
