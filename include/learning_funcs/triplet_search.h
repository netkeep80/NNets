/*
 * triplet_search.h - Функции генерации тройки нейронов
 *
 * Этот модуль содержит функции генерации трёх связанных нейронов:
 * - triplet_random (бывш. rndrod4) - генерация тройки со случайным поиском
 * - triplet_random_parallel (бывш. rndrod4_parallel) - многопоточная версия
 *
 * Тройка нейронов состоит из:
 * - Нейрон A: комбинирует существующие нейроны
 * - Нейрон B: комбинирует существующие нейроны
 * - Нейрон C: объединяет A и B
 *
 * Это основной метод обучения в текущей версии, обеспечивающий
 * создание более сложных функций за счёт иерархической структуры.
 */

#ifndef TRIPLET_SEARCH_H
#define TRIPLET_SEARCH_H

#include "learning_func_base.h"

// ============================================================================
// Последовательная версия функции
// ============================================================================

/**
 * Генерация тройки нейронов (triplet_random)
 *
 * Создаёт три связанных нейрона: A, B и C.
 * C объединяет A и B, обеспечивая более сложные функции.
 * Это основной метод обучения в текущей версии.
 *
 * Создаёт: 3 нейрона (A, B, C)
 * Сложность: O(Neirons * Receptors * 4 * op_count^2)
 *
 * @return минимальная достигнутая ошибка, или big если не найдено
 */
float triplet_random() {
    int     count, count_max = Neirons * Receptors * 4;
    float   min = big;
    float   square, sum;
    int     A_id = Neirons;
    int     B_id = Neirons + 1;
    int     C_id = Neirons + 2;
    Neiron& Neiron_A = nei[A_id];
    Neiron& Neiron_B = nei[B_id];
    Neiron& Neiron_C = nei[C_id];
    Neiron  optimal_A, optimal_B, optimal_C;
    bool    finded = false;

    // C объединяет A и B
    Neiron_C.i = A_id;
    Neiron_C.j = B_id;

    // Инициализируем A случайными значениями
    Neiron_A.i = rand() % Neirons;
    Neiron_A.j = rand() % Neirons;
    Neiron_A.op = op[rand() % op_count];
    Neiron_A.cached = false;

    for (count = 0; count < count_max; count++)
    {
        // Генерируем случайные параметры для B
        Neiron_B.i = rand() % Neirons;
        Neiron_B.j = rand() % Neirons;

        // Перебираем операции для B и C
        for (int B_op = 0; B_op < op_count; B_op++)
        {
            Neiron_B.op = op[B_op];
            Neiron_B.cached = false;

            for (int C_op = 0; C_op < op_count; C_op++)
            {
                Neiron_C.op = op[C_op];
                Neiron_C.cached = false;

                float* C_Vector = GetNeironVector(C_id);
                sum = 0.0;

                // Вычисляем ошибку по всем образам
                for (int index = 0; index < Images && sum < min; index++)
                {
                    square = vz[index] - C_Vector[index];
                    sum += square * square;
                }

                if (min > sum)
                {
                    finded = true;
                    min = sum;
                    optimal_A = Neiron_A;
                    optimal_B = Neiron_B;
                    optimal_C = Neiron_C;

                    // Используем оптимальный нейрон B как новый A
                    Neiron_A = Neiron_B;
                }
            }
        }
    }

    if (finded)
    {
        Neiron_A = optimal_A;
        Neiron_B = optimal_B;
        Neiron_C = optimal_C;
        Neirons += 3;
        return min;
    }
    else
        return big;
}

// ============================================================================
// Многопоточные версии функций
// ============================================================================

/**
 * Структура для хранения результата поиска в потоке
 */
struct TripletSearchResult {
    float min_error;
    Neiron optimal_A;
    Neiron optimal_B;
    Neiron optimal_C;
    bool found;

    TripletSearchResult() : min_error(big), found(false) {}
};

/**
 * Функция потока для параллельного поиска оптимальных нейронов
 */
void TripletSearchThreadFunc(
    int thread_id,
    int iterations_per_thread,
    int current_neirons,
    unsigned int seed,
    TripletSearchResult& result,
    std::atomic<float>* global_min)
{
    // Локальный генератор случайных чисел для потока
    unsigned int local_seed = seed + thread_id * 1099087573u;
    auto local_rand = [&local_seed]() -> int {
        local_seed = local_seed * 1103515245u + 12345u;
        return (int)((local_seed >> 16) & 0x7FFF);
    };

    Neiron local_A, local_B, local_C;
    std::vector<float> A_Vector(Images), B_Vector(Images), C_Vector(Images);

    // Инициализируем A случайными значениями
    local_A.i = local_rand() % current_neirons;
    local_A.j = local_rand() % current_neirons;
    local_A.op = op[local_rand() % op_count];

    float* A_i_cache = GetNeironVector(local_A.i);
    float* A_j_cache = GetNeironVector(local_A.j);
    (*local_A.op)(A_Vector.data(), A_i_cache, A_j_cache, Images);

    for (int count = 0; count < iterations_per_thread; count++)
    {
        // Генерируем случайные параметры для B
        local_B.i = local_rand() % current_neirons;
        local_B.j = local_rand() % current_neirons;

        float* B_i_cache = GetNeironVector(local_B.i);
        float* B_j_cache = GetNeironVector(local_B.j);

        // Перебираем операции для B и C
        for (int B_op = 0; B_op < op_count; B_op++)
        {
            local_B.op = op[B_op];
            (*local_B.op)(B_Vector.data(), B_i_cache, B_j_cache, Images);

            for (int C_op = 0; C_op < op_count; C_op++)
            {
                local_C.op = op[C_op];
                (*local_C.op)(C_Vector.data(), A_Vector.data(), B_Vector.data(), Images);

                // Вычисляем ошибку по всем образам
                float current_global_min = global_min->load(std::memory_order_relaxed);
                float sum = 0.0f;
                for (int index = 0; index < Images && sum < current_global_min; index++)
                {
                    float square = vz[index] - C_Vector[index];
                    sum += square * square;
                }

                if (result.min_error > sum)
                {
                    result.found = true;
                    result.min_error = sum;
                    result.optimal_A = local_A;
                    result.optimal_B = local_B;
                    result.optimal_C = local_C;

                    // Обновляем глобальный минимум для early termination в других потоках
                    float expected = global_min->load(std::memory_order_relaxed);
                    while (sum < expected) {
                        if (global_min->compare_exchange_weak(expected, sum, std::memory_order_relaxed)) {
                            break;
                        }
                    }

                    // Используем оптимальный нейрон B как новый A
                    local_A = local_B;
                    for (int im = 0; im < Images; im++) {
                        A_Vector[im] = B_Vector[im];
                    }
                }
            }
        }
    }
}

/**
 * Многопоточная генерация тройки нейронов (triplet_random_parallel)
 *
 * Параллельная версия triplet_random(), распределяющая поиск
 * между несколькими потоками. Каждый поток исследует
 * случайные комбинации параметров нейронов независимо.
 *
 * Стратегия распределения работы:
 * - Общее количество итераций делится между потоками
 * - Это обеспечивает приблизительно такое же качество поиска
 *   как однопоточная версия, но за меньшее время
 * - Каждый поток имеет минимальное гарантированное количество итераций
 *
 * Создаёт: 3 нейрона (A, B, C)
 *
 * @return минимальная достигнутая ошибка, или big если не найдено
 */
float triplet_random_parallel() {
    // Если многопоточность отключена или только 1 поток, используем последовательную версию
    if (!UseMultithreading || NumThreads <= 1) {
        return triplet_random();
    }

    // Общее количество итераций для однопоточного поиска
    int count_max = Neirons * Receptors * 4;

    // Минимальное количество итераций на поток для качественного поиска
    const int MIN_ITERATIONS_PER_THREAD = 1000;

    // Вычисляем количество итераций на поток
    int base_iterations_per_thread = (count_max + NumThreads - 1) / NumThreads;
    int iterations_per_thread = std::max(base_iterations_per_thread, MIN_ITERATIONS_PER_THREAD);

    // Используем однопоточную версию если count_max слишком мал
    if (count_max < 2000) {
        return triplet_random();
    }

    // Создаём результаты для каждого потока
    std::vector<TripletSearchResult> results(NumThreads);

    // Атомарная переменная для синхронизации глобального минимума
    std::atomic<float> global_min(big);

    // Получаем начальное значение для генераторов случайных чисел
    unsigned int base_seed = (unsigned int)(Neirons * 1099087573u + 12345u);

    // Предварительно прогреваем кэши всех существующих нейронов
    for (int n = 0; n < Neirons; n++) {
        GetNeironVector(n);
    }

    // Запускаем потоки
    std::vector<std::thread> threads;
    threads.reserve(NumThreads);

    for (int t = 0; t < NumThreads; t++) {
        threads.emplace_back(TripletSearchThreadFunc,
                             t,
                             iterations_per_thread,
                             Neirons,
                             base_seed,
                             std::ref(results[t]),
                             &global_min);
    }

    // Ждём завершения всех потоков
    for (auto& t : threads) {
        t.join();
    }

    // Находим лучший результат среди всех потоков
    float best_min = big;
    int best_thread = -1;
    for (int t = 0; t < NumThreads; t++) {
        if (results[t].found && results[t].min_error < best_min) {
            best_min = results[t].min_error;
            best_thread = t;
        }
    }

    if (best_thread >= 0) {
        // Сохраняем оптимальные нейроны
        int A_id = Neirons;
        int B_id = Neirons + 1;
        int C_id = Neirons + 2;

        nei[A_id] = results[best_thread].optimal_A;
        nei[A_id].cached = false;
        nei[B_id] = results[best_thread].optimal_B;
        nei[B_id].cached = false;
        nei[C_id] = results[best_thread].optimal_C;
        nei[C_id].cached = false;

        // Устанавливаем правильные связи для C
        nei[C_id].i = A_id;
        nei[C_id].j = B_id;

        Neirons += 3;
        return best_min;
    }
    else {
        return big;
    }
}

// Сохраняем обратную совместимость со старыми именами
inline float rndrod4() { return triplet_random(); }
inline float rndrod4_parallel() { return triplet_random_parallel(); }

#endif // TRIPLET_SEARCH_H
