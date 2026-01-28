/*
 * random_search.h - Функции случайного поиска для генерации нейронов
 *
 * Этот модуль содержит функции, использующие случайный поиск:
 * - random_neurons (бывш. rndrod) - случайная генерация заданного количества нейронов
 * - random_from_inputs (бывш. rndrod0) - случайная генерация на основе входов
 * - random_pair_optimized (бывш. rndrod2) - оптимизированная генерация пары нейронов
 * - random_pair_extended (бывш. rndrod3) - расширенная генерация пары нейронов
 *
 * Эти функции работают быстрее детерминированных методов, но не гарантируют
 * нахождение оптимального решения.
 */

#ifndef RANDOM_SEARCH_H
#define RANDOM_SEARCH_H

#include "learning_func_base.h"
#include <cstdlib>

// ============================================================================
// Последовательные версии функций
// ============================================================================

/**
 * Случайная генерация нейронов (random_neurons)
 *
 * Создаёт заданное количество случайных нейронов.
 * Используется для расширения пространства поиска.
 *
 * Создаёт: count нейронов
 * Сложность: O(count)
 *
 * @param count - количество создаваемых нейронов
 */
void random_neurons_n(unsigned count) {
    do
    {
        nei[Neirons].cached = false;
        nei[Neirons].i = rand() % (Neirons);
        nei[Neirons].j = rand() % (Neirons);
        nei[Neirons].op = op[rand() % op_count];
        std::cout << "(" << Neirons << ") = (" << nei[Neirons].i << ")op(" << nei[Neirons].j << ")\n";
        Neirons++;
    } while (--count > 0);
}

/**
 * Случайная генерация одного нейрона
 *
 * Создаёт один случайный нейрон с вычислением ошибки.
 * Обёртка для использования в системе функций обучения.
 *
 * @return ошибка нейрона (для совместимости с интерфейсом)
 */
float random_neurons() {
    nei[Neirons].cached = false;
    nei[Neirons].i = rand() % (Neirons);
    nei[Neirons].j = rand() % (Neirons);
    nei[Neirons].op = op[rand() % op_count];

    // Вычисляем ошибку созданного нейрона
    float* curval = GetNeironVector(Neirons);
    float sum = 0.0f;
    for (int index = 0; index < Images; index++) {
        float square = vz[index] - curval[index];
        sum += square * square;
    }

    std::cout << "(" << Neirons << ") = (" << nei[Neirons].i << ")op(" << nei[Neirons].j << "), error = " << sum << "\n";
    Neirons++;
    return sum;
}

/**
 * Случайная генерация на основе входов (random_from_inputs)
 *
 * Создаёт случайные нейроны, комбинируя только входы сети.
 * Полезно на начальных этапах обучения для создания базовых комбинаций.
 *
 * Создаёт: count нейронов
 * Сложность: O(count)
 *
 * @param count - количество создаваемых нейронов
 */
void random_from_inputs_n(unsigned count) {
    do
    {
        nei[Neirons].cached = false;
        nei[Neirons].i = rand() % (Inputs);
        nei[Neirons].j = rand() % (Receptors);
        nei[Neirons].op = op[rand() % op_count];
        std::cout << "(" << Neirons << ") = (" << nei[Neirons].i << ")op(" << nei[Neirons].j << ")\n";
        Neirons++;
    } while (--count > 0);
}

/**
 * Случайная генерация одного нейрона на основе входов
 *
 * @return ошибка нейрона
 */
float random_from_inputs() {
    nei[Neirons].cached = false;
    nei[Neirons].i = rand() % (Inputs);
    nei[Neirons].j = rand() % (Receptors);
    nei[Neirons].op = op[rand() % op_count];

    // Вычисляем ошибку
    float* curval = GetNeironVector(Neirons);
    float sum = 0.0f;
    for (int index = 0; index < Images; index++) {
        float square = vz[index] - curval[index];
        sum += square * square;
    }

    std::cout << "(" << Neirons << ") = (" << nei[Neirons].i << ")op(" << nei[Neirons].j << "), error = " << sum << "\n";
    Neirons++;
    return sum;
}

/**
 * Оптимизированная случайная генерация пары нейронов (random_pair_optimized)
 *
 * Создаёт пару нейронов с оптимизированными параметрами.
 * Ищет лучшую комбинацию среди случайных вариантов.
 * Первый нейрон комбинирует недавно созданные нейроны с остальными.
 *
 * Создаёт: 2 нейрона
 * Сложность: O(Inputs * Neirons * rndrod_iter)
 *
 * @return минимальная достигнутая ошибка
 */
float random_pair_optimized() {
    int     count, count_max = Inputs * Neirons * rndrod_iter;
    float   min = big;
    float   square, sum;
    int     r[5] = { 0,0,0,0,0 };
    oper    ro[5] = { 0,0,0,0,0 };
    int     Neirons_p_1 = Neirons + 1;
    Neiron& Neiron_A = nei[Neirons];
    Neiron& Neiron_B = nei[Neirons_p_1];

    Neiron_B.i = Neirons;

    for (count = 0; count < count_max; count++)
    {
        Neiron_A.cached = false;
        Neiron_A.i = rand() % rndrod_iter + Neirons - rndrod_iter;  // Последние случайные
        if (Neiron_A.i < 0) Neiron_A.i = 0;
        Neiron_A.j = rand() % std::max(1, Neirons - rndrod_iter);
        Neiron_A.op = op[rand() % op_count];

        Neiron_B.cached = false;
        Neiron_B.j = rand() % Inputs;
        Neiron_B.op = op[rand() % op_count];

        float* NBVal = GetNeironVector(Neirons_p_1);

        sum = 0.0;

        for (int index = 0; index < Images && sum < min; index++)
        {
            square = vz[index] - NBVal[index];
            sum += square * square;
        }

        if (min > sum)
        {
            min = sum;
            ro[0] = Neiron_A.op;
            r[1] = Neiron_A.i;
            r[2] = Neiron_A.j;
            ro[3] = Neiron_B.op;
            r[4] = Neiron_B.j;
        }
    }

    Neiron_A.cached = false;
    Neiron_A.i = r[1];
    Neiron_A.j = r[2];
    Neiron_A.op = ro[0];
    Neiron_B.cached = false;
    Neiron_B.j = r[4];
    Neiron_B.op = ro[3];
    std::cout << "min = " << min << ", (" << Neirons + 1 << ") = ((" << r[1] << ")op(" << r[2] << "))op(" << r[4] << ")\n";
    Neirons += 2;
    return min;
}

/**
 * Расширенная случайная генерация пары нейронов (random_pair_extended)
 *
 * Аналогична random_pair_optimized(), но с большим пространством поиска.
 * Оба входа нейрона A выбираются из всех существующих нейронов.
 *
 * Создаёт: 2 нейрона
 * Сложность: O(Neirons^2 * 6)
 *
 * @return минимальная достигнутая ошибка
 */
float random_pair_extended() {
    int     count, count_max = Neirons * Neirons * 6;
    float   min = big;
    float   square, sum;
    int     r[5] = { 0,0,0,0,0 };
    oper    ro[5] = { 0,0,0,0,0 };
    int     Neirons_p_1 = Neirons + 1;
    Neiron& Neiron_A = nei[Neirons];
    Neiron& Neiron_B = nei[Neirons_p_1];

    Neiron_B.i = Neirons;

    for (count = 0; count < count_max; count++)
    {
        Neiron_A.cached = false;
        Neiron_A.i = rand() % Neirons;
        Neiron_A.j = rand() % Neirons;
        Neiron_A.op = op[rand() % op_count];

        Neiron_B.cached = false;
        Neiron_B.j = rand() % Neirons;
        Neiron_B.op = op[rand() % op_count];

        float* NBVal = GetNeironVector(Neirons_p_1);

        sum = 0.0;

        for (int index = 0; index < Images && sum < min; index++)
        {
            square = vz[index] - NBVal[index];
            sum += square * square;
        }

        if (min > sum)
        {
            min = sum;
            ro[0] = Neiron_A.op;
            r[1] = Neiron_A.i;
            r[2] = Neiron_A.j;
            ro[3] = Neiron_B.op;
            r[4] = Neiron_B.j;
        }
    }

    Neiron_A.cached = false;
    Neiron_A.i = r[1];
    Neiron_A.j = r[2];
    Neiron_A.op = ro[0];
    Neiron_B.cached = false;
    Neiron_B.j = r[4];
    Neiron_B.op = ro[3];
    std::cout << "min = " << min << ", (" << Neirons + 1 << ") = ((" << r[1] << ")op(" << r[2] << "))op(" << r[4] << ")\n";
    Neirons += 2;
    return min;
}

// ============================================================================
// Многопоточные версии функций
// ============================================================================

/**
 * Структура результата для пары нейронов
 */
struct PairSearchResult {
    float min_error;
    int A_i, A_j, B_j;
    int A_op_index, B_op_index;
    bool found;

    PairSearchResult() : min_error(big), A_i(0), A_j(0), B_j(0), A_op_index(0), B_op_index(0), found(false) {}
};

/**
 * Функция потока для параллельного поиска пары нейронов
 */
void PairSearchThreadFunc(
    int thread_id,
    int iterations_per_thread,
    int current_neirons,
    unsigned int seed,
    bool optimized_mode,
    PairSearchResult& result,
    std::atomic<float>* global_min)
{
    // Локальный генератор случайных чисел
    unsigned int local_seed = seed + thread_id * 1099087573u;
    auto local_rand = [&local_seed]() -> int {
        local_seed = local_seed * 1103515245u + 12345u;
        return (int)((local_seed >> 16) & 0x7FFF);
    };

    std::vector<float> A_Vector(Images), B_Vector(Images);

    for (int count = 0; count < iterations_per_thread; count++)
    {
        int A_i, A_j, B_j;

        if (optimized_mode) {
            // Режим random_pair_optimized
            A_i = local_rand() % rndrod_iter + current_neirons - rndrod_iter;
            if (A_i < 0) A_i = 0;
            A_j = local_rand() % std::max(1, current_neirons - rndrod_iter);
            B_j = local_rand() % Inputs;
        } else {
            // Режим random_pair_extended
            A_i = local_rand() % current_neirons;
            A_j = local_rand() % current_neirons;
            B_j = local_rand() % current_neirons;
        }

        float* A_i_cache = GetNeironVector(A_i);
        float* A_j_cache = GetNeironVector(A_j);
        float* B_j_cache = GetNeironVector(B_j);

        for (int A_op = 0; A_op < op_count; A_op++)
        {
            (*op[A_op])(A_Vector.data(), A_i_cache, A_j_cache, Images);

            for (int B_op = 0; B_op < op_count; B_op++)
            {
                (*op[B_op])(B_Vector.data(), A_Vector.data(), B_j_cache, Images);

                float current_global_min = global_min->load(std::memory_order_relaxed);
                float sum = 0.0f;

                for (int index = 0; index < Images && sum < current_global_min; index++)
                {
                    float square = vz[index] - B_Vector[index];
                    sum += square * square;
                }

                if (result.min_error > sum)
                {
                    result.found = true;
                    result.min_error = sum;
                    result.A_i = A_i;
                    result.A_j = A_j;
                    result.B_j = B_j;
                    result.A_op_index = A_op;
                    result.B_op_index = B_op;

                    float expected = global_min->load(std::memory_order_relaxed);
                    while (sum < expected) {
                        if (global_min->compare_exchange_weak(expected, sum, std::memory_order_relaxed)) {
                            break;
                        }
                    }
                }
            }
        }
    }
}

/**
 * Параллельная оптимизированная генерация пары нейронов
 *
 * @return минимальная достигнутая ошибка
 */
float random_pair_optimized_parallel() {
    if (!UseMultithreading || NumThreads <= 1) {
        return random_pair_optimized();
    }

    int count_max = Inputs * Neirons * rndrod_iter;
    int iterations_per_thread = std::max(100, (count_max + NumThreads - 1) / NumThreads);

    // Прогреваем кэши
    for (int n = 0; n < Neirons; n++) {
        GetNeironVector(n);
    }

    std::vector<PairSearchResult> results(NumThreads);
    std::atomic<float> global_min(big);
    std::vector<std::thread> threads;
    threads.reserve(NumThreads);

    unsigned int base_seed = (unsigned int)(Neirons * 1099087573u + 12345u);

    for (int t = 0; t < NumThreads; t++) {
        threads.emplace_back(PairSearchThreadFunc,
                             t, iterations_per_thread, Neirons,
                             base_seed, true,
                             std::ref(results[t]), &global_min);
    }

    for (auto& t : threads) {
        t.join();
    }

    // Находим лучший результат
    float best_min = big;
    int best_thread = -1;
    for (int t = 0; t < NumThreads; t++) {
        if (results[t].found && results[t].min_error < best_min) {
            best_min = results[t].min_error;
            best_thread = t;
        }
    }

    if (best_thread >= 0) {
        Neiron& Neiron_A = nei[Neirons];
        Neiron& Neiron_B = nei[Neirons + 1];

        Neiron_A.cached = false;
        Neiron_A.i = results[best_thread].A_i;
        Neiron_A.j = results[best_thread].A_j;
        Neiron_A.op = op[results[best_thread].A_op_index];

        Neiron_B.cached = false;
        Neiron_B.i = Neirons;
        Neiron_B.j = results[best_thread].B_j;
        Neiron_B.op = op[results[best_thread].B_op_index];

        std::cout << "min = " << best_min << ", (" << Neirons + 1 << ") = (("
                  << Neiron_A.i << ")op(" << Neiron_A.j << "))op(" << Neiron_B.j << ") [parallel]\n";
        Neirons += 2;
        return best_min;
    }

    return big;
}

/**
 * Параллельная расширенная генерация пары нейронов
 *
 * @return минимальная достигнутая ошибка
 */
float random_pair_extended_parallel() {
    if (!UseMultithreading || NumThreads <= 1) {
        return random_pair_extended();
    }

    int count_max = Neirons * Neirons * 6;
    int iterations_per_thread = std::max(100, (count_max + NumThreads - 1) / NumThreads);

    // Прогреваем кэши
    for (int n = 0; n < Neirons; n++) {
        GetNeironVector(n);
    }

    std::vector<PairSearchResult> results(NumThreads);
    std::atomic<float> global_min(big);
    std::vector<std::thread> threads;
    threads.reserve(NumThreads);

    unsigned int base_seed = (unsigned int)(Neirons * 1099087573u + 12345u);

    for (int t = 0; t < NumThreads; t++) {
        threads.emplace_back(PairSearchThreadFunc,
                             t, iterations_per_thread, Neirons,
                             base_seed, false,
                             std::ref(results[t]), &global_min);
    }

    for (auto& t : threads) {
        t.join();
    }

    // Находим лучший результат
    float best_min = big;
    int best_thread = -1;
    for (int t = 0; t < NumThreads; t++) {
        if (results[t].found && results[t].min_error < best_min) {
            best_min = results[t].min_error;
            best_thread = t;
        }
    }

    if (best_thread >= 0) {
        Neiron& Neiron_A = nei[Neirons];
        Neiron& Neiron_B = nei[Neirons + 1];

        Neiron_A.cached = false;
        Neiron_A.i = results[best_thread].A_i;
        Neiron_A.j = results[best_thread].A_j;
        Neiron_A.op = op[results[best_thread].A_op_index];

        Neiron_B.cached = false;
        Neiron_B.i = Neirons;
        Neiron_B.j = results[best_thread].B_j;
        Neiron_B.op = op[results[best_thread].B_op_index];

        std::cout << "min = " << best_min << ", (" << Neirons + 1 << ") = (("
                  << Neiron_A.i << ")op(" << Neiron_A.j << "))op(" << Neiron_B.j << ") [parallel]\n";
        Neirons += 2;
        return best_min;
    }

    return big;
}

// Сохраняем обратную совместимость со старыми именами
inline void rndrod(unsigned count) { random_neurons_n(count); }
inline void rndrod0(unsigned count) { random_from_inputs_n(count); }
inline float rndrod2() { return random_pair_optimized(); }
inline float rndrod3() { return random_pair_extended(); }
inline float rndrod2_parallel() { return random_pair_optimized_parallel(); }
inline float rndrod3_parallel() { return random_pair_extended_parallel(); }

#endif // RANDOM_SEARCH_H
