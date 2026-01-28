/*
 * exhaustive_search.h - Функции полного перебора для генерации нейронов
 *
 * Этот модуль содержит функции, использующие полный перебор комбинаций:
 * - exhaustive_full_search (бывш. rod) - полный перебор всех пар нейронов
 * - exhaustive_last_combine (бывш. rod2) - комбинирование с последним нейроном
 * - combine_old_new (бывш. rod3) - комбинирование старых нейронов с новыми
 *
 * Эти функции гарантируют нахождение оптимального решения в пределах
 * заданного пространства поиска, но работают медленнее случайных методов.
 */

#ifndef EXHAUSTIVE_SEARCH_H
#define EXHAUSTIVE_SEARCH_H

#include "learning_func_base.h"

// ============================================================================
// Последовательные версии функций
// ============================================================================

/**
 * Полный перебор комбинаций нейронов (exhaustive_full_search)
 *
 * Перебирает все комбинации пар нейронов и операций,
 * выбирая оптимальную комбинацию с минимальной ошибкой.
 * Медленный, но гарантирует нахождение лучшего нейрона.
 *
 * Создаёт: 1 нейрон
 * Сложность: O(N^2 * O) где N - количество нейронов, O - количество операций
 *
 * @return минимальная достигнутая ошибка
 */
float exhaustive_full_search() {
    int     i;
    float   min = big;
    int     optimal_i = 0;
    int     optimal_j = 0;
    oper    optimal_op = op[0];
    float   square, sum;
    Neiron& cur = nei[Neirons];
    float*  curval;

    for (cur.i = 1; cur.i < Neirons; cur.i++)               // Выбор 1-го нейрона
    {
        for (cur.j = 0; cur.j < cur.i; cur.j++)             // Выбор 2-го нейрона
        {
            for (i = 0; i < op_count; i++)                  // Выбор операции
            {
                cur.cached = false;
                cur.op = op[i];
                sum = 0.0;
                curval = GetNeironVector(Neirons);

                // Вычисляем сумму квадратов ошибок
                for (int index = 0; index < Images && sum < min; index++)
                {
                    square = vz[index] - curval[index];
                    sum += square * square;
                }

                if (min > sum)
                {
                    min = sum;
                    optimal_op = cur.op;
                    optimal_i = cur.i;
                    optimal_j = cur.j;
                }
            }
        }
    }

    // Сохраняем оптимальные параметры
    cur.cached = false;
    cur.i = optimal_i;
    cur.j = optimal_j;
    cur.op = optimal_op;
    std::cout << "min = " << min << ", (" << Neirons << ") = (" << optimal_i << ")op(" << optimal_j << ")\n";
    Neirons++;
    return min;
}

/**
 * Комбинирование с последним нейроном (exhaustive_last_combine)
 *
 * Фиксирует первый вход как последний созданный нейрон
 * и перебирает только второй вход и операцию.
 * Быстрее, чем exhaustive_full_search(), но менее гибкий.
 *
 * Создаёт: 1 нейрон
 * Сложность: O(N * O) где N - количество нейронов, O - количество операций
 *
 * @return минимальная достигнутая ошибка
 */
float exhaustive_last_combine() {
    int     i;
    float   min = big;
    int     optimal_i = 0;
    int     optimal_j = 0;
    oper    optimal_op = op[0];
    float   square, sum;
    Neiron& cur = nei[Neirons];
    float*  curval;

    cur.i = Neirons - 1;                                    // Фиксируем последний нейрон

    for (cur.j = 0; cur.j < cur.i; cur.j++)                 // Выбор 2-го нейрона
    {
        for (i = 0; i < op_count; i++)                      // Выбор операции
        {
            cur.op = op[i];
            sum = 0.0;
            cur.cached = false;
            curval = GetNeironVector(Neirons);

            for (int index = 0; index < Images && sum < min; index++)
            {
                square = vz[index] - curval[index];
                sum += square * square;
            }

            if (min > sum)
            {
                min = sum;
                optimal_op = cur.op;
                optimal_i = cur.i;
                optimal_j = cur.j;
            }
        }
    }

    cur.cached = false;
    cur.i = optimal_i;
    cur.j = optimal_j;
    cur.op = optimal_op;
    std::cout << "min = " << min << ", (" << Neirons << ") = (" << optimal_i << ")op(" << optimal_j << ")\n";
    Neirons++;
    return min;
}

/**
 * Комбинирование старых и новых нейронов (combine_old_new)
 *
 * Комбинирует старые нейроны (до Classes*3) с новыми.
 * Специализированная стратегия для определённых этапов обучения.
 *
 * Создаёт: 1 нейрон
 * Сложность: O(N_old * N_new * O)
 *
 * @return минимальная достигнутая ошибка
 */
float combine_old_new() {
    int     i;
    float   min = big;
    int     optimal_i = 0;
    int     optimal_j = 0;
    oper    optimal_op = op[0];
    float   square, sum;
    Neiron& cur = nei[Neirons];
    float*  curval;

    for (cur.i = 0; cur.i < Neirons - Classes * 3; cur.i++)         // Старые нейроны
    {
        for (cur.j = Neirons - Classes * 3; cur.j < Neirons; cur.j++) // Новые нейроны
        {
            for (i = 0; i < op_count; i++)
            {
                cur.cached = false;
                cur.op = op[i];
                sum = 0.0;
                curval = GetNeironVector(Neirons);

                for (int index = 0; index < Images && sum < min; index++)
                {
                    square = vz[index] - curval[index];
                    sum += square * square;
                }

                if (min > sum)
                {
                    min = sum;
                    optimal_op = cur.op;
                    optimal_i = cur.i;
                    optimal_j = cur.j;
                }
            }
        }
    }

    cur.cached = false;
    cur.i = optimal_i;
    cur.j = optimal_j;
    cur.op = optimal_op;
    std::cout << "min = " << min << ", (" << Neirons << ") = (" << optimal_i << ")op(" << optimal_j << ")\n";
    Neirons++;
    return min;
}

// ============================================================================
// Многопоточные версии функций
// ============================================================================

/**
 * Структура результата поиска для потока (exhaustive search)
 */
struct ExhaustiveSearchResult {
    float min_error;
    int optimal_i;
    int optimal_j;
    int optimal_op_index;
    bool found;

    ExhaustiveSearchResult() : min_error(big), optimal_i(0), optimal_j(0), optimal_op_index(0), found(false) {}
};

/**
 * Функция потока для параллельного полного перебора
 */
void ExhaustiveSearchThreadFunc(
    int start_i,
    int end_i,
    int /* current_neirons */,
    ExhaustiveSearchResult& result,
    std::atomic<float>* global_min)
{
    Neiron local_cur;
    std::vector<float> local_cache(Images);

    for (local_cur.i = start_i; local_cur.i < end_i; local_cur.i++)
    {
        float* i_cache = GetNeironVector(local_cur.i);

        for (local_cur.j = 0; local_cur.j < local_cur.i; local_cur.j++)
        {
            float* j_cache = GetNeironVector(local_cur.j);

            for (int op_idx = 0; op_idx < op_count; op_idx++)
            {
                local_cur.op = op[op_idx];
                (*local_cur.op)(local_cache.data(), i_cache, j_cache, Images);

                float current_global_min = global_min->load(std::memory_order_relaxed);
                float sum = 0.0f;

                for (int index = 0; index < Images && sum < current_global_min; index++)
                {
                    float square = vz[index] - local_cache[index];
                    sum += square * square;
                }

                if (result.min_error > sum)
                {
                    result.found = true;
                    result.min_error = sum;
                    result.optimal_i = local_cur.i;
                    result.optimal_j = local_cur.j;
                    result.optimal_op_index = op_idx;

                    // Обновляем глобальный минимум
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
 * Параллельный полный перебор комбинаций нейронов
 *
 * Многопоточная версия exhaustive_full_search().
 * Распределяет поиск по первому входу нейрона между потоками.
 *
 * @return минимальная достигнутая ошибка
 */
float exhaustive_full_search_parallel() {
    if (!UseMultithreading || NumThreads <= 1 || Neirons < 10) {
        return exhaustive_full_search();
    }

    // Прогреваем кэши всех существующих нейронов
    for (int n = 0; n < Neirons; n++) {
        GetNeironVector(n);
    }

    std::vector<ExhaustiveSearchResult> results(NumThreads);
    std::atomic<float> global_min(big);
    std::vector<std::thread> threads;
    threads.reserve(NumThreads);

    // Распределяем работу по потокам (по первому индексу)
    int chunk = (Neirons + NumThreads - 1) / NumThreads;

    for (int t = 0; t < NumThreads; t++) {
        int start_i = std::max(1, t * chunk);
        int end_i = std::min(Neirons, (t + 1) * chunk);

        if (start_i < end_i) {
            threads.emplace_back(ExhaustiveSearchThreadFunc,
                                 start_i, end_i, Neirons,
                                 std::ref(results[t]), &global_min);
        }
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
        Neiron& cur = nei[Neirons];
        cur.cached = false;
        cur.i = results[best_thread].optimal_i;
        cur.j = results[best_thread].optimal_j;
        cur.op = op[results[best_thread].optimal_op_index];
        std::cout << "min = " << best_min << ", (" << Neirons << ") = ("
                  << cur.i << ")op(" << cur.j << ") [parallel]\n";
        Neirons++;
        return best_min;
    }

    return big;
}

/**
 * Параллельное комбинирование с последним нейроном
 *
 * Многопоточная версия exhaustive_last_combine().
 *
 * @return минимальная достигнутая ошибка
 */
float exhaustive_last_combine_parallel() {
    if (!UseMultithreading || NumThreads <= 1 || Neirons < 10) {
        return exhaustive_last_combine();
    }

    // Прогреваем кэши
    for (int n = 0; n < Neirons; n++) {
        GetNeironVector(n);
    }

    int last_neuron = Neirons - 1;
    float* last_cache = GetNeironVector(last_neuron);

    std::vector<ExhaustiveSearchResult> results(NumThreads);
    std::atomic<float> global_min(big);
    std::vector<std::thread> threads;
    threads.reserve(NumThreads);

    // Лямбда для потока
    auto thread_func = [&](int start_j, int end_j, int thread_id) {
        std::vector<float> local_cache(Images);

        for (int j = start_j; j < end_j; j++) {
            float* j_cache = GetNeironVector(j);

            for (int op_idx = 0; op_idx < op_count; op_idx++) {
                (*op[op_idx])(local_cache.data(), last_cache, j_cache, Images);

                float current_global_min = global_min.load(std::memory_order_relaxed);
                float sum = 0.0f;

                for (int index = 0; index < Images && sum < current_global_min; index++) {
                    float square = vz[index] - local_cache[index];
                    sum += square * square;
                }

                if (results[thread_id].min_error > sum) {
                    results[thread_id].found = true;
                    results[thread_id].min_error = sum;
                    results[thread_id].optimal_i = last_neuron;
                    results[thread_id].optimal_j = j;
                    results[thread_id].optimal_op_index = op_idx;

                    float expected = global_min.load(std::memory_order_relaxed);
                    while (sum < expected) {
                        if (global_min.compare_exchange_weak(expected, sum, std::memory_order_relaxed)) {
                            break;
                        }
                    }
                }
            }
        }
    };

    int chunk = (last_neuron + NumThreads - 1) / NumThreads;

    for (int t = 0; t < NumThreads; t++) {
        int start_j = t * chunk;
        int end_j = std::min(last_neuron, (t + 1) * chunk);

        if (start_j < end_j) {
            threads.emplace_back(thread_func, start_j, end_j, t);
        }
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
        Neiron& cur = nei[Neirons];
        cur.cached = false;
        cur.i = results[best_thread].optimal_i;
        cur.j = results[best_thread].optimal_j;
        cur.op = op[results[best_thread].optimal_op_index];
        std::cout << "min = " << best_min << ", (" << Neirons << ") = ("
                  << cur.i << ")op(" << cur.j << ") [parallel]\n";
        Neirons++;
        return best_min;
    }

    return big;
}

/**
 * Параллельное комбинирование старых и новых нейронов
 *
 * Многопоточная версия combine_old_new().
 *
 * @return минимальная достигнутая ошибка
 */
float combine_old_new_parallel() {
    if (!UseMultithreading || NumThreads <= 1) {
        return combine_old_new();
    }

    int boundary = Neirons - Classes * 3;
    if (boundary <= 0) {
        return combine_old_new();
    }

    // Прогреваем кэши
    for (int n = 0; n < Neirons; n++) {
        GetNeironVector(n);
    }

    std::vector<ExhaustiveSearchResult> results(NumThreads);
    std::atomic<float> global_min(big);
    std::vector<std::thread> threads;
    threads.reserve(NumThreads);

    auto thread_func = [&](int start_i, int end_i, int thread_id) {
        std::vector<float> local_cache(Images);

        for (int i = start_i; i < end_i; i++) {
            float* i_cache = GetNeironVector(i);

            for (int j = boundary; j < Neirons; j++) {
                float* j_cache = GetNeironVector(j);

                for (int op_idx = 0; op_idx < op_count; op_idx++) {
                    (*op[op_idx])(local_cache.data(), i_cache, j_cache, Images);

                    float current_global_min = global_min.load(std::memory_order_relaxed);
                    float sum = 0.0f;

                    for (int index = 0; index < Images && sum < current_global_min; index++) {
                        float square = vz[index] - local_cache[index];
                        sum += square * square;
                    }

                    if (results[thread_id].min_error > sum) {
                        results[thread_id].found = true;
                        results[thread_id].min_error = sum;
                        results[thread_id].optimal_i = i;
                        results[thread_id].optimal_j = j;
                        results[thread_id].optimal_op_index = op_idx;

                        float expected = global_min.load(std::memory_order_relaxed);
                        while (sum < expected) {
                            if (global_min.compare_exchange_weak(expected, sum, std::memory_order_relaxed)) {
                                break;
                            }
                        }
                    }
                }
            }
        }
    };

    int chunk = (boundary + NumThreads - 1) / NumThreads;

    for (int t = 0; t < NumThreads; t++) {
        int start_i = t * chunk;
        int end_i = std::min(boundary, (t + 1) * chunk);

        if (start_i < end_i) {
            threads.emplace_back(thread_func, start_i, end_i, t);
        }
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
        Neiron& cur = nei[Neirons];
        cur.cached = false;
        cur.i = results[best_thread].optimal_i;
        cur.j = results[best_thread].optimal_j;
        cur.op = op[results[best_thread].optimal_op_index];
        std::cout << "min = " << best_min << ", (" << Neirons << ") = ("
                  << cur.i << ")op(" << cur.j << ") [parallel]\n";
        Neirons++;
        return best_min;
    }

    return big;
}

// Сохраняем обратную совместимость со старыми именами
inline float rod() { return exhaustive_full_search(); }
inline float rod2() { return exhaustive_last_combine(); }
inline float rod3() { return combine_old_new(); }
inline float rod_parallel() { return exhaustive_full_search_parallel(); }
inline float rod2_parallel() { return exhaustive_last_combine_parallel(); }
inline float rod3_parallel() { return combine_old_new_parallel(); }

#endif // EXHAUSTIVE_SEARCH_H
