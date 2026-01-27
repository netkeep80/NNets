/*
 * neuron_generation.h - Функции генерации и обучения нейронов
 *
 * Этот модуль содержит функции для:
 * - Вычисления выходных значений нейронов
 * - Генерации новых нейронов (рождение нейронов)
 * - Различные стратегии поиска оптимальных нейронов
 *
 * Примечание: Этот файл предназначен для включения в main.cpp после
 * определения всех необходимых типов и переменных.
 */

#ifndef NEURON_GENERATION_H
#define NEURON_GENERATION_H

// ============================================================================
// Функции инициализации и работы с кэшем
// ============================================================================

/**
 * Инициализация массива нейронов
 *
 * Выделяет память для нейронов и их кэшей.
 * Должна вызываться после загрузки конфигурации.
 */
void initNeurons() {
    nei.resize(MAX_NEURONS);
    for (int n = 0; n < MAX_NEURONS; n++) {
        nei[n].c.resize(Images);
        nei[n].cached = false;
        nei[n].val_cached = false;
    }
}

/**
 * Очистка кэша одиночных значений
 *
 * Необходимо вызывать перед каждым новым вычислением в режиме инференса.
 *
 * @param n - массив нейронов
 * @param size - размер массива
 */
void clear_val_cache(vector<Neiron>& n, const int size) {
    int limit = (size < (int)n.size()) ? size : (int)n.size();
    for (int i = 0; i < limit; i++)
        n[i].val_cached = false;
}

// ============================================================================
// Функции вычисления значений нейронов
// ============================================================================

/**
 * Расчёт вектора значений для i-го нейрона
 *
 * Вычисляет выходные значения нейрона для всех образов одновременно.
 * Использует кэширование для избежания повторных вычислений.
 *
 * @param i - номер нейрона
 * @return указатель на массив значений для всех образов
 */
float* __fastcall GetNeironVector(const int i) {
    Neiron& current = nei[i];
    if (!current.cached)
    {
        current.cached = true;
        if (i < Receptors)
        {
            // Входные нейроны: берём значения из образов
            for (int im = 0; im < Images; im++)
                current.c[im] = vx[im][i];
        }
        else if (i < Inputs)
        {
            // Базисные нейроны: постоянные значения
            for (int im = 0; im < Images; im++)
                current.c[im] = NetInput[i];
        }
        else
        {
            // Вычисляемые нейроны: применяем операцию к входам
            float* icache = GetNeironVector(current.i);
            float* jcache = GetNeironVector(current.j);
            (*current.op)(current.c.data(), icache, jcache, Images);
        }
    }

    return current.c.data();
}

/**
 * Расчёт одиночного значения для i-го нейрона
 *
 * Вычисляет выходное значение нейрона для текущего входа (NetInput).
 * Используется в режиме инференса.
 *
 * @param i - номер нейрона
 * @return выходное значение нейрона
 */
float __fastcall GetNeironVal(const int i) {
    if (i < Inputs)
    {
        // Входные и базисные нейроны: возвращаем напрямую
        return NetInput[i];
    }
    else
    {
        Neiron& current = nei[i];
        if (current.val_cached)
            return current.val;
        else
        {
            // Рекурсивно вычисляем значение
            float ival = GetNeironVal(current.i);
            float jval = GetNeironVal(current.j);
            (*current.op)(&current.val, &ival, &jval, 1);
            current.val_cached = true;
            return current.val;
        }
    }
}

// ============================================================================
// Функции генерации (рождения) нейронов
// ============================================================================

/**
 * Полный перебор комбинаций нейронов
 *
 * Перебирает все комбинации пар нейронов и операций,
 * выбирая оптимальную комбинацию с минимальной ошибкой.
 * Медленный, но гарантирует нахождение лучшего нейрона.
 *
 * @return минимальная достигнутая ошибка
 */
float rod() {
    // Полный перебор всех комбинаций нейронов
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
    cout << "min = " << min << ", (" << Neirons << ") = (" << optimal_i << ")op(" << optimal_j << ")\n";
    Neirons++;
    return min;
}

/**
 * Комбинирование с последним нейроном
 *
 * Фиксирует первый вход как последний созданный нейрон
 * и перебирает только второй вход и операцию.
 * Быстрее, чем rod(), но менее гибкий.
 *
 * @return минимальная достигнутая ошибка
 */
float rod2() {
    // Комбинирование с последним нейроном
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
    cout << "min = " << min << ", (" << Neirons << ") = (" << optimal_i << ")op(" << optimal_j << ")\n";
    Neirons++;
    return min;
}

/**
 * Комбинирование старых и новых нейронов
 *
 * Комбинирует старые нейроны (до Classes*3) с новыми.
 * Специализированная стратегия для определённых этапов обучения.
 *
 * @return минимальная достигнутая ошибка
 */
float rod3() {
    // Комбинирование старых нейронов с новыми
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
    cout << "min = " << min << ", (" << Neirons << ") = (" << optimal_i << ")op(" << optimal_j << ")\n";
    Neirons++;
    return min;
}

/**
 * Случайная генерация нейронов
 *
 * Создаёт заданное количество случайных нейронов.
 * Используется для расширения пространства поиска.
 *
 * @param count - количество создаваемых нейронов
 */
void rndrod(unsigned count) {
    // Генерация случайных нейронов
    do
    {
        nei[Neirons].cached = false;
        nei[Neirons].i = rand() % (Neirons);
        nei[Neirons].j = rand() % (Neirons);
        nei[Neirons].op = op[rand() % op_count];
        cout << "(" << Neirons << ") = (" << nei[Neirons].i << ")op(" << nei[Neirons].j << ")\n";
        Neirons++;
    } while (--count > 0);
}

/**
 * Случайная генерация на основе входов
 *
 * Создаёт случайные нейроны, комбинируя только входы сети.
 *
 * @param count - количество создаваемых нейронов
 */
void rndrod0(unsigned count) {
    // Генерация случайных нейронов на основе входов
    do
    {
        nei[Neirons].cached = false;
        nei[Neirons].i = rand() % (Inputs);
        nei[Neirons].j = rand() % (Receptors);
        nei[Neirons].op = op[rand() % op_count];
        cout << "(" << Neirons << ") = (" << nei[Neirons].i << ")op(" << nei[Neirons].j << ")\n";
        Neirons++;
    } while (--count > 0);
}

/**
 * Оптимизированная случайная генерация (2 нейрона)
 *
 * Создаёт пару нейронов с оптимизированными параметрами.
 * Ищет лучшую комбинацию среди случайных вариантов.
 *
 * @return минимальная достигнутая ошибка
 */
float rndrod2() {
    // Оптимизированная генерация пары нейронов
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
        Neiron_A.j = rand() % (Neirons - rndrod_iter);
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
    cout << "min = " << min << ", (" << Neirons + 1 << ") = ((" << r[1] << ")op(" << r[2] << "))op(" << r[4] << ")\n";
    Neirons += 2;
    return min;
}

/**
 * Расширенная случайная генерация (2 нейрона)
 *
 * Аналогична rndrod2(), но с большим пространством поиска.
 *
 * @return минимальная достигнутая ошибка
 */
float rndrod3() {
    // Расширенная случайная генерация пары нейронов
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
    cout << "min = " << min << ", (" << Neirons + 1 << ") = ((" << r[1] << ")op(" << r[2] << "))op(" << r[4] << ")\n";
    Neirons += 2;
    return min;
}

/**
 * Генерация тройки нейронов (основной метод обучения)
 *
 * Создаёт три связанных нейрона: A, B и C.
 * C объединяет A и B, обеспечивая более сложные функции.
 * Это основной метод обучения в текущей версии.
 *
 * @return минимальная достигнутая ошибка, или big если не найдено
 */
float rndrod4() {
    // Генерация тройки связанных нейронов - основной метод обучения
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
                // vz[index] содержит ожидаемое значение для образа index
                // (1.0 если образ принадлежит текущему классу, 0.0 иначе)
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
// Многопоточные версии функций генерации нейронов
// ============================================================================

/**
 * Структура для хранения результата поиска в потоке
 */
struct ThreadSearchResult {
    float min_error;
    Neiron optimal_A;
    Neiron optimal_B;
    Neiron optimal_C;
    bool found;

    ThreadSearchResult() : min_error(big), found(false) {}
};

/**
 * Вычисление вектора значений для заданного нейрона
 * (с использованием кэша для уже существующих нейронов)
 *
 * @param neuron_i - первый вход нейрона
 * @param neuron_j - второй вход нейрона
 * @param neuron_op - операция нейрона
 * @param result - буфер для результата
 */
void ComputeNeironVector_Direct(
    int neuron_i,
    int neuron_j,
    oper neuron_op,
    vector<float>& result)
{
    result.resize(Images);
    float* i_cache = GetNeironVector(neuron_i);
    float* j_cache = GetNeironVector(neuron_j);
    (*neuron_op)(result.data(), i_cache, j_cache, Images);
}

/**
 * Функция потока для параллельного поиска оптимальных нейронов
 *
 * Каждый поток получает своё подмножество случайных комбинаций
 * параметров нейронов для поиска оптимального решения.
 *
 * @param thread_id - ID потока
 * @param iterations_per_thread - количество итераций для этого потока
 * @param current_neirons - текущее количество нейронов
 * @param seed - начальное значение для генератора случайных чисел
 * @param result - результат поиска (выходной параметр)
 * @param global_min - указатель на атомарную переменную с глобальным минимумом
 */
void SearchThreadFunc(
    int thread_id,
    int iterations_per_thread,
    int current_neirons,
    unsigned int seed,
    ThreadSearchResult& result,
    std::atomic<float>* global_min)
{
    // Локальный генератор случайных чисел для потока
    // Используем линейный конгруэнтный генератор для независимости от глобального состояния
    unsigned int local_seed = seed + thread_id * 1099087573u;
    auto local_rand = [&local_seed]() -> int {
        local_seed = local_seed * 1103515245u + 12345u;
        return (int)((local_seed >> 16) & 0x7FFF);
    };

    Neiron local_A, local_B, local_C;
    vector<float> A_Vector(Images), B_Vector(Images), C_Vector(Images);

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
 * Многопоточная генерация тройки нейронов
 *
 * Параллельная версия rndrod4(), распределяющая поиск
 * между несколькими потоками. Каждый поток исследует
 * случайные комбинации параметров нейронов независимо.
 *
 * Ключевая идея: общее количество итераций увеличивается
 * пропорционально количеству потоков, что обеспечивает
 * эквивалентное качество поиска при большей скорости.
 *
 * @return минимальная достигнутая ошибка, или big если не найдено
 */
float rndrod4_parallel() {
    // Если многопоточность отключена или только 1 поток, используем последовательную версию
    if (!UseMultithreading || NumThreads <= 1) {
        return rndrod4();
    }

    // Общее количество итераций для параллельного поиска
    // Каждый поток получает count_max / NumThreads итераций
    // Но общее количество итераций (сумма по всем потокам) равно count_max
    int count_max = Neirons * Receptors * 4;
    // Каждый поток выполняет count_max / NumThreads итераций
    // Общее количество итераций (сумма по всем потокам) равно count_max,
    // что эквивалентно однопоточной версии
    int iterations_per_thread = (count_max + NumThreads - 1) / NumThreads;

    // Минимум 50 итераций на поток для стабильного качества поиска
    // При очень малом количестве итераций качество поиска может страдать
    // из-за недостаточного исследования пространства
    if (iterations_per_thread < 50) {
        iterations_per_thread = 50;
    }

    // Создаём результаты для каждого потока
    vector<ThreadSearchResult> results(NumThreads);

    // Атомарная переменная для синхронизации глобального минимума
    std::atomic<float> global_min(big);

    // Получаем начальное значение для генераторов случайных чисел
    unsigned int base_seed = (unsigned int)time(nullptr);

    // Запускаем потоки
    vector<std::thread> threads;
    threads.reserve(NumThreads);

    for (int t = 0; t < NumThreads; t++) {
        threads.emplace_back(SearchThreadFunc,
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

#endif // NEURON_GENERATION_H
