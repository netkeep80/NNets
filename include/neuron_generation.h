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

#endif // NEURON_GENERATION_H
