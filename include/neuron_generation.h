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
 *
 * Начиная с версии 1.1, функции обучения вынесены в отдельные файлы
 * в директории learning_funcs/ для лучшей модульности:
 * - exhaustive_search.h - функции полного перебора
 * - random_search.h - функции случайного поиска
 * - triplet_search.h - функции генерации тройки нейронов
 * - learning_funcs.h - единый интерфейс и реестр функций
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
// Подключение модульных функций обучения
// ============================================================================

// Подключаем все функции обучения через единый заголовок
#include "learning_funcs/learning_funcs.h"

#endif // NEURON_GENERATION_H
