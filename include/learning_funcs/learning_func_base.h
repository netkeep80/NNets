/*
 * learning_func_base.h - Базовые определения для функций обучения
 *
 * Этот модуль содержит:
 * - Общие типы и структуры для функций генерации нейронов
 * - Объявление глобальных переменных, используемых функциями обучения
 * - Вспомогательные функции для работы с нейронами
 *
 * Примечание: Этот файл должен включаться первым перед другими
 * файлами функций обучения.
 */

#ifndef LEARNING_FUNC_BASE_H
#define LEARNING_FUNC_BASE_H

#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <algorithm>
#include <iostream>

// Максимальное значение ошибки (используется для инициализации)
extern const float big;

// Количество операций
extern const int op_count;

// Массив операций
extern oper op[];

// Количество потоков и флаг многопоточности
extern int NumThreads;
extern bool UseMultithreading;

// Глобальные переменные сети
extern int Neirons;
extern int Images;
extern int Inputs;
extern int Receptors;
extern int Classes;
extern std::vector<Neiron> nei;
extern std::vector<float> vz;
extern std::vector<std::vector<float>> vx;
extern std::vector<float> NetInput;

// Константы итераций
extern const int rod2_iter;
extern const int rndrod_iter;
extern const int rndrod2_iter;
extern const int MAX_NEURONS;

// ============================================================================
// Типы для регистрации функций обучения
// ============================================================================

/**
 * Тип функции обучения
 *
 * @return ошибка обучения (чем меньше - тем лучше)
 */
typedef float (*LearningFunc)();

/**
 * Структура описания функции обучения
 */
struct LearningFuncInfo {
    std::string name;        // Имя функции для использования в конфиге
    std::string description; // Описание функции
    LearningFunc func;       // Указатель на функцию
    bool is_parallel;        // Является ли функция параллельной
};

// ============================================================================
// Прототипы функций из neuron_generation.h
// ============================================================================

// Функция получения вектора значений нейрона (для всех образов)
float* __fastcall GetNeironVector(const int i);

// Функция получения одиночного значения нейрона
float __fastcall GetNeironVal(const int i);

// Функция очистки кэша значений
void clear_val_cache(std::vector<Neiron>& n, const int size);

// Функция инициализации нейронов
void initNeurons();

#endif // LEARNING_FUNC_BASE_H
