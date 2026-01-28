/*
 * learning_funcs.h - Единый интерфейс для всех функций обучения
 *
 * Этот модуль предоставляет:
 * - Единую точку включения для всех функций обучения
 * - Реестр функций обучения для динамического выбора
 * - Функции для работы с последовательностями обучения
 *
 * Использование:
 * 1. Подключите только этот файл: #include "learning_funcs/learning_funcs.h"
 * 2. Используйте getLearningFunc() для получения функции по имени
 * 3. Используйте getAvailableLearningFuncs() для получения списка функций
 */

#ifndef LEARNING_FUNCS_H
#define LEARNING_FUNCS_H

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <iostream>

// Подключаем все модули функций обучения
// Примечание: эти заголовки зависят от определений в neuron_generation.h,
// поэтому должны включаться после него
#include "exhaustive_search.h"
#include "random_search.h"
#include "triplet_search.h"

// ============================================================================
// Реестр функций обучения
// ============================================================================

/**
 * Информация о функции обучения
 */
struct LearningFunctionInfo {
    std::string name;              // Имя функции для использования в конфиге
    std::string description;       // Описание функции на русском
    std::string old_name;          // Старое имя для обратной совместимости
    LearningFunc func;             // Указатель на функцию
    bool is_parallel;              // Является ли функция параллельной
    int neurons_created;           // Сколько нейронов создаёт функция (0 = переменное)
};

/**
 * Получение списка всех доступных функций обучения
 *
 * @return вектор информации о функциях
 */
inline std::vector<LearningFunctionInfo> getAvailableLearningFuncs() {
    return {
        // Полный перебор (последовательные)
        {
            "exhaustive_full",
            "Полный перебор всех пар нейронов и операций",
            "rod",
            exhaustive_full_search,
            false,
            1
        },
        {
            "exhaustive_last",
            "Комбинирование с последним созданным нейроном",
            "rod2",
            exhaustive_last_combine,
            false,
            1
        },
        {
            "combine_old_new",
            "Комбинирование старых нейронов с новыми",
            "rod3",
            combine_old_new,
            false,
            1
        },

        // Полный перебор (параллельные)
        {
            "exhaustive_full_parallel",
            "Параллельный полный перебор всех пар нейронов",
            "rod_parallel",
            exhaustive_full_search_parallel,
            true,
            1
        },
        {
            "exhaustive_last_parallel",
            "Параллельное комбинирование с последним нейроном",
            "rod2_parallel",
            exhaustive_last_combine_parallel,
            true,
            1
        },
        {
            "combine_old_new_parallel",
            "Параллельное комбинирование старых с новыми",
            "rod3_parallel",
            combine_old_new_parallel,
            true,
            1
        },

        // Случайный поиск (последовательные)
        {
            "random_single",
            "Генерация одного случайного нейрона",
            "rndrod",
            random_neurons,
            false,
            1
        },
        {
            "random_from_inputs",
            "Случайная генерация на основе входов",
            "rndrod0",
            random_from_inputs,
            false,
            1
        },
        {
            "random_pair_opt",
            "Оптимизированная генерация пары нейронов",
            "rndrod2",
            random_pair_optimized,
            false,
            2
        },
        {
            "random_pair_ext",
            "Расширенная генерация пары нейронов",
            "rndrod3",
            random_pair_extended,
            false,
            2
        },

        // Случайный поиск (параллельные)
        {
            "random_pair_opt_parallel",
            "Параллельная оптимизированная генерация пары",
            "rndrod2_parallel",
            random_pair_optimized_parallel,
            true,
            2
        },
        {
            "random_pair_ext_parallel",
            "Параллельная расширенная генерация пары",
            "rndrod3_parallel",
            random_pair_extended_parallel,
            true,
            2
        },

        // Генерация тройки нейронов (последовательная)
        {
            "triplet",
            "Генерация тройки связанных нейронов (основной метод)",
            "rndrod4",
            triplet_random,
            false,
            3
        },

        // Генерация тройки нейронов (параллельная) - метод по умолчанию
        {
            "triplet_parallel",
            "Параллельная генерация тройки нейронов (метод по умолчанию)",
            "rndrod4_parallel",
            triplet_random_parallel,
            true,
            3
        }
    };
}

/**
 * Получение функции обучения по имени
 *
 * Поддерживает как новые, так и старые имена для обратной совместимости.
 *
 * @param name - имя функции (новое или старое)
 * @return указатель на функцию или nullptr если не найдена
 */
inline LearningFunc getLearningFunc(const std::string& name) {
    auto funcs = getAvailableLearningFuncs();
    for (const auto& info : funcs) {
        if (info.name == name || info.old_name == name) {
            return info.func;
        }
    }
    return nullptr;
}

/**
 * Получение информации о функции обучения по имени
 *
 * @param name - имя функции
 * @param info - выходной параметр с информацией
 * @return true если функция найдена
 */
inline bool getLearningFuncInfo(const std::string& name, LearningFunctionInfo& info) {
    auto funcs = getAvailableLearningFuncs();
    for (const auto& f : funcs) {
        if (f.name == name || f.old_name == name) {
            info = f;
            return true;
        }
    }
    return false;
}

/**
 * Проверка существования функции обучения
 *
 * @param name - имя функции
 * @return true если функция существует
 */
inline bool learningFuncExists(const std::string& name) {
    auto funcs = getAvailableLearningFuncs();
    for (const auto& info : funcs) {
        if (info.name == name || info.old_name == name) {
            return true;
        }
    }
    return false;
}

/**
 * Вывод списка доступных функций обучения
 */
inline void printAvailableLearningFuncs() {
    std::cout << "\nДоступные функции обучения:" << std::endl;
    std::cout << "==========================" << std::endl;

    auto funcs = getAvailableLearningFuncs();

    std::cout << "\nПолный перебор (детерминированные):" << std::endl;
    for (const auto& f : funcs) {
        if (f.name.find("exhaustive") != std::string::npos || f.name.find("combine") != std::string::npos) {
            std::cout << "  " << f.name;
            if (f.is_parallel) std::cout << " [parallel]";
            std::cout << " - " << f.description;
            std::cout << " (создаёт " << f.neurons_created << " нейрон(ов))" << std::endl;
        }
    }

    std::cout << "\nСлучайный поиск:" << std::endl;
    for (const auto& f : funcs) {
        if (f.name.find("random") != std::string::npos) {
            std::cout << "  " << f.name;
            if (f.is_parallel) std::cout << " [parallel]";
            std::cout << " - " << f.description;
            std::cout << " (создаёт " << f.neurons_created << " нейрон(ов))" << std::endl;
        }
    }

    std::cout << "\nГенерация тройки (рекомендуемые):" << std::endl;
    for (const auto& f : funcs) {
        if (f.name.find("triplet") != std::string::npos) {
            std::cout << "  " << f.name;
            if (f.is_parallel) std::cout << " [parallel]";
            std::cout << " - " << f.description;
            std::cout << " (создаёт " << f.neurons_created << " нейрон(ов))" << std::endl;
        }
    }

    std::cout << "\nПо умолчанию: triplet_parallel" << std::endl;
}

/**
 * Получение функции обучения по умолчанию
 *
 * @return указатель на функцию triplet_random_parallel
 */
inline LearningFunc getDefaultLearningFunc() {
    return triplet_random_parallel;
}

/**
 * Получение имени функции по умолчанию
 *
 * @return "triplet_parallel"
 */
inline std::string getDefaultLearningFuncName() {
    return "triplet_parallel";
}

#endif // LEARNING_FUNCS_H
