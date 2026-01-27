/*
 * NNets - Самообучающаяся нейронная сеть с саморождающимися структурами
 *
 * Основные принципы работы:
 * - При создании сети задаётся фиксированное количество входов (рецепторов)
 * - Задаётся максимальное количество выходов (классов)
 * - При обучении каждый новый образ, если он не может однозначно
 *   классифицироваться сетью, приводит к созданию новых нейронов
 * - Сеть автоматически генерирует оптимальную структуру для классификации
 */

#include <iostream>
#include <fstream>
#include <strstream>
#include <math.h>
#include <string.h>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <numeric>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

// ============================================================================
// Структура данных для обучающего образа
// ============================================================================

struct Image
{
	string word;  // Строковое представление образа
	int id;       // Идентификатор класса
};

// ============================================================================
// Глобальные переменные конфигурации
// ============================================================================

int Classes = 4;                                  // Количество классов
vector<string> classes;                           // Имена классов
vector<Image> const_words;                        // Обучающие образы

// ============================================================================
// Константы и параметры обучения
// ============================================================================

// Параметры многопоточности
int NumThreads = 0;                               // Количество потоков (0 = авто)
bool UseMultithreading = true;                    // Флаг использования многопоточности

const int rod2_iter = 2;                          // Итерации метода rod2
const int rndrod_iter = 10;                       // Итерации случайного поиска
const int rndrod2_iter = rndrod_iter;             // Итерации оптимизированного поиска
const float base[] = {                            // Базисные значения для входов
	0.125,
	0.25,
	0.5,
	1.0,
	2.0,
	4.0,
	8.0,
	-0.125,
	-0.25,
	-0.5,
	-1.0,
	-2.0,
	-4.0,
	-8.0,
};
const float big = 1000000000000000000.f;           // Большое число для инициализации
const int max_num = 256;                          // Количество состояний входа
int Images = 0;                                   // Количество обучающих образов
int Receptors = 20;                               // Количество входов сети
const int StringSize = 256;                       // Максимальный размер строки
const int base_size = sizeof(base) / sizeof(float);
int Inputs = 0;                                   // Общее количество входов (Receptors + base_size)
int Neirons = 0;                                  // Количество созданных нейронов
vector<float> NetInput;                           // Входные значения сети
vector<vector<float>> vx;                         // Входные значения для образов
vector<float> vz;                                 // Ожидаемые выходные значения
vector<int> NetOutput;                            // Выходные нейроны для классов
char InputStr[StringSize], word_buf[StringSize];  // Буферы для ввода

// ============================================================================
// Операции нейронов - элементарные математические действия
// ============================================================================

typedef void(__fastcall *oper)(float*, const float*, const float*, const int);

// Сумма
void __fastcall op_1(float* r, const float* z1, const float* z2, const int size) {
	for (int i = size; i > 0; i--, r++, z1++, z2++) *r = *z1 + *z2;
}

// Разность (z1 - z2)
void __fastcall op_2(float* r, const float* z1, const float* z2, const int size) {
	for (int i = size; i > 0; i--, r++, z1++, z2++) *r = *z1 - *z2;
}

// Разность (z2 - z1)
void __fastcall op_3(float* r, const float* z1, const float* z2, const int size) {
	for (int i = size; i > 0; i--, r++, z1++, z2++) *r = *z2 - *z1;
}

// Произведение
void __fastcall op_4(float* r, const float* z1, const float* z2, const int size) {
	for (int i = size; i > 0; i--, r++, z1++, z2++) *r = *z1 * *z2;
}

// Деление (z1 / z2)
void __fastcall op_5(float* r, const float* z1, const float* z2, const int size) {
	for (int i = size; i > 0; i--, r++, z1++, z2++)
		if (*z2 != 0.0) *r = *z1 / *z2; else *r = big;
}

// Деление (z2 / z1)
void __fastcall op_6(float* r, const float* z1, const float* z2, const int size) {
	for (int i = size; i > 0; i--, r++, z1++, z2++)
		if (*z1 != 0.0) *r = *z2 / *z1; else *r = big;
}

// Квадрат z2 + z1
void __fastcall op_7(float* r, const float* z1, const float* z2, const int size) {
	for (int i = size; i > 0; i--, r++, z1++, z2++) *r = *z2 * *z2 + *z1;
}

// Квадрат z1 + z2
void __fastcall op_8(float* r, const float* z1, const float* z2, const int size) {
	for (int i = size; i > 0; i--, r++, z1++, z2++) *r = *z1 * *z1 + *z2;
}

// Квадрат z2 - z1
void __fastcall op_9(float* r, const float* z1, const float* z2, const int size) {
	for (int i = size; i > 0; i--, r++, z1++, z2++) *r = *z2 * *z2 - *z1;
}

// Квадрат z1 - z2
void __fastcall op_10(float* r, const float* z1, const float* z2, const int size) {
	for (int i = size; i > 0; i--, r++, z1++, z2++) *r = *z1 * *z1 - *z2;
}

// Параллельное соединение
void __fastcall op_11(float* r, const float* z1, const float* z2, const int size) {
	for (int i = size; i > 0; i--, r++, z1++, z2++) *r = *z1 * *z2 / (*z1 + *z2);
}

// Массив доступных операций (используются только первые 4 для скорости)
oper op[] = {
	op_1,   // Сумма
	op_2,   // Разность
	op_3,   // Обратная разность
	op_4,   // Произведение
};
const int op_count = sizeof(op) / sizeof(oper);

// Получение индекса операции по указателю (для сериализации)
int getOpIndex(oper operation) {
	for (int i = 0; i < op_count; i++) {
		if (op[i] == operation) return i;
	}
	return 0;  // По умолчанию первая операция
}

// ============================================================================
// Класс нейрона
// ============================================================================

class Neiron
{
public:
	int i;                                        // Номер первого входного нейрона
	int j;                                        // Номер второго входного нейрона
	oper op;                                      // Операция нейрона
	vector<float> c;                              // Кэш значений для образов
	bool cached;                                  // Флаг валидности кэша образов
	float val;                                    // Кэш одиночного значения
	bool val_cached;                              // Флаг валидности одиночного значения

	Neiron() : i(0), j(0), op(nullptr), cached(false), val(0), val_cached(false) {}
};

vector<Neiron> nei;                               // Массив нейронов
const int MAX_NEURONS = 64000;                    // Максимальное количество нейронов

// ============================================================================
// Подключение модулей
// ============================================================================

#include "json_io.h"
#include "neuron_generation.h"

// ============================================================================
// Вспомогательные функции
// ============================================================================

/**
 * Вывод справки по использованию программы
 */
void printUsage(const char* programName) {
	cout << "Usage: " << programName << " [options]" << endl;
	cout << endl;
	cout << "MODES:" << endl;
	cout << "  Training mode (default): Train network and optionally save to file" << endl;
	cout << "  Inference mode: Load trained network and classify inputs" << endl;
	cout << endl;
	cout << "TRAINING OPTIONS:" << endl;
	cout << "  -c, --config <file>  Load training configuration from JSON file" << endl;
	cout << "  -s, --save <file>    Save trained network to JSON file after training" << endl;
	cout << "  -t, --test           Run automated test after training (no interactive mode)" << endl;
	cout << "  -b, --benchmark      Run benchmark to measure training speed" << endl;
	cout << endl;
	cout << "INFERENCE OPTIONS:" << endl;
	cout << "  -l, --load <file>    Load trained network from JSON file (inference mode)" << endl;
	cout << "  -i, --input <text>   Classify single input text and exit (non-interactive)" << endl;
	cout << endl;
	cout << "PERFORMANCE OPTIONS:" << endl;
	cout << "  -j, --threads <n>    Number of threads to use (0 = auto, default)" << endl;
	cout << "  --single-thread      Disable multithreading (use single thread)" << endl;
	cout << endl;
	cout << "GENERAL OPTIONS:" << endl;
	cout << "  -h, --help           Show this help message" << endl;
	cout << endl;
	cout << "EXAMPLES:" << endl;
	cout << "  " << programName << " -c configs/default.json -s model.json  # Train and save" << endl;
	cout << "  " << programName << " -l model.json                          # Interactive inference" << endl;
	cout << "  " << programName << " -l model.json -i \"time\"                # Single classification" << endl;
	cout << endl;
	cout << "JSON config format (training):" << endl;
	cout << "  {" << endl;
	cout << "    \"receptors\": 20," << endl;
	cout << "    \"classes\": [" << endl;
	cout << "      { \"id\": 0, \"word\": \"\" }," << endl;
	cout << "      { \"id\": 1, \"word\": \"time\" }" << endl;
	cout << "    ]," << endl;
	cout << "    \"generate_shifts\": true" << endl;
	cout << "  }" << endl;
}

/**
 * Чтение строки с клавиатуры
 */
int readkeyboard(char* str)
{
	char one;
	int i = 0;

	do {
		cin.read(&one, 1);
		if (one == '\n')
		{
			str[i] = 0;
			return i;
		}
		else
		{
			str[i] = one;
		}
	} while (++i < StringSize - 1);

	str[i] = 0;
	return i;
}

/**
 * Сравнение строки с буфером
 */
bool cmp(char* str)
{
	return strcmp(str, word_buf) == 0;
}

/**
 * Сумма элементов массива
 */
float sum(const float* ar, const int size)
{
	float res = 0;
	for (int i = 0; i < size; i++) res += ar[i];
	return res;
}

/**
 * Классификация входного текста
 *
 * Устанавливает входные значения сети из текста и выводит результат классификации.
 *
 * @param inputText - входной текст для классификации
 * @param verbose - выводить ли результаты на экран
 */
void classifyInput(const string& inputText, bool verbose = true) {
	// Устанавливаем входные значения из текста
	for (int d = 0; d < Receptors; d++) {
		if (d < (int)inputText.length() && inputText[d] != 0) {
			NetInput[d] = float((unsigned char)inputText[d]) / float(max_num);
		} else {
			NetInput[d] = float((unsigned char)' ') / float(max_num);
		}
	}

	// Очищаем кэш значений
	clear_val_cache(nei, MAX_NEURONS);

	// Вычисляем и выводим результаты для каждого класса
	if (verbose) {
		for (int out = 0; out < Classes; out++) {
			float z1 = GetNeironVal(NetOutput[out]) * 100.0f;
			// Обработка NaN и бесконечных значений
			if (!std::isfinite(z1)) z1 = 0.0f;
			if (z1 < 0.0f) z1 = 0.0f;
			if (z1 > 100.0f) z1 = 100.0f;
			cout << long(z1) << "%" << " - " << classes[out] << endl;
		}
	}
}

// ============================================================================
// Главная функция
// ============================================================================

int main(int argc, char* argv[])
{
	// Разбор аргументов командной строки
	string configPath = "";
	string savePath = "";
	string loadPath = "";
	string inputText = "";
	bool testMode = false;
	bool benchmarkMode = false;
	bool inferenceMode = false;

	for (int i = 1; i < argc; i++) {
		string arg = argv[i];
		if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
			configPath = argv[++i];
		} else if ((arg == "-s" || arg == "--save") && i + 1 < argc) {
			savePath = argv[++i];
		} else if ((arg == "-l" || arg == "--load") && i + 1 < argc) {
			loadPath = argv[++i];
			inferenceMode = true;
		} else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
			inputText = argv[++i];
		} else if (arg == "-t" || arg == "--test") {
			testMode = true;
		} else if (arg == "-b" || arg == "--benchmark") {
			benchmarkMode = true;
		} else if ((arg == "-j" || arg == "--threads") && i + 1 < argc) {
			NumThreads = atoi(argv[++i]);
		} else if (arg == "--single-thread") {
			UseMultithreading = false;
		} else if (arg == "-h" || arg == "--help") {
			printUsage(argv[0]);
			return 0;
		}
	}

	// ===== РЕЖИМ ИНФЕРЕНСА =====
	// Загружаем обученную сеть и переходим к классификации
	if (inferenceMode) {
		if (!loadNetwork(loadPath)) {
			return 1;
		}

		// Если задан входной текст - классифицируем и выходим
		if (!inputText.empty()) {
			cout << "\nClassifying: \"" << inputText << "\"" << endl;
			classifyInput(inputText);
			return 0;
		}

		// Иначе - интерактивный режим инференса
		cout << "\nEntering interactive inference mode..." << endl;
		cout << "Enter text to classify (or 'Q' to quit):" << endl;

		do {
			cout << "\ninput word:";
			if (InputStr[0] == 0) {
				readkeyboard(InputStr);
			}

			memset(word_buf, 0, StringSize);
			strcpy_s(word_buf, InputStr);
			memset(InputStr, 0, StringSize);

			if (cmp("Q") || cmp("q")) return 0;

			classifyInput(word_buf);

		} while (true);
	}

	// ===== РЕЖИМ ОБУЧЕНИЯ =====

	// Загружаем конфигурацию или используем значения по умолчанию
	if (!configPath.empty()) {
		if (!loadConfig(configPath, Receptors)) {
			return 1;
		}
	} else {
		initDefaultConfig(Receptors);
	}

	cout << "Random seed: " << rand() << endl;

	// Настройка многопоточности
	if (UseMultithreading) {
		if (NumThreads <= 0) {
			NumThreads = std::thread::hardware_concurrency();
			if (NumThreads == 0) NumThreads = 4;  // Значение по умолчанию
		}
		cout << "Multithreading: enabled, " << NumThreads << " threads" << endl;
	} else {
		NumThreads = 1;
		cout << "Multithreading: disabled (single-threaded mode)" << endl;
	}

	// Вычисляем производные значения после загрузки конфигурации
	Images = const_words.size();
	Inputs = Receptors + base_size;
	Neirons = Inputs;

	// Выделяем динамические массивы
	NetInput.resize(Inputs);
	vx.resize(Images);
	for (int i = 0; i < Images; i++) {
		vx[i].resize(Receptors);
	}
	vz.resize(Classes);
	NetOutput.resize(Classes);

	// Инициализируем массив нейронов
	initNeurons();

	// Задаём базисные значения
	for (int i = 0; i < base_size; i++)
		NetInput[i + Receptors] = base[i];

	// Генерируем образы из слов
	for (int index = 0; index < Images; index++)
	{
		memset(word_buf, 0, StringSize);
		strcpy_s(word_buf, const_words[index].word.c_str());
		cout << "img:" << const_words[index].word << endl;

		for (int d = 0; d < Receptors; d++)
		{
			if (word_buf[d] == 0)
			{
				// Заполняем оставшиеся позиции пробелами
				for (; d < Receptors; d++)
					vx[index][d] = float((unsigned char)' ') / float(max_num);

				break;
			}
			else
			{
				vx[index][d] = float((unsigned char)word_buf[d]) / float(max_num);
			}
		}
	}

	int classIndex = 0;
	// Отслеживаем ошибку для каждого класса
	vector<float> class_er(Classes, big);
	float er = .01f;  // Допустимая ошибка

	// Засекаем время обучения
	auto trainingStartTime = chrono::high_resolution_clock::now();
	int trainingIterations = 0;

	// Цикл обучения
	do
	{
		trainingIterations++;
		cout << "train class:" << classes[classIndex] << " (id=" << classIndex << ")";

		// Задаём ожидаемый вектор выходов:
		// 1.0 для образов текущего класса, 0.0 для остальных
		vz.resize(Images);
		for (int img = 0; img < Images; img++)
		{
			if (const_words[img].id == classIndex)
				vz[img] = 1.0;  // Образ принадлежит обучаемому классу
			else
				vz[img] = 0.0;  // Образ НЕ принадлежит классу
		}

		// Обучаем распознавание текущего класса
		if (class_er[classIndex] > er)
		{
			// Используем метод rndrod4_parallel - многопоточная генерация тройки нейронов
			class_er[classIndex] = rndrod4_parallel();

			// Сохраняем выходной нейрон для класса
			NetOutput[classIndex] = Neirons - 1;
		}

		cout << ", n" << NetOutput[classIndex] << " = " << class_er[classIndex] << endl;

		if (++classIndex >= Classes)  // Переходим к следующему классу по кругу
			classIndex = 0;

	} while (sum(class_er.data(), Classes) > Classes * er);

	// Конец обучения
	auto trainingEndTime = chrono::high_resolution_clock::now();
	auto trainingDuration = chrono::duration_cast<chrono::milliseconds>(trainingEndTime - trainingStartTime);

	cout << "\nTraining completed!" << endl;
	cout << "Final errors per class:" << endl;
	for (int c = 0; c < Classes; c++) {
		cout << "  Class " << c << " (" << classes[c] << "): error = " << class_er[c] << endl;
	}

	// Сохраняем сеть если указан путь
	if (!savePath.empty()) {
		if (!saveNetwork(savePath)) {
			cerr << "Warning: Failed to save network to " << savePath << endl;
		}
	}

	// Режим бенчмарка: выводим метрики скорости обучения
	if (benchmarkMode) {
		cout << "\n=== Training Speed Benchmark Results ===" << endl;
		cout << "Configuration:" << endl;
		cout << "  Receptors (inputs): " << Receptors << endl;
		cout << "  Classes: " << Classes << endl;
		cout << "  Images: " << Images << endl;
		cout << "  Neurons created: " << (Neirons - Inputs) << endl;
		cout << "  Threads: " << NumThreads << (UseMultithreading ? " (multithreaded)" : " (single-threaded)") << endl;
		cout << "Timing:" << endl;
		cout << "  Training time: " << trainingDuration.count() << " ms" << endl;
		cout << "  Training iterations: " << trainingIterations << endl;
		if (trainingIterations > 0) {
			double msPerIteration = (double)trainingDuration.count() / trainingIterations;
			cout << "  Time per iteration: " << msPerIteration << " ms" << endl;
		}
		if (trainingDuration.count() > 0) {
			double classesPerSecond = (double)Classes * 1000.0 / trainingDuration.count();
			cout << "  Training speed: " << classesPerSecond << " classes/sec" << endl;
			double neuronsPerSecond = (double)(Neirons - Inputs) * 1000.0 / trainingDuration.count();
			cout << "  Neuron creation speed: " << neuronsPerSecond << " neurons/sec" << endl;
		}
		cout << "=== End Benchmark ===" << endl;

		return 0;
	}

	// Режим тестирования: проверяем точность классификации
	if (testMode) {
		cout << "\n=== Running automated classification test ===" << endl;
		int passed = 0;
		int failed = 0;
		float threshold = 0.5f;  // Порог классификации

		for (int img = 0; img < Images; img++) {
			// Устанавливаем входы сети для текущего образа
			for (int d = 0; d < Receptors; d++) {
				NetInput[d] = vx[img][d];
			}
			clear_val_cache(nei, MAX_NEURONS);

			// Находим класс с максимальным выходом
			int predictedClass = -1;
			float maxOutput = -big;
			for (int c = 0; c < Classes; c++) {
				float output = GetNeironVal(NetOutput[c]);
				if (output > maxOutput) {
					maxOutput = output;
					predictedClass = c;
				}
			}

			int expectedClass = const_words[img].id;
			float expectedOutput = GetNeironVal(NetOutput[expectedClass]);

			// Тест проходит если:
			// 1. Предсказанный класс совпадает с ожидаемым, ИЛИ
			// 2. Выход ожидаемого класса выше порога
			bool testPassed = (predictedClass == expectedClass) || (expectedOutput >= threshold);

			if (testPassed) {
				passed++;
				cout << "[PASS] Image " << img << " (\"" << const_words[img].word.substr(0, 10)
					 << "...\"): expected class " << expectedClass
					 << ", predicted " << predictedClass
					 << " (output=" << expectedOutput << ")" << endl;
			} else {
				failed++;
				cout << "[FAIL] Image " << img << " (\"" << const_words[img].word.substr(0, 10)
					 << "...\"): expected class " << expectedClass
					 << ", predicted " << predictedClass
					 << " (output=" << expectedOutput << ")" << endl;
			}
		}

		cout << "\n=== Test Summary ===" << endl;
		cout << "Total images: " << Images << endl;
		cout << "Passed: " << passed << endl;
		cout << "Failed: " << failed << endl;
		float accuracy = (float)passed / (float)Images * 100.0f;
		cout << "Accuracy: " << accuracy << "%" << endl;

		if (failed == 0) {
			cout << "\nAll tests PASSED!" << endl;
			return 0;
		} else {
			cout << "\nSome tests FAILED!" << endl;
			return 1;
		}
	}

	// Интерактивный режим
	do
	{
		// Считываем и обрабатываем ввод пользователя
		if (InputStr[0] == 0)
		{
			cout << "input word:";
			readkeyboard(InputStr);
		}

		memset(word_buf, 0, StringSize);
		strcpy_s(word_buf, InputStr);
		memset(InputStr, 0, StringSize);

		if (cmp("Q") || cmp("q")) return 0;

		// Устанавливаем входные значения
		for (int d = 0; d < Receptors; d++)
		{
			if (word_buf[d] == 0)
			{
				NetInput[d] = float((unsigned char)' ') / float(max_num);
			}
			else
			{
				NetInput[d] = float((unsigned char)word_buf[d]) / float(max_num);
			}
		}

		clear_val_cache(nei, MAX_NEURONS);

		// Выводим состояние выходов нейросети
		for (int out = 0; out < Classes; out++)
		{
			float z1 = GetNeironVal(NetOutput[out]) * 100.0f;
			// Обработка некорректных значений
			if (!std::isfinite(z1)) z1 = 0.0f;
			if (z1 < 0.0f) z1 = 0.0f;
			if (z1 > 100.0f) z1 = 100.0f;
			cout << long(z1) << "%" << " - " << classes[out] << endl;
		}

	} while (true);
}
