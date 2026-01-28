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
#include <csignal>
#include <nlohmann/json.hpp>
#include "simd_ops.h"

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
// Глобальные переменные для прерывания обучения
// ============================================================================

volatile std::sig_atomic_t g_interruptRequested = 0;  // Флаг запроса прерывания
string g_autoSavePath = "";                           // Путь для автосохранения при прерывании

// ============================================================================
// Операции нейронов - элементарные математические действия
// SIMD-оптимизированные версии для ускорения вычислений
// ============================================================================

typedef void(__fastcall *oper)(float*, const float*, const float*, const int);

// Флаг для включения/выключения SIMD во время выполнения
bool UseSIMD = true;

// Сумма - SIMD оптимизированная версия
void __fastcall op_1(float* r, const float* z1, const float* z2, const int size) {
	op_add_simd(r, z1, z2, size);
}

// Разность (z1 - z2) - SIMD оптимизированная версия
void __fastcall op_2(float* r, const float* z1, const float* z2, const int size) {
	op_sub_simd(r, z1, z2, size);
}

// Разность (z2 - z1) - SIMD оптимизированная версия
void __fastcall op_3(float* r, const float* z1, const float* z2, const int size) {
	op_rsub_simd(r, z1, z2, size);
}

// Произведение - SIMD оптимизированная версия
void __fastcall op_4(float* r, const float* z1, const float* z2, const int size) {
	op_mul_simd(r, z1, z2, size);
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
 * Обработчик сигнала прерывания (Ctrl+C)
 *
 * Устанавливает флаг прерывания для корректного завершения обучения.
 */
void interruptHandler(int signal) {
	if (signal == SIGINT) {
		if (g_interruptRequested == 0) {
			g_interruptRequested = 1;
			cout << "\n[INTERRUPT] Ctrl+C detected. Training will stop after current iteration..." << endl;
			cout << "[INTERRUPT] Press Ctrl+C again to force exit (may lose progress)." << endl;
		} else {
			cout << "\n[INTERRUPT] Force exit requested." << endl;
			exit(1);
		}
	}
}

/**
 * Вывод справки по использованию программы
 */
void printUsage(const char* programName) {
	cout << "Usage: " << programName << " [options]" << endl;
	cout << endl;
	cout << "MODES:" << endl;
	cout << "  Training mode (default): Train network and optionally save to file" << endl;
	cout << "  Inference mode: Load trained network and classify inputs" << endl;
	cout << "  Retraining mode: Load existing network and continue training with new data" << endl;
	cout << endl;
	cout << "TRAINING OPTIONS:" << endl;
	cout << "  -c, --config <file>  Load training configuration from JSON file" << endl;
	cout << "  -s, --save <file>    Save trained network to JSON file after training" << endl;
	cout << "  -t, --test           Run automated test after training (no interactive mode)" << endl;
	cout << "  -b, --benchmark      Run benchmark to measure training speed" << endl;
	cout << endl;
	cout << "RETRAINING OPTIONS:" << endl;
	cout << "  -r, --retrain <file> Load existing network and continue training (retraining mode)" << endl;
	cout << "                       Combines -l (load) with training mode. Requires -c for new data." << endl;
	cout << "                       New classes in config (without output_neuron) will be trained." << endl;
	cout << endl;
	cout << "INFERENCE OPTIONS:" << endl;
	cout << "  -l, --load <file>    Load trained network from JSON file (inference mode)" << endl;
	cout << "  -i, --input <text>   Classify single input text and exit (non-interactive)" << endl;
	cout << "  --verify             Verify accuracy of loaded model on training config (-c required)" << endl;
	cout << endl;
	cout << "PERFORMANCE OPTIONS:" << endl;
	cout << "  -j, --threads <n>    Number of threads to use (0 = auto, default)" << endl;
	cout << "  --single-thread      Disable multithreading (use single thread)" << endl;
	cout << "  --no-simd            Disable SIMD optimizations (use scalar operations)" << endl;
	cout << endl;
	cout << "GENERAL OPTIONS:" << endl;
	cout << "  -h, --help           Show this help message" << endl;
	cout << "  --list-funcs         List available training functions" << endl;
	cout << endl;
	cout << "INTERRUPTION:" << endl;
	cout << "  Press Ctrl+C during training to interrupt gracefully." << endl;
	cout << "  The network will be saved if -s is specified." << endl;
	cout << "  Training can be continued later with -r option." << endl;
	cout << endl;
	cout << "EXAMPLES:" << endl;
	cout << "  " << programName << " -c configs/default.json -s model.json  # Train and save" << endl;
	cout << "  " << programName << " -l model.json                          # Interactive inference" << endl;
	cout << "  " << programName << " -l model.json -i \"time\"                # Single classification" << endl;
	cout << "  " << programName << " -r model.json -c configs/new.json -s model_v2.json  # Retrain" << endl;
	cout << "  " << programName << " -l model.json -c configs/test.json --verify  # Verify accuracy" << endl;
	cout << endl;
	cout << "JSON config format (training):" << endl;
	cout << "  {" << endl;
	cout << "    \"receptors\": 20," << endl;
	cout << "    \"classes\": [" << endl;
	cout << "      { \"id\": 0, \"word\": \"\" }," << endl;
	cout << "      { \"id\": 1, \"word\": \"time\" }" << endl;
	cout << "    ]," << endl;
	cout << "    \"generate_shifts\": true," << endl;
	cout << "    \"funcs\": [\"triplet_parallel\"]  // Optional: specify training functions" << endl;
	cout << "  }" << endl;
	cout << endl;
	cout << "Use --list-funcs to see all available training functions." << endl;
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
	string retrainPath = "";
	string inputText = "";
	bool testMode = false;
	bool benchmarkMode = false;
	bool inferenceMode = false;
	bool retrainMode = false;
	bool verifyMode = false;

	for (int i = 1; i < argc; i++) {
		string arg = argv[i];
		if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
			configPath = argv[++i];
		} else if ((arg == "-s" || arg == "--save") && i + 1 < argc) {
			savePath = argv[++i];
		} else if ((arg == "-l" || arg == "--load") && i + 1 < argc) {
			loadPath = argv[++i];
			inferenceMode = true;
		} else if ((arg == "-r" || arg == "--retrain") && i + 1 < argc) {
			retrainPath = argv[++i];
			retrainMode = true;
		} else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
			inputText = argv[++i];
		} else if (arg == "-t" || arg == "--test") {
			testMode = true;
		} else if (arg == "-b" || arg == "--benchmark") {
			benchmarkMode = true;
		} else if (arg == "--verify") {
			verifyMode = true;
		} else if ((arg == "-j" || arg == "--threads") && i + 1 < argc) {
			NumThreads = atoi(argv[++i]);
		} else if (arg == "--single-thread") {
			UseMultithreading = false;
		} else if (arg == "--no-simd") {
			UseSIMD = false;
		} else if (arg == "-h" || arg == "--help") {
			printUsage(argv[0]);
			return 0;
		} else if (arg == "--list-funcs") {
			printAvailableLearningFuncs();
			return 0;
		}
	}

	// Установка обработчика сигнала Ctrl+C
	std::signal(SIGINT, interruptHandler);
	g_autoSavePath = savePath;

	// ===== РЕЖИМ ВЕРИФИКАЦИИ ТОЧНОСТИ =====
	// Загружаем обученную сеть и тестируем на данных из конфига
	if (verifyMode) {
		if (loadPath.empty()) {
			cerr << "Error: --verify requires -l <model.json>" << endl;
			return 1;
		}
		if (configPath.empty()) {
			cerr << "Error: --verify requires -c <config.json> for test data" << endl;
			return 1;
		}

		// Загружаем сеть
		if (!loadNetwork(loadPath)) {
			return 1;
		}

		// Загружаем конфиг для тестовых данных (сохраняем текущие Receptors для проверки)
		int savedReceptors = Receptors;
		if (!loadConfig(configPath, Receptors)) {
			return 1;
		}

		// Проверяем совместимость
		if (Receptors != savedReceptors) {
			cerr << "Error: Config receptors (" << Receptors << ") don't match model (" << savedReceptors << ")" << endl;
			return 1;
		}

		cout << "\n=== Verifying model accuracy ===" << endl;

		int passed = 0;
		int failed = 0;
		int total = const_words.size();

		for (int img = 0; img < total; img++) {
			// Устанавливаем входы сети из образа
			for (int d = 0; d < Receptors; d++) {
				if (d < (int)const_words[img].word.length() && const_words[img].word[d] != 0) {
					NetInput[d] = float((unsigned char)const_words[img].word[d]) / float(max_num);
				} else {
					NetInput[d] = float((unsigned char)' ') / float(max_num);
				}
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
			float expectedOutput = (expectedClass < Classes) ? GetNeironVal(NetOutput[expectedClass]) : 0.0f;

			// Проверяем корректность
			bool testPassed = (predictedClass == expectedClass) || (expectedOutput >= 0.5f);

			if (testPassed) {
				passed++;
			} else {
				failed++;
				string shortWord = const_words[img].word.substr(0, 10);
				cout << "[FAIL] \"" << shortWord << "...\" expected class " << expectedClass
					 << ", predicted " << predictedClass << " (conf=" << (int)(expectedOutput*100) << "%)" << endl;
			}
		}

		cout << "\n=== Verification Summary ===" << endl;
		cout << "Total samples: " << total << endl;
		cout << "Passed: " << passed << endl;
		cout << "Failed: " << failed << endl;
		float accuracy = (float)passed / (float)total * 100.0f;
		cout << "Accuracy: " << accuracy << "%" << endl;

		return (failed == 0) ? 0 : 1;
	}

	// ===== РЕЖИМ ИНФЕРЕНСА =====
	// Загружаем обученную сеть и переходим к классификации
	if (inferenceMode && !retrainMode) {
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

	// ===== РЕЖИМ ДООБУЧЕНИЯ =====
	// Загружаем существующую сеть и дообучаем на новых данных
	vector<int> trainedClasses;  // Индексы уже обученных классов
	vector<int> newClassIds;     // Индексы новых классов для обучения

	if (retrainMode) {
		if (configPath.empty()) {
			cerr << "Error: Retraining mode requires -c <config.json> for new training data" << endl;
			return 1;
		}

		// Загружаем сеть для дообучения
		if (!loadNetworkForRetraining(retrainPath, trainedClasses)) {
			return 1;
		}

		// Объединяем с новой конфигурацией
		if (!mergeConfigForRetraining(configPath, trainedClasses, newClassIds)) {
			return 1;
		}

		if (newClassIds.empty()) {
			cout << "\nAll classes are already trained. Nothing to do." << endl;
			cout << "Use --verify to check accuracy or -l for inference mode." << endl;
			return 0;
		}

		cout << "\nRetraining mode: will train " << newClassIds.size() << " new class(es)" << endl;
	}

	// ===== РЕЖИМ ОБУЧЕНИЯ =====

	// Загружаем конфигурацию или используем значения по умолчанию (если не дообучение)
	if (!retrainMode) {
		if (!configPath.empty()) {
			if (!loadConfig(configPath, Receptors)) {
				return 1;
			}
		} else {
			initDefaultConfig(Receptors);
		}
	}

	// Инициализация генератора случайных чисел
	// В тестовом режиме используем фиксированный seed для воспроизводимости
	unsigned int randomSeed;
	if (testMode) {
		randomSeed = 42;  // Фиксированный seed для детерминированных тестов
	} else {
		randomSeed = (unsigned int)time(nullptr);
	}
	srand(randomSeed);
	cout << "Random seed: " << randomSeed << endl;

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

	// Вывод информации о SIMD
	cout << "SIMD: " << getSIMDInfo() << (UseSIMD ? "" : " (disabled via --no-simd)") << endl;

	// Вычисляем производные значения после загрузки конфигурации
	Images = const_words.size();
	Inputs = Receptors + base_size;

	// В режиме дообучения Neirons уже установлен из загруженной сети
	if (!retrainMode) {
		Neirons = Inputs;
	}

	// Выделяем динамические массивы
	NetInput.resize(Inputs);
	vx.resize(Images);
	for (int i = 0; i < Images; i++) {
		vx[i].resize(Receptors);
	}
	vz.resize(Classes);
	NetOutput.resize(Classes);

	// Инициализируем массив нейронов
	// В режиме дообучения нейроны уже инициализированы, но нужно расширить кэши под новые образы
	if (retrainMode) {
		// Расширяем кэши под новое количество образов
		for (int n = 0; n < MAX_NEURONS; n++) {
			nei[n].c.resize(Images);
			nei[n].cached = false;  // Сбрасываем кэши, т.к. теперь другое количество образов
		}
	} else {
		initNeurons();
	}

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

	// В режиме дообучения устанавливаем ошибку 0 для уже обученных классов
	if (retrainMode) {
		for (int c : trainedClasses) {
			class_er[c] = 0.0f;
		}
		// Начинаем с первого нового класса
		if (!newClassIds.empty()) {
			classIndex = newClassIds[0];
		}
	}

	// Засекаем время обучения
	auto trainingStartTime = chrono::high_resolution_clock::now();
	int trainingIterations = 0;
	bool trainingInterrupted = false;

	// Цикл обучения
	do
	{
		// Проверяем запрос на прерывание
		if (g_interruptRequested) {
			cout << "\n[INTERRUPT] Training interrupted by user." << endl;
			trainingInterrupted = true;
			break;
		}

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
			// Если заданы функции обучения в конфиге - используем их последовательно
			if (!g_trainingFuncs.empty()) {
				// Вызываем все указанные функции в указанной последовательности
				for (const auto& funcName : g_trainingFuncs) {
					if (class_er[classIndex] <= er) break;  // Уже достигли нужной ошибки

					LearningFunc func = getLearningFunc(funcName);
					if (func != nullptr) {
						float newError = func();
						if (newError < class_er[classIndex]) {
							class_er[classIndex] = newError;
							NetOutput[classIndex] = Neirons - 1;
						}
					} else {
						cerr << "Warning: Unknown training function '" << funcName << "', skipping" << endl;
					}
				}
			} else {
				// По умолчанию: используем triplet_random_parallel (rndrod4_parallel)
				class_er[classIndex] = triplet_random_parallel();
				NetOutput[classIndex] = Neirons - 1;
			}
		}

		cout << ", n" << NetOutput[classIndex] << " = " << class_er[classIndex] << endl;

		if (++classIndex >= Classes)  // Переходим к следующему классу по кругу
			classIndex = 0;

		// Проверяем достижение лимита нейронов
		if (Neirons >= MAX_NEURONS - 10) {  // Оставляем запас в 10 нейронов
			cout << "\n[WARNING] Maximum neuron limit (" << MAX_NEURONS << ") nearly reached. Stopping training." << endl;
			break;
		}

	} while (sum(class_er.data(), Classes) > Classes * er);

	// Конец обучения
	auto trainingEndTime = chrono::high_resolution_clock::now();
	auto trainingDuration = chrono::duration_cast<chrono::milliseconds>(trainingEndTime - trainingStartTime);

	if (trainingInterrupted) {
		cout << "\nTraining interrupted after " << trainingIterations << " iterations." << endl;
	} else {
		cout << "\nTraining completed!" << endl;
	}

	cout << "Errors per class:" << endl;
	int trainedCount = 0;
	int untrainedCount = 0;
	for (int c = 0; c < Classes; c++) {
		bool isTrained = (class_er[c] <= er);
		cout << "  Class " << c << " (" << classes[c] << "): error = " << class_er[c];
		if (isTrained) {
			cout << " [trained]";
			trainedCount++;
		} else {
			cout << " [not trained]";
			untrainedCount++;
		}
		cout << endl;
	}

	if (trainingInterrupted && untrainedCount > 0) {
		cout << "\nWarning: " << untrainedCount << " class(es) are not fully trained." << endl;
		cout << "Use -r option to continue training later." << endl;
	}

	// Сохраняем сеть если указан путь
	if (!savePath.empty()) {
		if (!saveNetwork(savePath)) {
			cerr << "Warning: Failed to save network to " << savePath << endl;
		} else if (trainingInterrupted) {
			cout << "\nNetwork state saved. To continue training, use:" << endl;
			cout << "  " << argv[0] << " -r " << savePath << " -c <config.json> -s <output.json>" << endl;
		}
	} else if (trainingInterrupted) {
		cout << "\nWarning: Network state not saved (no -s option specified)." << endl;
		cout << "Progress will be lost. Use -s to save network for later continuation." << endl;
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
		cout << "  SIMD: " << getSIMDInfo() << (UseSIMD ? " (enabled)" : " (disabled)") << endl;
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
