/*
 * json_io.h - Функции загрузки и сохранения данных в формате JSON
 *
 * Этот модуль содержит функции для:
 * - Загрузки конфигурации обучения из JSON файла
 * - Сохранения обученной нейронной сети в JSON файл
 * - Загрузки обученной нейронной сети из JSON файла для инференса
 *
 * Примечание: Этот файл предназначен для включения в main.cpp после
 * определения всех необходимых типов и переменных.
 */

#ifndef JSON_IO_H
#define JSON_IO_H

// ============================================================================
// Функции загрузки конфигурации
// ============================================================================

/**
 * Генерация сдвинутых вариантов слова
 *
 * Создаёт все возможные позиции слова в пределах заданной ширины рецепторов.
 * Например, слово "time" длиной 4 символа при receptors=20 даст 17 вариантов:
 * "time                " (слово в начале)
 * " time               " (слово сдвинуто на 1)
 * и т.д.
 *
 * @param word - исходное слово
 * @param id - идентификатор класса
 * @param receptors - количество входов нейронной сети
 */
void generateShiftedImages(const string& word, int id, int receptors) {
    // Дополняем слово пробелами до длины receptors
    string padded = word;
    while ((int)padded.length() < receptors) {
        padded += ' ';
    }
    padded = padded.substr(0, receptors);

    // Добавляем оригинальную версию (выровненную по левому краю)
    const_words.push_back({padded, id});

    // Генерируем все сдвинутые версии (слово может находиться в любой позиции)
    int wordLen = word.length();
    for (int shift = 1; shift <= receptors - wordLen; shift++) {
        string shifted(shift, ' ');
        shifted += word;
        while ((int)shifted.length() < receptors) {
            shifted += ' ';
        }
        const_words.push_back({shifted.substr(0, receptors), id});
    }
}

/**
 * Загрузка конфигурации из JSON файла
 *
 * Конфигурационный файл может содержать:
 * - receptors: количество входов нейронной сети
 * - classes: массив классов с их словами для обучения
 * - images: напрямую заданные образы (альтернатива classes)
 * - generate_shifts: флаг генерации сдвинутых образов
 * - description: описание конфигурации
 *
 * @param configPath - путь к JSON файлу конфигурации
 * @param receptors - выходной параметр: количество рецепторов
 * @return true при успешной загрузке, false при ошибке
 */
bool loadConfig(const string& configPath, int& receptors) {
    ifstream configFile(configPath);
    if (!configFile.is_open()) {
        cerr << "Error: Cannot open config file: " << configPath << endl;
        return false;
    }

    try {
        json config;
        configFile >> config;

        // Загружаем количество рецепторов (входов нейронной сети)
        if (config.contains("receptors")) {
            receptors = config["receptors"].get<int>();
        }

        // Очищаем векторы перед загрузкой
        const_words.clear();
        classes.clear();

        // Проверяем, заданы ли образы напрямую
        if (config.contains("images")) {
            // Загружаем образы напрямую из JSON
            Classes = 0;
            for (const auto& img : config["images"]) {
                string word = img["word"].get<string>();
                int id = img["id"].get<int>();
                const_words.push_back({word, id});
                if (id >= Classes) {
                    Classes = id + 1;
                }
            }
            // Создаём вектор классов из уникальных идентификаторов
            classes.resize(Classes);
            for (const auto& img : const_words) {
                if (classes[img.id].empty()) {
                    // Сохраняем первое слово как имя класса (с удалением пробелов)
                    string trimmed = img.word;
                    size_t end = trimmed.find_last_not_of(' ');
                    if (end != string::npos) {
                        trimmed = trimmed.substr(0, end + 1);
                    } else {
                        trimmed = "";  // Все пробелы - пустой класс
                    }
                    classes[img.id] = trimmed;
                }
            }
        }
        // Иначе генерируем из классов
        else if (config.contains("classes")) {
            Classes = config["classes"].size();
            classes.resize(Classes);
            bool generateShifts = true;
            if (config.contains("generate_shifts")) {
                generateShifts = config["generate_shifts"].get<bool>();
            }

            for (const auto& cls : config["classes"]) {
                string word = cls["word"].get<string>();
                int id = cls["id"].get<int>();

                // Сохраняем уникальное имя класса
                classes[id] = word;

                if (generateShifts && word.length() > 0) {
                    generateShiftedImages(word, id, receptors);
                } else {
                    // Просто добавляем слово, дополненное до длины receptors
                    string padded = word;
                    while ((int)padded.length() < receptors) {
                        padded += ' ';
                    }
                    const_words.push_back({padded.substr(0, receptors), id});
                }
            }
        }

        cout << "Loaded config: " << configPath << endl;
        cout << "  Receptors: " << receptors << endl;
        cout << "  Classes: " << Classes << endl;
        cout << "  Images: " << const_words.size() << endl;
        if (config.contains("description")) {
            cout << "  Description: " << config["description"].get<string>() << endl;
        }

        return true;
    }
    catch (const json::exception& e) {
        cerr << "JSON parsing error: " << e.what() << endl;
        return false;
    }
}

/**
 * Инициализация конфигурации по умолчанию
 *
 * Создаёт стандартную конфигурацию с 4 классами:
 * - пустая строка (класс 0)
 * - "time" (класс 1)
 * - "hour" (класс 2)
 * - "main" (класс 3)
 *
 * @param receptors - количество входов нейронной сети
 */
void initDefaultConfig(int receptors) {
    Classes = 4;
    const_words.clear();
    classes.clear();

    // Генерируем пустой класс
    string empty(receptors, ' ');
    const_words.push_back({empty, 0});
    classes.push_back(empty);

    // Генерируем сдвинутые образы для каждого слова
    generateShiftedImages("time", 1, receptors);
    classes.push_back("time");
    generateShiftedImages("hour", 2, receptors);
    classes.push_back("hour");
    generateShiftedImages("main", 3, receptors);
    classes.push_back("main");

    cout << "Using default configuration" << endl;
    cout << "  Receptors: " << receptors << endl;
    cout << "  Classes: " << classes.size() << endl;
    cout << "  Images: " << const_words.size() << endl;
}

// ============================================================================
// Функции сохранения и загрузки обученной сети
// ============================================================================

/**
 * Сохранение обученной нейронной сети в JSON файл
 *
 * Сохраняемая информация:
 * - Параметры сети (receptors, inputs, neurons_count)
 * - Базисные значения
 * - Классы и их выходные нейроны
 * - Структура нейронов (входы i, j и операция op)
 *
 * @param filePath - путь для сохранения JSON файла
 * @return true при успешном сохранении, false при ошибке
 */
bool saveNetwork(const string& filePath) {
    try {
        json network;

        // Сохраняем конфигурацию
        network["receptors"] = Receptors;
        network["base_size"] = base_size;
        network["inputs"] = Inputs;
        network["neurons_count"] = Neirons;

        // Сохраняем базисные значения
        json basisArray = json::array();
        for (int i = 0; i < base_size; i++) {
            basisArray.push_back(base[i]);
        }
        network["basis"] = basisArray;

        // Сохраняем имена классов
        json classesArray = json::array();
        for (int c = 0; c < Classes; c++) {
            json cls;
            cls["id"] = c;
            cls["name"] = classes[c];
            cls["output_neuron"] = NetOutput[c];
            classesArray.push_back(cls);
        }
        network["classes"] = classesArray;

        // Сохраняем нейроны (только созданные, т.к. 0..Inputs-1 это входы)
        // ID нейронов неявные - это Inputs + индекс_в_массиве
        json neuronsArray = json::array();
        for (int n = Inputs; n < Neirons; n++) {
            json neuron;
            neuron["i"] = nei[n].i;
            neuron["j"] = nei[n].j;
            neuron["op"] = getOpIndex(nei[n].op);
            neuronsArray.push_back(neuron);
        }
        network["neurons"] = neuronsArray;

        // Добавляем метаданные
        network["version"] = "1.0";
        network["description"] = "Trained neural network model";

        // Записываем в файл с форматированием
        ofstream outFile(filePath);
        if (!outFile.is_open()) {
            cerr << "Error: Cannot open output file: " << filePath << endl;
            return false;
        }
        outFile << network.dump(2);
        outFile.close();

        cout << "Network saved to: " << filePath << endl;
        cout << "  Classes: " << Classes << endl;
        cout << "  Neurons: " << (Neirons - Inputs) << endl;
        cout << "  Total nodes: " << Neirons << endl;

        return true;
    }
    catch (const exception& e) {
        cerr << "Error saving network: " << e.what() << endl;
        return false;
    }
}

/**
 * Загрузка обученной нейронной сети из JSON файла
 *
 * Используется в режиме инференса для загрузки ранее обученной сети.
 * После загрузки сеть готова к классификации входных данных.
 *
 * @param filePath - путь к JSON файлу с моделью
 * @return true при успешной загрузке, false при ошибке
 */
bool loadNetwork(const string& filePath) {
    ifstream inFile(filePath);
    if (!inFile.is_open()) {
        cerr << "Error: Cannot open network file: " << filePath << endl;
        return false;
    }

    try {
        json network;
        inFile >> network;

        // Загружаем конфигурацию
        Receptors = network["receptors"].get<int>();
        Inputs = network["inputs"].get<int>();
        Neirons = network["neurons_count"].get<int>();

        // Проверяем совпадение размера базиса
        int loadedBaseSize = network["base_size"].get<int>();
        if (loadedBaseSize != base_size) {
            cerr << "Warning: Basis size mismatch (file: " << loadedBaseSize
                 << ", expected: " << base_size << ")" << endl;
        }

        // Выделяем массив входов сети
        NetInput.resize(Inputs);

        // Устанавливаем базисные значения
        for (int i = 0; i < base_size; i++) {
            NetInput[i + Receptors] = base[i];
        }

        // Загружаем классы
        const auto& classesArray = network["classes"];
        Classes = classesArray.size();
        classes.clear();
        classes.resize(Classes);
        NetOutput.resize(Classes);

        for (const auto& cls : classesArray) {
            int id = cls["id"].get<int>();
            classes[id] = cls["name"].get<string>();
            NetOutput[id] = cls["output_neuron"].get<int>();
        }

        // Инициализируем нейроны (без кэша образов, т.к. данные обучения не нужны)
        nei.resize(MAX_NEURONS);
        for (int n = 0; n < MAX_NEURONS; n++) {
            nei[n].cached = false;
            nei[n].val_cached = false;
        }

        // Загружаем структуру нейронов
        // ID нейронов неявные - это Inputs + индекс_в_массиве
        const auto& neuronsArray = network["neurons"];
        int neuronIndex = Inputs;  // Начинаем с Inputs, т.к. 0..Inputs-1 это входы
        for (const auto& neuron : neuronsArray) {
            nei[neuronIndex].i = neuron["i"].get<int>();
            nei[neuronIndex].j = neuron["j"].get<int>();
            int opIndex = neuron["op"].get<int>();
            if (opIndex >= 0 && opIndex < op_count) {
                nei[neuronIndex].op = op[opIndex];
            } else {
                nei[neuronIndex].op = op[0];  // По умолчанию первая операция
            }
            neuronIndex++;
        }

        cout << "Network loaded from: " << filePath << endl;
        cout << "  Receptors: " << Receptors << endl;
        cout << "  Classes: " << Classes << endl;
        for (int c = 0; c < Classes; c++) {
            cout << "    " << c << ": " << classes[c] << endl;
        }
        cout << "  Neurons: " << (Neirons - Inputs) << endl;

        return true;
    }
    catch (const json::exception& e) {
        cerr << "JSON parsing error: " << e.what() << endl;
        return false;
    }
    catch (const exception& e) {
        cerr << "Error loading network: " << e.what() << endl;
        return false;
    }
}

#endif // JSON_IO_H
