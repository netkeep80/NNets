/*Самообучение в саморождающихся и гибнущих структурах


  при создании НС ей задаётся фиксированное количество входов
  и задаётся максимум количества выходов
  при обучении каждый новый образ, если он не может однозначно
  классифицироваться сетью, будет добавлен как новый образ
  когда количество свободных выходов иссякнет,
  сеть должна будет каждый раз переобучатсья чтобы классифицировать новый образ как
  один из уже известных

*/

#include <iostream>
#include <fstream>
#include <strstream>
#include <math.h>
#include <string.h>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

struct Image
{
	string word;
	int	id;
};

// Dynamic configuration loaded from JSON
int		Classes = 4;
vector<string>	classes;
vector<Image>	const_words;

// Function to generate shifted variants of a word
void generateShiftedImages(const string& word, int id, int receptors) {
	// Pad word to receptors length
	string padded = word;
	while ((int)padded.length() < receptors) {
		padded += ' ';
	}
	padded = padded.substr(0, receptors);

	// Add original (left-aligned) version
	const_words.push_back({padded, id});

	// Generate all shifted versions (word can appear at any position)
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

// Load configuration from JSON file
bool loadConfig(const string& configPath, int& receptors) {
	ifstream configFile(configPath);
	if (!configFile.is_open()) {
		cerr << "Error: Cannot open config file: " << configPath << endl;
		return false;
	}

	try {
		json config;
		configFile >> config;

		// Load receptors (number of neural network inputs)
		if (config.contains("receptors")) {
			receptors = config["receptors"].get<int>();
		}

		// Clear vectors before loading
		const_words.clear();
		classes.clear();

		// Check if custom images are provided directly
		if (config.contains("images")) {
			// Load images directly from JSON
			Classes = 0;
			for (const auto& img : config["images"]) {
				string word = img["word"].get<string>();
				int id = img["id"].get<int>();
				const_words.push_back({word, id});
				if (id >= Classes) {
					Classes = id + 1;
				}
			}
			// Build classes vector from unique class IDs
			classes.resize(Classes);
			for (const auto& img : const_words) {
				if (classes[img.id].empty()) {
					// Store the first word as the class name (trimmed)
					string trimmed = img.word;
					size_t end = trimmed.find_last_not_of(' ');
					if (end != string::npos) {
						trimmed = trimmed.substr(0, end + 1);
					} else {
						trimmed = "";  // All spaces - empty class
					}
					classes[img.id] = trimmed;
				}
			}
		}
		// Otherwise generate from classes
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

				// Store unique class name
				classes[id] = word;

				if (generateShifts && word.length() > 0) {
					generateShiftedImages(word, id, receptors);
				} else {
					// Just add the word padded to receptors length
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

// Initialize default configuration (original hardcoded values)
void initDefaultConfig(int receptors) {
	Classes = 4;
	const_words.clear();
	classes.clear();

	// Generate empty class
	string empty(receptors, ' ');
	const_words.push_back({empty, 0});
	classes.push_back(empty);

	// Generate shifted images for each word
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

// Print usage information
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

const	int		rod2_iter = 2;
const	int		rndrod_iter = 10;
const	int		rndrod2_iter = rndrod_iter;
const	float	base[] = {
	0.125,
	0.25,	//1355
	0.5,	//1620
	1.0,	//1010
	2.0,	//1120
	4.0,	//1202
	8.0,
	-0.125,
	-0.25,	//1408
	-0.5,	//1078
	-1.0,	//1088
	-2.0,	//1152
	-4.0,	//1192
	-8.0,
};	//	785
const	float	big = 1000000000000000000.f;
const	int		max_num = 256;				//	число возможных состояний каждого входа
int		Images = 0;							//	число подаваемых образов на нейроннуйю сеть (загружается из конфига)
int		Receptors = 20;						//	число рабочих входов нейронной сети (загружается из конфига)
const	int		StringSize = 256;			//	максимальный размер строки
const	int		base_size = sizeof(base) / sizeof(float);
int		Inputs = 0;							//	общее число входов нейронной сети (рассчитывается как Receptors + base_size)
int		Neirons = 0;						//	число рожденных нейронов
vector<float>	NetInput;					//	входные напряжения
vector<vector<float>>	vx;					//	вектор входных значений
vector<float>	vz;							//	вектор выходных значений
vector<int>		NetOutput;
char	InputStr[StringSize], word_buf[StringSize];


//////////////////////////////////////////////////////////////////////////////
//	возможные операции нейрона
typedef	void(__fastcall *oper)(float*, const float*, const float*, const int);
/*
#define nop(name,a,b) float op_##name(float z1, float z2) { return a * z1 + b * z2; }
#define nop_vec(name,b)		\
nop(0##name,  0.9f, b)		\
nop(1##name, -0.9f, b)		\
nop(2##name,  0.7f, b)		\
nop(3##name, -0.7f, b)		\
nop(4##name,  0.5f, b)		\
nop(5##name, -0.5f, b)		\
nop(6##name,  0.3f, b)		\
nop(7##name, -0.3f, b)		\
nop(8##name,  0.1f, b)		\
nop(9##name, -0.1f, b)

nop_vec(0,  0.95f)
nop_vec(1, -0.95f)
nop_vec(2,  0.75f)
nop_vec(3, -0.75f)
nop_vec(4,  0.55f)
nop_vec(5, -0.55f)
nop_vec(6,  0.35f)
nop_vec(7, -0.35f)
nop_vec(8,  0.15f)
nop_vec(9, -0.15f)

float op__0(float z1, float z2) { if (z2 != 0.0) return z1 / z2; else return big; }
float op__1(float z1, float z2) { if (z1 != 0.0) return z2 / z1; else return big; }
float op__2(float z1, float z2) { return z2 * z2 + z1; }
float op__3(float z1, float z2) { return z1 * z1 + z2; }
float op__4(float z1, float z2) { return z2 * z2 - z1; }
float op__5(float z1, float z2) { return z1 * z1 - z2; }
float op__6(float z1, float z2) { return z1 * z2 / (z1 + z2); }

oper	op[] = { op__0, op__1, op__2, op__3, op__4, op__5, op__6,
	op_00, op_01, op_02, op_03, op_04, op_05, op_06, op_07, op_08, op_09,	//	0b
	op_10, op_11, op_12, op_13, op_14, op_15, op_16, op_17, op_18, op_19,	//	1b
	op_20, op_21, op_22, op_23, op_24, op_25, op_26, op_27, op_28, op_29,	//	2b
	op_30, op_31, op_32, op_33, op_34, op_35, op_36, op_37, op_38, op_39,	//	3b
	op_40, op_41, op_42, op_43, op_44, op_45, op_46, op_47, op_48, op_49,	//	4b
	op_50, op_51, op_52, op_53, op_54, op_55, op_56, op_57, op_58, op_59,	//	5b
	op_60, op_61, op_62, op_63, op_64, op_65, op_66, op_67, op_68, op_69,	//	6b
	op_70, op_71, op_72, op_73, op_74, op_75, op_76, op_77, op_78, op_79,	//	7b
	op_80, op_81, op_82, op_83, op_84, op_85, op_86, op_87, op_88, op_89,	//	8b
	op_90, op_91, op_92, op_93, op_94, op_95, op_96, op_97, op_98, op_99	//	9b
};
*/

//	sum
void __fastcall op_1(float* r, const float* z1, const float* z2, const int size) { for (int i = size; i > 0; i--, r++, z1++, z2++) *r = *z1 + *z2; }
void __fastcall op_2(float* r, const float* z1, const float* z2, const int size) { for (int i = size; i > 0; i--, r++, z1++, z2++) *r = *z1 - *z2; }
void __fastcall op_3(float* r, const float* z1, const float* z2, const int size) { for (int i = size; i > 0; i--, r++, z1++, z2++) *r = *z2 - *z1; }
void __fastcall op_4(float* r, const float* z1, const float* z2, const int size) { for (int i = size; i > 0; i--, r++, z1++, z2++) *r = *z1 * *z2; }
void __fastcall op_5(float* r, const float* z1, const float* z2, const int size) { for (int i = size; i > 0; i--, r++, z1++, z2++) if (*z2 != 0.0) *r = *z1 / *z2; else *r = big; }
void __fastcall op_6(float* r, const float* z1, const float* z2, const int size) { for (int i = size; i > 0; i--, r++, z1++, z2++) if (*z1 != 0.0) *r = *z2 / *z1; else *r = big; }
void __fastcall op_7(float* r, const float* z1, const float* z2, const int size) { for (int i = size; i > 0; i--, r++, z1++, z2++) *r = *z2 * *z2 + *z1; }
void __fastcall op_8(float* r, const float* z1, const float* z2, const int size) { for (int i = size; i > 0; i--, r++, z1++, z2++) *r = *z1 * *z1 + *z2; }
void __fastcall op_9(float* r, const float* z1, const float* z2, const int size) { for (int i = size; i > 0; i--, r++, z1++, z2++) *r = *z2 * *z2 - *z1; }
void __fastcall op_10(float* r, const float* z1, const float* z2, const int size) { for (int i = size; i > 0; i--, r++, z1++, z2++) *r = *z1 * *z1 - *z2; }
void __fastcall op_11(float* r, const float* z1, const float* z2, const int size) { for (int i = size; i > 0; i--, r++, z1++, z2++) *r = *z1 * *z2 / (*z1 + *z2); }

oper	op[] = {
	op_1,	//	655
	op_2,	//	841
	op_3,	//	733
	op_4,	//	751
	//op_5,	//	721
	//op_6,	//	1038
	//op_7,	//	1122
	//op_8,	//	1094
	//op_9,	//	1172
	//op_10,	//	984
	//op_11,	//	1423
};
const int op_count = sizeof(op) / sizeof(oper);
//oper	op[] = { op_1,op_2,op_3,op_4,op_5,op_6 };
/*
953
*/

// Get operation index from function pointer (for serialization)
int getOpIndex(oper operation) {
	for (int i = 0; i < op_count; i++) {
		if (op[i] == operation) return i;
	}
	return 0;  // Default to first operation if not found
}
//////////////////////////////////////////////////////////////////////////////

/*
Каждая подсеть имеет фиксированный набор входов, которые подключены к другим подсетям
и имеет неограниченный набор выходов число которых увеличивается в следствии самообучения.
Выход одной подсети поединён со входами другой след образом: когда одна подсеть опознаёт образ
(явный выраженный сигнал только на одном выходе) то номер выхода опознанного образа либо
последовательность номеров является уникальным образом для входов другой подсети.
*/
class Neiron
{
public:
	int		i;                                       /* номер первого входного нейрона */
	int		j;                                       /* номер второго входного нейрона */
	oper	op;								         /* элементарное действие */
	vector<float>	c;								 /* кэш значений для каждого образа */
	bool	cached;
	float	val;
	bool	val_cached;

	Neiron() : i(0), j(0), op(nullptr), cached(false), val(0), val_cached(false) {}
};

vector<Neiron>	nei;									/* нейроны, способные родиться  */
const int MAX_NEURONS = 64000;

// Initialize neurons with proper vector sizes
void initNeurons() {
	nei.resize(MAX_NEURONS);
	for (int n = 0; n < MAX_NEURONS; n++) {
		nei[n].c.resize(Images);
		nei[n].cached = false;
		nei[n].val_cached = false;
	}
}

// Save trained neural network to JSON file
bool saveNetwork(const string& filePath) {
	try {
		json network;

		// Store configuration
		network["receptors"] = Receptors;
		network["base_size"] = base_size;
		network["inputs"] = Inputs;
		network["neurons_count"] = Neirons;

		// Store basis values
		json basisArray = json::array();
		for (int i = 0; i < base_size; i++) {
			basisArray.push_back(base[i]);
		}
		network["basis"] = basisArray;

		// Store class names
		json classesArray = json::array();
		for (int c = 0; c < Classes; c++) {
			json cls;
			cls["id"] = c;
			cls["name"] = classes[c];
			cls["output_neuron"] = NetOutput[c];
			classesArray.push_back(cls);
		}
		network["classes"] = classesArray;

		// Store neurons (only the ones above Inputs, since 0..Inputs-1 are inputs)
		json neuronsArray = json::array();
		for (int n = Inputs; n < Neirons; n++) {
			json neuron;
			neuron["id"] = n;
			neuron["i"] = nei[n].i;
			neuron["j"] = nei[n].j;
			neuron["op"] = getOpIndex(nei[n].op);
			neuronsArray.push_back(neuron);
		}
		network["neurons"] = neuronsArray;

		// Add metadata
		network["version"] = "1.0";
		network["description"] = "Trained neural network model";

		// Write to file with pretty printing
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

// Load trained neural network from JSON file (for inference mode)
bool loadNetwork(const string& filePath) {
	ifstream inFile(filePath);
	if (!inFile.is_open()) {
		cerr << "Error: Cannot open network file: " << filePath << endl;
		return false;
	}

	try {
		json network;
		inFile >> network;

		// Load configuration
		Receptors = network["receptors"].get<int>();
		Inputs = network["inputs"].get<int>();
		Neirons = network["neurons_count"].get<int>();

		// Verify basis size matches
		int loadedBaseSize = network["base_size"].get<int>();
		if (loadedBaseSize != base_size) {
			cerr << "Warning: Basis size mismatch (file: " << loadedBaseSize
				 << ", expected: " << base_size << ")" << endl;
		}

		// Allocate network input array
		NetInput.resize(Inputs);

		// Set basis values
		for (int i = 0; i < base_size; i++) {
			NetInput[i + Receptors] = base[i];
		}

		// Load classes
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

		// Initialize neurons (without image cache since we don't need training data)
		nei.resize(MAX_NEURONS);
		for (int n = 0; n < MAX_NEURONS; n++) {
			nei[n].cached = false;
			nei[n].val_cached = false;
		}

		// Load neuron structure
		const auto& neuronsArray = network["neurons"];
		for (const auto& neuron : neuronsArray) {
			int id = neuron["id"].get<int>();
			nei[id].i = neuron["i"].get<int>();
			nei[id].j = neuron["j"].get<int>();
			int opIndex = neuron["op"].get<int>();
			if (opIndex >= 0 && opIndex < op_count) {
				nei[id].op = op[opIndex];
			} else {
				nei[id].op = op[0];  // Default to first operation
			}
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

void	Print(int i)
{
	/*if( i < Inputs ) cout << "x" << i;
	else
	{
		if( nei[i-Inputs].i >= i || nei[i-Inputs].i >= i )
		{
			cout << "Error";
			return;
		}
		char	str[2];
		str[0] = nei[i-Inputs].op;
		str[1] = 0;
		cout << "(";
		Print(nei[i-Inputs].i);
		cout << " " << str << " ";
		Print(nei[i-Inputs].j);
		cout << ")";
	}*/
}


float* __fastcall GetNeironVector(const int i)                   /* расчет вектора значений, выдаваемого i нейроном*/
{
	Neiron&	current = nei[i];
	if (!current.cached)
	{
		current.cached = true;
		if (i < Receptors)
		{
			for (int im = 0; im < Images; im++)
				current.c[im] = vx[im][i];
		}
		else if (i < Inputs)
		{
			for (int im = 0; im < Images; im++)
				current.c[im] = NetInput[i];
		}
		else
		{
			float*	icache = GetNeironVector(current.i);
			float*	jcache = GetNeironVector(current.j);
			(*current.op)(current.c.data(), icache, jcache, Images);
		}
	}

	return current.c.data();
}


float __fastcall GetNeironVal(const int i)                   /* расчет значения, выдаваемого i нейроном*/
{
	if (i < Inputs)
	{
		return NetInput[i];
	}
	else
	{
		Neiron&	current = nei[i];
		if (current.val_cached)
			return current.val;
		else
		{
			float ival = GetNeironVal(current.i);
			float jval = GetNeironVal(current.j);
			(*current.op)(&current.val, &ival, &jval, 1);
			current.val_cached = true;
			return current.val;
		}
	}
}

void	clear_val_cache(vector<Neiron>& n, const int size)
{
	int limit = (size < (int)n.size()) ? size : (int)n.size();
	for (int i = 0; i < limit; i++)
		n[i].val_cached = false;
}

// Perform classification on input text and return results
void classifyInput(const string& inputText, bool verbose = true) {
	// Set network input from text
	for (int d = 0; d < Receptors; d++) {
		if (d < (int)inputText.length() && inputText[d] != 0) {
			NetInput[d] = float((unsigned char)inputText[d]) / float(max_num);
		} else {
			NetInput[d] = float((unsigned char)' ') / float(max_num);
		}
	}

	// Clear value cache
	clear_val_cache(nei, MAX_NEURONS);

	// Calculate and display results for each class
	if (verbose) {
		for (int out = 0; out < Classes; out++) {
			float z1 = GetNeironVal(NetOutput[out]) * 100.0f;
			// Handle NaN and infinite values
			if (!std::isfinite(z1)) z1 = 0.0f;
			if (z1 < 0.0f) z1 = 0.0f;
			if (z1 > 100.0f) z1 = 100.0f;
			cout << long(z1) << "%" << " - " << classes[out] << endl;
		}
	}
}


float	rod()                    /* подпрограмма для рождения новых элементов */
{                                /* количество рожденных нейронов (Neirons++) */
	//	тупой перебор половины всех комбинаций каждого с каждым
	int		i;
	float	min = big;
	int		optimal_i = 0;
	int		optimal_j = 0;
	oper	optimal_op = op[0];
	float	square, sum;
	Neiron&	cur = nei[Neirons];
	float*	curval;

	for (cur.i = 1; cur.i < Neirons; cur.i++)               // выбор 1-го нейрона 
	{
		for (cur.j = 0; cur.j < cur.i; cur.j++)              // выбор 2-го - нейрона
		{
			for (i = 0; i < sizeof(op) / sizeof(oper); i++)/*выбор операции*/
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

float	rod2()                    /* подпрограмма для рождения новых элементов */
{                                /* количество рожденных нейронов (Neirons++) */
	int		i;
	float	min = big;
	int		optimal_i = 0;
	int		optimal_j = 0;
	oper	optimal_op = op[0];
	float	square, sum;
	Neiron&	cur = nei[Neirons];
	float*	curval;

	cur.i = Neirons - 1;               // выбор последнего нейрона 

	for (cur.j = 0; cur.j < cur.i; cur.j++)              // выбор 2-го - нейрона
	{
		for (i = 0; i < sizeof(op) / sizeof(oper); i++)/*выбор операции*/
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

float	rod3()                    /* подпрограмма для рождения новых элементов */
{                                /* количество рожденных нейронов (Neirons++) */
								 //	тупой перебор половины всех комбинаций каждого с каждым
	int		i;
	float	min = big;
	int		optimal_i = 0;
	int		optimal_j = 0;
	oper	optimal_op = op[0];
	float	square, sum;
	Neiron&	cur = nei[Neirons];
	float*	curval;

	for (cur.i = 0; cur.i < Neirons - Classes * 3; cur.i++)               // выбор 1-го нейрона 
	{
		for (cur.j = Neirons - Classes * 3; cur.j < Neirons; cur.j++)              // выбор 2-го - нейрона
		{
			for (i = 0; i < sizeof(op) / sizeof(oper); i++)/*выбор операции*/
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

void	rndrod(unsigned count) /* подпрограмма для рождения новых элементов */
{                                /* количество рожденных нейронов (Neirons++) */
	do
	{
		nei[Neirons].cached = false;
		nei[Neirons].i = rand() % (Neirons);
		nei[Neirons].j = rand() % (Neirons);
		nei[Neirons].op = op[rand() % (sizeof(op) / sizeof(oper))];
		cout << "(" << Neirons << ") = (" << nei[Neirons].i << ")op(" << nei[Neirons].j << ")\n";
		Neirons++;
	} while (--count > 0);
}

void	rndrod0(unsigned count) /* подпрограмма для рождения новых элементов */
{                                /* количество рожденных нейронов (Neirons++) */
	do
	{
		nei[Neirons].cached = false;
		nei[Neirons].i = rand() % (Inputs);
		nei[Neirons].j = rand() % (Receptors);
		nei[Neirons].op = op[rand() % (sizeof(op) / sizeof(oper))];
		cout << "(" << Neirons << ") = (" << nei[Neirons].i << ")op(" << nei[Neirons].j << ")\n";
		Neirons++;
	} while (--count > 0);
}


float	rndrod2()                    /* подпрограмма для рождения новых элементов */
{                                /* количество рожденных нейронов (Neirons+2) */
	int		count, count_max = Inputs * Neirons * rndrod_iter;
	float	min = big;
	float	square, sum;
	int		r[5] = { 0,0,0,0,0 };
	oper	ro[5] = { 0,0,0,0,0 };
	int		Neirons_p_1 = Neirons + 1;
	Neiron&	Neiron_A = nei[Neirons];
	Neiron&	Neiron_B = nei[Neirons_p_1];

	Neiron_B.i = Neirons;

	for (count = 0; count < count_max; count++)
	{
		Neiron_A.cached = false;
		Neiron_A.i = rand() % rndrod_iter + Neirons - rndrod_iter; // последние случайные
		Neiron_A.j = rand() % (Neirons - rndrod_iter);
		Neiron_A.op = op[rand() % (sizeof(op) / sizeof(oper))];

		Neiron_B.cached = false;
		Neiron_B.j = rand() % Inputs;
		Neiron_B.op = op[rand() % (sizeof(op) / sizeof(oper))];

		float*	NBVal = GetNeironVector(Neirons_p_1);

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


float	rndrod3()                    /* подпрограмма для рождения новых элементов */
{                                /* количество рожденных нейронов (Neirons+2) */
	int		count, count_max = Neirons * Neirons * 6;
	float	min = big;
	float	square, sum;
	int		r[5] = { 0,0,0,0,0 };
	oper	ro[5] = { 0,0,0,0,0 };
	int		Neirons_p_1 = Neirons + 1;
	Neiron&	Neiron_A = nei[Neirons];
	Neiron&	Neiron_B = nei[Neirons_p_1];

	Neiron_B.i = Neirons;

	for (count = 0; count < count_max; count++)
	{
		Neiron_A.cached = false;
		Neiron_A.i = rand() % Neirons;
		Neiron_A.j = rand() % Neirons;
		Neiron_A.op = op[rand() % (sizeof(op) / sizeof(oper))];

		Neiron_B.cached = false;
		Neiron_B.j = rand() % Neirons;
		Neiron_B.op = op[rand() % (sizeof(op) / sizeof(oper))];

		float*	NBVal = GetNeironVector(Neirons_p_1);

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


float	rndrod4()                    /* подпрограмма для рождения новых элементов */
{                                /* количество рожденных нейронов (Neirons+2) */
	int		count, count_max = Neirons * Receptors * 4;// / 4;// *10;
	float	min = big;
	float	square, sum;
	int		A_id = Neirons;
	int		B_id = Neirons + 1;
	int		C_id = Neirons + 2;
	Neiron&	Neiron_A = nei[A_id];
	Neiron&	Neiron_B = nei[B_id];
	Neiron&	Neiron_C = nei[C_id];
	Neiron	optimal_A, optimal_B, optimal_C;
	bool	finded = false;

	Neiron_C.i = A_id;
	Neiron_C.j = B_id;

	Neiron_A.i = rand() % Neirons;
	Neiron_A.j = rand() % Neirons;
	Neiron_A.op = op[rand() % (sizeof(op) / sizeof(oper))];
	Neiron_A.cached = false;

	for (count = 0; count < count_max; count++)
	{
		Neiron_B.i = rand() % Neirons;
		Neiron_B.j = rand() % Neirons;

		for (int B_op = 0; B_op < sizeof(op) / sizeof(oper); B_op++)
		{
			Neiron_B.op = op[B_op];
			Neiron_B.cached = false;
			//Neiron_B.op = op[rand() % (sizeof(op) / sizeof(oper))];

			for (int C_op = 0; C_op < sizeof(op) / sizeof(oper); C_op++)
			{
				Neiron_C.op = op[C_op];
				Neiron_C.cached = false;
				//Neiron_C.op = op[rand() % (sizeof(op) / sizeof(oper))];
				float*	C_Vector = GetNeironVector(C_id);
				sum = 0.0;

				// Calculate error across all images
				// vz[index] contains expected value for image index
				// (1.0 if image belongs to current class, 0.0 otherwise)
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

					Neiron_A = Neiron_B;	//	используем оптимальный нейрон
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


int	readkeyboard(char* str)
{
	char	one;
	int		i = 0;

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


bool	cmp(char* str)
{
	return strcmp(str, word_buf) == 0;
}

float	sum(const float* ar, const int size)
{
	float	res = 0;
	for (int i = 0; i < size; i++) res += ar[i];
	return res;
}

int	main(int argc, char* argv[])
{
	// Parse command line arguments
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
		} else if (arg == "-h" || arg == "--help") {
			printUsage(argv[0]);
			return 0;
		}
	}

	// ===== INFERENCE MODE =====
	// If loading a trained network, skip training and go straight to inference
	if (inferenceMode) {
		if (!loadNetwork(loadPath)) {
			return 1;
		}

		// If single input text provided, classify and exit
		if (!inputText.empty()) {
			cout << "\nClassifying: \"" << inputText << "\"" << endl;
			classifyInput(inputText);
			return 0;
		}

		// Otherwise, enter interactive inference mode
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

	// ===== TRAINING MODE =====

	// Load configuration or use defaults
	if (!configPath.empty()) {
		if (!loadConfig(configPath, Receptors)) {
			return 1;
		}
	} else {
		initDefaultConfig(Receptors);
	}

	cout << "Random seed: " << rand() << endl;

	// Calculate derived values after configuration is loaded
	Images = const_words.size();
	Inputs = Receptors + base_size;
	Neirons = Inputs;

	// Allocate dynamic arrays
	NetInput.resize(Inputs);
	vx.resize(Images);
	for (int i = 0; i < Images; i++) {
		vx[i].resize(Receptors);
	}
	vz.resize(Classes);
	NetOutput.resize(Classes);

	// Initialize neurons
	initNeurons();

	// Зададим базис
	for (int i = 0; i < base_size; i++)
		NetInput[i + Receptors] = base[i];

	// генерируем образы из слов
	for (int index = 0; index < Images; index++)
	{
		memset(word_buf, 0, StringSize);
		strcpy_s(word_buf, const_words[index].word.c_str());
		cout << "img:" << const_words[index].word << endl;

		for (int d = 0; d < Receptors; d++)
		{
			if (word_buf[d] == 0)
			{
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
	// Track error per class (not per image) - each class can have multiple images
	vector<float> class_er(Classes, big);
	float er = .01f; /* допустимая ошибка в %*/

	//rndrod0(Inputs*Receptors);

	// Start training timer for benchmark
	auto trainingStartTime = chrono::high_resolution_clock::now();
	int trainingIterations = 0;

	do
	{
		trainingIterations++;
		cout << "train class:" << classes[classIndex] << " (id=" << classIndex << ")";

		// Set expected output vector: 1.0 for images of current class, 0.0 for others
		// vz[image_index] = expected value for that image based on training to recognize classIndex
		vz.resize(Images);  // Resize vz to number of images
		for (int img = 0; img < Images; img++)
		{
			if (const_words[img].id == classIndex)
				vz[img] = 1.0;  // This image belongs to the class we're training
			else
				vz[img] = 0.0;  // This image does NOT belong to the class
		}

		// Train to recognize current class
		if (class_er[classIndex] > er)
		{/*
			// среди существующих нейронов невозможно найти решение
			// создаём слой случайных нейронов
			rndrod(rndrod_iter);

			// поверх создаём слой 3х входовых случайных оптимизированных нейронов
			// ищём при этом подходящий нейрон для опознания образа
			class_er[classIndex] = rndrod2();

			int iter = rndrod2_iter - 1;
			while (iter-- > 0 && class_er[classIndex] > er)
				class_er[classIndex] = rndrod2();

			if (class_er[classIndex] > er)
			{
				// пытаемся найти подходящий выходной нейрон для текущего класса
				class_er[classIndex] = rod();

				// создаём дополнительные нейроны до тех пор пока не появится подходящий
				float old_er;
				iter = rod2_iter - 1;
				while (iter-- > 0 && class_er[classIndex] > er)
				{
					old_er = class_er[classIndex];
					class_er[classIndex] = rod2();
					// если ошибка не уменьшилась то нет смысла создавать новые нейроны данным методом
					if (old_er == class_er[classIndex]) break;
				}
			}
			*/

			class_er[classIndex] = rndrod4();
			//class_er[classIndex] = rod();
			//class_er[classIndex] = rod2();
			//class_er[classIndex] = rod3();

			// Store the output neuron for this class
			NetOutput[classIndex] = Neirons - 1;
		}

		cout << ", n" << NetOutput[classIndex] << " = " << class_er[classIndex] << endl;

		if (++classIndex >= Classes)	//	идём по кругу
			classIndex = 0;

	} while (sum(class_er.data(), Classes) > Classes * er);

	// End training timer
	auto trainingEndTime = chrono::high_resolution_clock::now();
	auto trainingDuration = chrono::duration_cast<chrono::milliseconds>(trainingEndTime - trainingStartTime);

	cout << "\nTraining completed!" << endl;
	cout << "Final errors per class:" << endl;
	for (int c = 0; c < Classes; c++) {
		cout << "  Class " << c << " (" << classes[c] << "): error = " << class_er[c] << endl;
	}

	// Save trained network if requested
	if (!savePath.empty()) {
		if (!saveNetwork(savePath)) {
			cerr << "Warning: Failed to save network to " << savePath << endl;
		}
	}

	// Benchmark mode: output training speed metrics
	if (benchmarkMode) {
		cout << "\n=== Training Speed Benchmark Results ===" << endl;
		cout << "Configuration:" << endl;
		cout << "  Receptors (inputs): " << Receptors << endl;
		cout << "  Classes: " << Classes << endl;
		cout << "  Images: " << Images << endl;
		cout << "  Neurons created: " << (Neirons - Inputs) << endl;
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

		// In benchmark mode, return success unless there was an error
		return 0;
	}

	// Test mode: validate classification accuracy and exit
	if (testMode) {
		cout << "\n=== Running automated classification test ===" << endl;
		int passed = 0;
		int failed = 0;
		float threshold = 0.5f;  // Classification threshold

		for (int img = 0; img < Images; img++) {
			// Set network input for this image
			for (int d = 0; d < Receptors; d++) {
				NetInput[d] = vx[img][d];
			}
			clear_val_cache(nei, MAX_NEURONS);

			// Find the class with highest output
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

			// Test passes if:
			// 1. Predicted class matches expected class, OR
			// 2. Expected class output is above threshold
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

		// Exit with appropriate code
		if (failed == 0) {
			cout << "\nAll tests PASSED!" << endl;
			return 0;
		} else {
			cout << "\nSome tests FAILED!" << endl;
			return 1;
		}
	}

	// Interactive mode
	do
	{
		/* считывается новая порция данных и подается на вход/вых.*/
		if (InputStr[0] == 0)
		{
			cout << "input word:";
			readkeyboard(InputStr);
		}

		memset(word_buf, 0, StringSize);
		strcpy_s(word_buf, InputStr);
		memset(InputStr, 0, StringSize);

		if (cmp("Q") || cmp("q"))	return 0;

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

		// выводим состояние выходов нейросети
		for (int out = 0; out < Classes; out++)
		{
			float z1 = GetNeironVal(NetOutput[out])*100.0f;
			// Handle NaN and infinite values that cause incorrect percentage display
			if (!std::isfinite(z1)) z1 = 0.0f;
			if (z1 < 0.0f) z1 = 0.0f;
			if (z1 > 100.0f) z1 = 100.0f;
			cout << long(z1) << "%" << " - " << classes[out] << endl; // расчет результата
		}

	} while (true);
}

