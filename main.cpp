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

		// Check if custom images are provided directly
		if (config.contains("images")) {
			// Load images directly from JSON
			const_words.clear();
			for (const auto& img : config["images"]) {
				string word = img["word"].get<string>();
				int id = img["id"].get<int>();
				const_words.push_back({word, id});
				if (id >= Classes) {
					Classes = id + 1;
				}
			}
		}
		// Otherwise generate from classes
		else if (config.contains("classes")) {
			const_words.clear();
			Classes = config["classes"].size();
			bool generateShifts = true;
			if (config.contains("generate_shifts")) {
				generateShifts = config["generate_shifts"].get<bool>();
			}

			for (const auto& cls : config["classes"]) {
				string word = cls["word"].get<string>();
				int id = cls["id"].get<int>();

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

	// Generate empty class
	string empty(receptors, ' ');
	const_words.push_back({empty, 0});

	// Generate shifted images for each word
	generateShiftedImages("time", 1, receptors);
	generateShiftedImages("hour", 2, receptors);
	generateShiftedImages("main", 3, receptors);

	cout << "Using default configuration" << endl;
	cout << "  Receptors: " << receptors << endl;
	cout << "  Classes: " << Classes << endl;
	cout << "  Images: " << const_words.size() << endl;
}

// Print usage information
void printUsage(const char* programName) {
	cout << "Usage: " << programName << " [options]" << endl;
	cout << "Options:" << endl;
	cout << "  -c, --config <file>  Load configuration from JSON file" << endl;
	cout << "  -h, --help           Show this help message" << endl;
	cout << endl;
	cout << "JSON config format:" << endl;
	cout << "  {" << endl;
	cout << "    \"receptors\": 20,  // Number of neural network inputs" << endl;
	cout << "    \"classes\": [" << endl;
	cout << "      { \"id\": 0, \"word\": \"\" }," << endl;
	cout << "      { \"id\": 1, \"word\": \"time\" }," << endl;
	cout << "      ..." << endl;
	cout << "    ]," << endl;
	cout << "    \"generate_shifts\": true  // Generate position-shifted variants" << endl;
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
//oper	op[] = { op_1,op_2,op_3,op_4,op_5,op_6 };
/*
953
*/
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
	for (int i = 1; i < argc; i++) {
		string arg = argv[i];
		if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
			configPath = argv[++i];
		} else if (arg == "-h" || arg == "--help") {
			printUsage(argv[0]);
			return 0;
		}
	}

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
	vz.resize(Images);
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

		cout << "img:";
		for (int d = 0; d < Receptors; d++)
		{
			if (word_buf[d] == 0)
			{
				for (; d < Receptors; d++)
				{
					vx[index][d] = float((unsigned char)' ') / float(max_num);
					cout << " ";
				}

				break;
			}
			else
			{
				vx[index][d] = float((unsigned char)word_buf[d]) / float(max_num);
				cout << word_buf[d];
			}
		}
		cout << endl;
	}

	int iter, Class = 0;
	vector<float> cur_er(Classes, big);
	float old_er, er = .01f; /* допустимая ошибка в %*/

	//rndrod0(Inputs*Receptors);

	do
	{
		cout << "train to:" << const_words[Class].word;

		// задаём вектор выходных значений распознавания текущего образа
		for (int index = 0; index < Images; index++)
		{
			if (const_words[index].id == Class) vz[index] = 1.0;
			else vz[index] = 0.0;
		}

		// обучаемся распознавать текущий образ
		if (cur_er[Class] > er)
		{/*
			// среди существующих нейронов невозможно найти решение
			// создаём слой случайных нейронов
			rndrod(rndrod_iter);

			// поверх создаём слой 3х входовых случайных оптимизированных нейронов
			// ищём при этом подходящий нейрон для опознания образа
			cur_er[Class] = rndrod2();

			iter = rndrod2_iter - 1;
			while (iter-- > 0 && cur_er[Class] > er)
				cur_er[Class] = rndrod2();

			if (cur_er[Class] > er)
			{
				// пытаемся найти подходящий выходной нейрон для текущего образа  с номером image
				cur_er[Class] = rod();

				// создаём дополнительные нейроны до тех пор пока не появится подходящий
				iter = rod2_iter - 1;
				while (iter-- > 0 && cur_er[Class] > er)
				{
					old_er = cur_er[Class];
					cur_er[Class] = rod2();
					// если ошибка не уменьшилась то нет смысла создавать новые нейроны данным методом
					if (old_er == cur_er[Class]) break;
				}
			}
			*/

			cur_er[Class] = rndrod4();
			//cur_er[Class] = rod();
			//cur_er[Class] = rod2();
			//cur_er[Class] = rod3();

			// запоминаем номер выходного нейрона который опознаёт текущий образ
			NetOutput[Class] = Neirons - 1;
		}

		cout << ", n" << NetOutput[Class] << " = " << cur_er[Class] << endl;

		if (++Class >= Classes)	//	идём по кругу
			Class = 0;

	} while (sum(cur_er.data(), Classes) > Classes * er);


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
			cout << long(z1) << "%" << " - " << const_words[out].word << endl; // расчет результата
		}

	} while (true);
}

