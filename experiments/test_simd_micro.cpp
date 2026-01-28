/*
 * test_simd_micro.cpp - Микро-бенчмарк для тестирования SIMD операций
 *
 * Тестирует производительность SIMD-оптимизированных векторных операций
 * на больших массивах для измерения реального ускорения.
 *
 * Компиляция:
 *   g++ -O3 -march=native -o test_simd_micro test_simd_micro.cpp
 *
 * Запуск:
 *   ./test_simd_micro
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <iomanip>

// Глобальный флаг для SIMD (требуется для simd_ops.h)
bool UseSIMD = true;

#include "../include/simd_ops.h"

using namespace std;
using namespace std::chrono;

// Количество повторений для усреднения результатов
const int ITERATIONS = 1000;

// Размеры массивов для тестирования
const int SIZES[] = {16, 48, 100, 256, 1000, 10000};
const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

// Инициализация массива случайными значениями
void initRandom(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

// Измерение времени выполнения операции сложения
double benchmarkAdd(float* r, const float* a, const float* b, int size, bool useSIMD) {
    UseSIMD = useSIMD;
    auto start = high_resolution_clock::now();

    for (int iter = 0; iter < ITERATIONS; iter++) {
        op_add_simd(r, a, b, size);
    }

    auto end = high_resolution_clock::now();
    return duration_cast<nanoseconds>(end - start).count() / (double)ITERATIONS;
}

// Измерение времени выполнения операции вычитания
double benchmarkSub(float* r, const float* a, const float* b, int size, bool useSIMD) {
    UseSIMD = useSIMD;
    auto start = high_resolution_clock::now();

    for (int iter = 0; iter < ITERATIONS; iter++) {
        op_sub_simd(r, a, b, size);
    }

    auto end = high_resolution_clock::now();
    return duration_cast<nanoseconds>(end - start).count() / (double)ITERATIONS;
}

// Измерение времени выполнения операции умножения
double benchmarkMul(float* r, const float* a, const float* b, int size, bool useSIMD) {
    UseSIMD = useSIMD;
    auto start = high_resolution_clock::now();

    for (int iter = 0; iter < ITERATIONS; iter++) {
        op_mul_simd(r, a, b, size);
    }

    auto end = high_resolution_clock::now();
    return duration_cast<nanoseconds>(end - start).count() / (double)ITERATIONS;
}

// Проверка корректности результатов
bool verifyResults(const float* r_simd, const float* r_scalar, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(r_simd[i] - r_scalar[i]) > 1e-6) {
            return false;
        }
    }
    return true;
}

int main() {
    cout << "=== SIMD Micro-Benchmark ===" << endl;
    cout << "SIMD Extension: " << getSIMDInfo() << endl;
    cout << "Iterations per test: " << ITERATIONS << endl;
    cout << endl;

    srand(42);  // Фиксированный seed для воспроизводимости

    cout << fixed << setprecision(2);
    cout << "| Size    | Op   | Scalar (ns) | SIMD (ns) | Speedup |" << endl;
    cout << "|---------|------|-------------|-----------|---------|" << endl;

    for (int s = 0; s < NUM_SIZES; s++) {
        int size = SIZES[s];

        // Выделяем массивы
        vector<float> a(size), b(size), r_simd(size), r_scalar(size);

        // Инициализируем случайными значениями
        initRandom(a.data(), size);
        initRandom(b.data(), size);

        // Тест сложения
        double scalarAddTime = benchmarkAdd(r_scalar.data(), a.data(), b.data(), size, false);
        double simdAddTime = benchmarkAdd(r_simd.data(), a.data(), b.data(), size, true);
        double addSpeedup = scalarAddTime / simdAddTime;

        if (!verifyResults(r_simd.data(), r_scalar.data(), size)) {
            cout << "ERROR: Add verification failed for size " << size << endl;
        }

        cout << "| " << setw(7) << size << " | ADD  | " << setw(11) << scalarAddTime
             << " | " << setw(9) << simdAddTime << " | " << setw(6) << addSpeedup << "x |" << endl;

        // Тест вычитания
        double scalarSubTime = benchmarkSub(r_scalar.data(), a.data(), b.data(), size, false);
        double simdSubTime = benchmarkSub(r_simd.data(), a.data(), b.data(), size, true);
        double subSpeedup = scalarSubTime / simdSubTime;

        if (!verifyResults(r_simd.data(), r_scalar.data(), size)) {
            cout << "ERROR: Sub verification failed for size " << size << endl;
        }

        cout << "| " << setw(7) << size << " | SUB  | " << setw(11) << scalarSubTime
             << " | " << setw(9) << simdSubTime << " | " << setw(6) << subSpeedup << "x |" << endl;

        // Тест умножения
        double scalarMulTime = benchmarkMul(r_scalar.data(), a.data(), b.data(), size, false);
        double simdMulTime = benchmarkMul(r_simd.data(), a.data(), b.data(), size, true);
        double mulSpeedup = scalarMulTime / simdMulTime;

        if (!verifyResults(r_simd.data(), r_scalar.data(), size)) {
            cout << "ERROR: Mul verification failed for size " << size << endl;
        }

        cout << "| " << setw(7) << size << " | MUL  | " << setw(11) << scalarMulTime
             << " | " << setw(9) << simdMulTime << " | " << setw(6) << mulSpeedup << "x |" << endl;
    }

    cout << endl;
    cout << "=== End Micro-Benchmark ===" << endl;

    return 0;
}
