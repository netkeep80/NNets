/*
 * simd_ops.h - SIMD-оптимизированные векторные операции
 *
 * Этот модуль содержит SIMD-версии базовых векторных операций:
 * - Сумма (op_1)
 * - Разность z1 - z2 (op_2)
 * - Разность z2 - z1 (op_3)
 * - Произведение (op_4)
 *
 * Поддерживаемые наборы инструкций (в порядке приоритета):
 * - AVX (256-bit, 8 float за раз)
 * - SSE (128-bit, 4 float за раз)
 * - Скалярный код (fallback)
 *
 * Примечание: Для включения SIMD оптимизаций необходимо
 * компилировать с флагами -mavx/-msse или использовать -march=native
 */

#ifndef SIMD_OPS_H
#define SIMD_OPS_H

// Определяем доступность SIMD инструкций на этапе компиляции
// AVX проверяется первым, так как он более эффективен
#if defined(__AVX__) || defined(__AVX2__)
    #define SIMD_AVX_ENABLED 1
    #include <immintrin.h>
#elif defined(__SSE__) || defined(__SSE2__) || defined(__SSE4_1__)
    #define SIMD_SSE_ENABLED 1
    #include <xmmintrin.h>
    #include <emmintrin.h>
#endif

// Флаг для включения/выключения SIMD во время выполнения
// Позволяет тестировать производительность с/без SIMD
extern bool UseSIMD;

// ============================================================================
// Скалярные версии операций (базовые, без оптимизаций)
// ============================================================================

/**
 * Скалярная сумма векторов: r[i] = z1[i] + z2[i]
 */
inline void op_add_scalar(float* r, const float* z1, const float* z2, const int size) {
    for (int i = 0; i < size; i++) {
        r[i] = z1[i] + z2[i];
    }
}

/**
 * Скалярная разность векторов: r[i] = z1[i] - z2[i]
 */
inline void op_sub_scalar(float* r, const float* z1, const float* z2, const int size) {
    for (int i = 0; i < size; i++) {
        r[i] = z1[i] - z2[i];
    }
}

/**
 * Скалярная обратная разность векторов: r[i] = z2[i] - z1[i]
 */
inline void op_rsub_scalar(float* r, const float* z1, const float* z2, const int size) {
    for (int i = 0; i < size; i++) {
        r[i] = z2[i] - z1[i];
    }
}

/**
 * Скалярное произведение векторов: r[i] = z1[i] * z2[i]
 */
inline void op_mul_scalar(float* r, const float* z1, const float* z2, const int size) {
    for (int i = 0; i < size; i++) {
        r[i] = z1[i] * z2[i];
    }
}

// ============================================================================
// AVX версии операций (256-bit, 8 float за итерацию)
// ============================================================================

#ifdef SIMD_AVX_ENABLED

/**
 * AVX сумма векторов: r[i] = z1[i] + z2[i]
 * Обрабатывает 8 элементов за итерацию
 */
inline void op_add_avx(float* r, const float* z1, const float* z2, const int size) {
    int i = 0;

    // Обработка по 8 элементов (256 бит)
    for (; i <= size - 8; i += 8) {
        __m256 v1 = _mm256_loadu_ps(z1 + i);
        __m256 v2 = _mm256_loadu_ps(z2 + i);
        __m256 result = _mm256_add_ps(v1, v2);
        _mm256_storeu_ps(r + i, result);
    }

    // Обработка оставшихся элементов скалярно
    for (; i < size; i++) {
        r[i] = z1[i] + z2[i];
    }
}

/**
 * AVX разность векторов: r[i] = z1[i] - z2[i]
 */
inline void op_sub_avx(float* r, const float* z1, const float* z2, const int size) {
    int i = 0;

    for (; i <= size - 8; i += 8) {
        __m256 v1 = _mm256_loadu_ps(z1 + i);
        __m256 v2 = _mm256_loadu_ps(z2 + i);
        __m256 result = _mm256_sub_ps(v1, v2);
        _mm256_storeu_ps(r + i, result);
    }

    for (; i < size; i++) {
        r[i] = z1[i] - z2[i];
    }
}

/**
 * AVX обратная разность векторов: r[i] = z2[i] - z1[i]
 */
inline void op_rsub_avx(float* r, const float* z1, const float* z2, const int size) {
    int i = 0;

    for (; i <= size - 8; i += 8) {
        __m256 v1 = _mm256_loadu_ps(z1 + i);
        __m256 v2 = _mm256_loadu_ps(z2 + i);
        __m256 result = _mm256_sub_ps(v2, v1);
        _mm256_storeu_ps(r + i, result);
    }

    for (; i < size; i++) {
        r[i] = z2[i] - z1[i];
    }
}

/**
 * AVX произведение векторов: r[i] = z1[i] * z2[i]
 */
inline void op_mul_avx(float* r, const float* z1, const float* z2, const int size) {
    int i = 0;

    for (; i <= size - 8; i += 8) {
        __m256 v1 = _mm256_loadu_ps(z1 + i);
        __m256 v2 = _mm256_loadu_ps(z2 + i);
        __m256 result = _mm256_mul_ps(v1, v2);
        _mm256_storeu_ps(r + i, result);
    }

    for (; i < size; i++) {
        r[i] = z1[i] * z2[i];
    }
}

#endif // SIMD_AVX_ENABLED

// ============================================================================
// SSE версии операций (128-bit, 4 float за итерацию)
// ============================================================================

#ifdef SIMD_SSE_ENABLED

/**
 * SSE сумма векторов: r[i] = z1[i] + z2[i]
 * Обрабатывает 4 элемента за итерацию
 */
inline void op_add_sse(float* r, const float* z1, const float* z2, const int size) {
    int i = 0;

    // Обработка по 4 элемента (128 бит)
    for (; i <= size - 4; i += 4) {
        __m128 v1 = _mm_loadu_ps(z1 + i);
        __m128 v2 = _mm_loadu_ps(z2 + i);
        __m128 result = _mm_add_ps(v1, v2);
        _mm_storeu_ps(r + i, result);
    }

    // Обработка оставшихся элементов
    for (; i < size; i++) {
        r[i] = z1[i] + z2[i];
    }
}

/**
 * SSE разность векторов: r[i] = z1[i] - z2[i]
 */
inline void op_sub_sse(float* r, const float* z1, const float* z2, const int size) {
    int i = 0;

    for (; i <= size - 4; i += 4) {
        __m128 v1 = _mm_loadu_ps(z1 + i);
        __m128 v2 = _mm_loadu_ps(z2 + i);
        __m128 result = _mm_sub_ps(v1, v2);
        _mm_storeu_ps(r + i, result);
    }

    for (; i < size; i++) {
        r[i] = z1[i] - z2[i];
    }
}

/**
 * SSE обратная разность векторов: r[i] = z2[i] - z1[i]
 */
inline void op_rsub_sse(float* r, const float* z1, const float* z2, const int size) {
    int i = 0;

    for (; i <= size - 4; i += 4) {
        __m128 v1 = _mm_loadu_ps(z1 + i);
        __m128 v2 = _mm_loadu_ps(z2 + i);
        __m128 result = _mm_sub_ps(v2, v1);
        _mm_storeu_ps(r + i, result);
    }

    for (; i < size; i++) {
        r[i] = z2[i] - z1[i];
    }
}

/**
 * SSE произведение векторов: r[i] = z1[i] * z2[i]
 */
inline void op_mul_sse(float* r, const float* z1, const float* z2, const int size) {
    int i = 0;

    for (; i <= size - 4; i += 4) {
        __m128 v1 = _mm_loadu_ps(z1 + i);
        __m128 v2 = _mm_loadu_ps(z2 + i);
        __m128 result = _mm_mul_ps(v1, v2);
        _mm_storeu_ps(r + i, result);
    }

    for (; i < size; i++) {
        r[i] = z1[i] * z2[i];
    }
}

#endif // SIMD_SSE_ENABLED

// ============================================================================
// Диспетчеризация: выбор оптимальной реализации
// Использует флаг UseSIMD для возможности отключения SIMD во время выполнения
// ============================================================================

/**
 * Сумма векторов с автоматическим выбором реализации
 * Выбирает AVX -> SSE -> скалярный код в зависимости от доступности и флага UseSIMD
 */
inline void op_add_simd(float* r, const float* z1, const float* z2, const int size) {
#ifdef SIMD_AVX_ENABLED
    if (UseSIMD) {
        op_add_avx(r, z1, z2, size);
    } else {
        op_add_scalar(r, z1, z2, size);
    }
#elif defined(SIMD_SSE_ENABLED)
    if (UseSIMD) {
        op_add_sse(r, z1, z2, size);
    } else {
        op_add_scalar(r, z1, z2, size);
    }
#else
    op_add_scalar(r, z1, z2, size);
#endif
}

/**
 * Разность векторов с автоматическим выбором реализации
 */
inline void op_sub_simd(float* r, const float* z1, const float* z2, const int size) {
#ifdef SIMD_AVX_ENABLED
    if (UseSIMD) {
        op_sub_avx(r, z1, z2, size);
    } else {
        op_sub_scalar(r, z1, z2, size);
    }
#elif defined(SIMD_SSE_ENABLED)
    if (UseSIMD) {
        op_sub_sse(r, z1, z2, size);
    } else {
        op_sub_scalar(r, z1, z2, size);
    }
#else
    op_sub_scalar(r, z1, z2, size);
#endif
}

/**
 * Обратная разность векторов с автоматическим выбором реализации
 */
inline void op_rsub_simd(float* r, const float* z1, const float* z2, const int size) {
#ifdef SIMD_AVX_ENABLED
    if (UseSIMD) {
        op_rsub_avx(r, z1, z2, size);
    } else {
        op_rsub_scalar(r, z1, z2, size);
    }
#elif defined(SIMD_SSE_ENABLED)
    if (UseSIMD) {
        op_rsub_sse(r, z1, z2, size);
    } else {
        op_rsub_scalar(r, z1, z2, size);
    }
#else
    op_rsub_scalar(r, z1, z2, size);
#endif
}

/**
 * Произведение векторов с автоматическим выбором реализации
 */
inline void op_mul_simd(float* r, const float* z1, const float* z2, const int size) {
#ifdef SIMD_AVX_ENABLED
    if (UseSIMD) {
        op_mul_avx(r, z1, z2, size);
    } else {
        op_mul_scalar(r, z1, z2, size);
    }
#elif defined(SIMD_SSE_ENABLED)
    if (UseSIMD) {
        op_mul_sse(r, z1, z2, size);
    } else {
        op_mul_scalar(r, z1, z2, size);
    }
#else
    op_mul_scalar(r, z1, z2, size);
#endif
}

// ============================================================================
// Информация о доступных SIMD расширениях
// ============================================================================

/**
 * Возвращает строку с информацией о доступных SIMD расширениях
 */
inline const char* getSIMDInfo() {
#ifdef SIMD_AVX_ENABLED
    return "AVX (256-bit, 8 floats per operation)";
#elif defined(SIMD_SSE_ENABLED)
    return "SSE (128-bit, 4 floats per operation)";
#else
    return "None (scalar operations)";
#endif
}

/**
 * Проверяет, включены ли SIMD оптимизации при компиляции
 */
inline bool isSIMDEnabled() {
#if defined(SIMD_AVX_ENABLED) || defined(SIMD_SSE_ENABLED)
    return true;
#else
    return false;
#endif
}

#endif // SIMD_OPS_H
