/**
 * @file avx_add.cpp
 * @author Sravan Senthilnathan 
 * @brief AVX implementation of Tensor Subtraction
 * @version 0.1
 * @date 2023-01-02
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <iostream>
#include <immintrin.h>

// misc lib:
#include <algorithm>
#include <numeric>
#include <chrono>

using namespace std;

// implicit Tensor Declaration:
using Tensor = float[4][4];

/**
 * @brief Standard Tensor Subtraction function:
 * 
 * @param a first operand tensor
 * @param b second operand tensor
 * @param c output tensor
 */
void sub(const Tensor a, const Tensor b, Tensor &c){
    for(int i = 0; i < 4; ++i){
        for(int j = 0; j < 4; ++j){
            c[i][j] = a[i][j] + b[i][j];
        }
    }
}

/**
 * @brief AVX accelerated Tensor Subtraction function (uses AVX1):
 * 
 * @param a first operand tensor
 * @param b second operand tensor
 * @param c output tensor
 */
void avx_sub(const Tensor a, const Tensor b, Tensor &c){
    __m128 Ta0, Ta1, Ta2, Ta3;
    __m128 Tb0, Tb1, Tb2, Tb3;
    
    Ta0 = _mm_loadu_ps(a[0]);
    Ta1 = _mm_loadu_ps(a[1]);
    Ta2 = _mm_loadu_ps(a[2]);
    Ta3 = _mm_loadu_ps(a[3]);

    Tb0 = _mm_loadu_ps(b[0]);
    Tb1 = _mm_loadu_ps(b[1]);
    Tb2 = _mm_loadu_ps(b[2]);
    Tb3 = _mm_loadu_ps(b[3]);

    
    _mm_storeu_ps(c[0], _mm_sub_ps(Ta0, Tb0));
    _mm_storeu_ps(c[1], _mm_sub_ps(Ta1, Tb1));
    _mm_storeu_ps(c[2], _mm_sub_ps(Ta2, Tb2));
    _mm_storeu_ps(c[3], _mm_sub_ps(Ta3, Tb3));
}

int main(){
    Tensor a = {{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}, {4, 5, 6, 7}};
    Tensor b = {{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}, {4, 5, 6, 7}};
    Tensor c = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

    // normal implementation:
    auto st1 = chrono::high_resolution_clock::now();
    sub(a, b, c); 
    auto sp1 = chrono::high_resolution_clock::now();

    // vectorized implementation:
    auto st2 = chrono::high_resolution_clock::now();
    avx_sub(a, b, c); 
    auto sp2 = chrono::high_resolution_clock::now();

    cout << "-------------------AVX-TENSOR-SUB------------------" << endl;

    auto d1 = chrono::duration_cast<std::chrono::nanoseconds>(sp1 - st1);
 
    cout << "Time taken by normal function: "
         << d1.count() << " nanoseconds" << endl;

    auto d2 = chrono::duration_cast<std::chrono::nanoseconds>(sp2 - st2);
 
    cout << "Time taken by AVX function: "
         << d2.count() << " nanoseconds" << endl;

    const float percent = (float)d1.count()/(float)d2.count() * 100;

    cout << "Speed Uplift: "
        << percent << " %" << endl;

    return(0);
}