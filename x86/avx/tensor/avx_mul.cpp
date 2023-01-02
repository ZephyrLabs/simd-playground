/**
 * @file avx_add.cpp
 * @author Sravan Senthilnathan 
 * @brief AVX implementation of Tensor Multiplication
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
 * @brief Standard Tensor Multiplication function:
 * 
 * @param a first operand tensor
 * @param b second operand tensor
 * @param c output tensor
 */
void mul(const Tensor a, const Tensor b, Tensor &c){
    for(int i = 0; i < 4; ++i){
        for(int j = 0; j < 4; ++j){
            c[i][j] = 0;
                for(int k = 0; k < 4; ++k){
                    c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

/**
 * @brief AVX accelerated Tensor Multiplication function (uses AVX1):
 * 
 * @param a first operand tensor
 * @param b second operand tensor
 * @param c output tensor
 */
void avx_mul(const Tensor a, const Tensor b, Tensor &c){
    __m128 Ta0, Ta1, Ta2, Ta3;

    Ta0 = _mm_load_ps(a[0]);
    Ta1 = _mm_load_ps(a[1]);
    Ta2 = _mm_load_ps(a[2]);
    Ta3 = _mm_load_ps(a[3]);

    __m128 Tb0 = {b[0][0], b[1][0], b[2][0], b[3][0]};
    __m128 Tb1 = {b[0][1], b[1][1], b[2][1], b[3][1]};
    __m128 Tb2 = {b[0][2], b[1][2], b[2][2], b[3][2]};
    __m128 Tb3 = {b[0][3], b[1][3], b[2][3], b[3][3]};

    __m128 Residual;
    
    Residual = _mm_mul_ps(Ta0, Tb0);
    Residual = _mm_hadd_ps(Residual, Residual);
    Residual = _mm_hadd_ps(Residual, Residual);
    c[0][0] = Residual[0];

    Residual = _mm_mul_ps(Ta0, Tb1);
    Residual = _mm_hadd_ps(Residual, Residual);
    Residual = _mm_hadd_ps(Residual, Residual);
    c[0][1] = Residual[0];

    Residual = _mm_mul_ps(Ta0, Tb2);
    Residual = _mm_hadd_ps(Residual, Residual);
    Residual = _mm_hadd_ps(Residual, Residual);
    c[0][2] = Residual[0];

    Residual = _mm_mul_ps(Ta0, Tb3);
    Residual = _mm_hadd_ps(Residual, Residual);
    Residual = _mm_hadd_ps(Residual, Residual);
    c[0][3] = Residual[0];

    Residual = _mm_mul_ps(Ta1, Tb0);
    Residual = _mm_hadd_ps(Residual, Residual);
    Residual = _mm_hadd_ps(Residual, Residual);
    c[1][0] = Residual[0];

    Residual = _mm_mul_ps(Ta1, Tb1);
    Residual = _mm_hadd_ps(Residual, Residual);
    Residual = _mm_hadd_ps(Residual, Residual);
    c[1][1] = Residual[0];

    Residual = _mm_mul_ps(Ta1, Tb2);
    Residual = _mm_hadd_ps(Residual, Residual);
    Residual = _mm_hadd_ps(Residual, Residual);
    c[1][2] = Residual[0];

    Residual = _mm_mul_ps(Ta1, Tb3);
    Residual = _mm_hadd_ps(Residual, Residual);
    Residual = _mm_hadd_ps(Residual, Residual);
    c[1][3] = Residual[0];

    Residual = _mm_mul_ps(Ta2, Tb0);
    Residual = _mm_hadd_ps(Residual, Residual);
    Residual = _mm_hadd_ps(Residual, Residual);
    c[2][0] = Residual[0];

    Residual = _mm_mul_ps(Ta2, Tb1);
    Residual = _mm_hadd_ps(Residual, Residual);
    Residual = _mm_hadd_ps(Residual, Residual);
    c[2][1] = Residual[0];

    Residual = _mm_mul_ps(Ta2, Tb2);
    Residual = _mm_hadd_ps(Residual, Residual);
    Residual = _mm_hadd_ps(Residual, Residual);
    c[2][2] = Residual[0];

    Residual = _mm_mul_ps(Ta2, Tb3);
    Residual = _mm_hadd_ps(Residual, Residual);
    Residual = _mm_hadd_ps(Residual, Residual);
    c[2][3] = Residual[0];

    Residual = _mm_mul_ps(Ta3, Tb0);
    Residual = _mm_hadd_ps(Residual, Residual);
    Residual = _mm_hadd_ps(Residual, Residual);
    c[3][0] = Residual[0];

    Residual = _mm_mul_ps(Ta3, Tb1);
    Residual = _mm_hadd_ps(Residual, Residual);
    Residual = _mm_hadd_ps(Residual, Residual);
    c[3][1] = Residual[0];

    Residual = _mm_mul_ps(Ta3, Tb2);
    Residual = _mm_hadd_ps(Residual, Residual);
    Residual = _mm_hadd_ps(Residual, Residual);
    c[3][2] = Residual[0];

    Residual = _mm_mul_ps(Ta3, Tb3);
    Residual = _mm_hadd_ps(Residual, Residual);
    Residual = _mm_hadd_ps(Residual, Residual);
    c[3][3] = Residual[0];
}

int main(){
    Tensor a = {{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}, {4, 5, 6, 7}};
    Tensor b = {{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}, {4, 5, 6, 7}};
    Tensor c = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
    Tensor d = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

    // normal implementation:
    auto st1 = chrono::high_resolution_clock::now();
    mul(a, b, c); 
    auto sp1 = chrono::high_resolution_clock::now();

    // vectorized implementation:
    auto st2 = chrono::high_resolution_clock::now();
    avx_mul(a, b, d); 
    auto sp2 = chrono::high_resolution_clock::now();

    auto d1 = chrono::duration_cast<std::chrono::nanoseconds>(sp1 - st1);
 
    cout << "Time taken by normal function: "
         << d1.count() << " nanoseconds" << endl;

    auto d2 = chrono::duration_cast<std::chrono::nanoseconds>(sp2 - st2);
 
    cout << "Time taken by AVX function: "
         << d2.count() << " nanoseconds" << endl;

    const float diff = d1.count() - d2.count();
    float percent = diff/(float)d1.count() * 100;

    cout << "Speed Uplift: "
        << percent << " %" << endl;

    return(0);
}
