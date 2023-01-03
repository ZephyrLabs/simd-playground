/**
 * @file neon_mul.cpp
 * @author Sravan Senthilnathan
 * @brief NEON_SIMD implementation of Tensor Multiplication
 * @version 0.1
 * @date 2023-01-03
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <iostream>
#include <arm_neon.h>
#include <vector>

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
 * @brief NEON_SIMD accelerated Tensor Multiplication function:
 * 
 * @param a first operand tensor
 * @param b second operand tensor
 * @param c output tensor
 */
void neon_mul(const Tensor a, const Tensor b, Tensor &c){

    float32x4_t Ta0 = vld1q_f32(a[0]);
    float32x4_t Ta1 = vld1q_f32(a[1]);
    float32x4_t Ta2 = vld1q_f32(a[2]);
    float32x4_t Ta3 = vld1q_f32(a[3]);

    float32x4_t Tb0 = {b[0][0], b[1][0], b[2][0], b[3][0]};
    float32x4_t Tb1 = {b[0][1], b[1][1], b[2][1], b[3][1]};
    float32x4_t Tb2 = {b[0][2], b[1][2], b[2][2], b[3][2]};
    float32x4_t Tb3 = {b[0][3], b[1][3], b[2][3], b[3][3]};

    float32x4_t Residual;
    
    Residual = vmulq_f32(Ta0, Tb0);
    c[0][0] = vaddvq_f32(Residual);

    Residual = vmulq_f32(Ta0, Tb1);
    c[0][1] = vaddvq_f32(Residual);

    Residual = vmulq_f32(Ta0, Tb2);
    c[0][2] = vaddvq_f32(Residual);

    Residual = vmulq_f32(Ta0, Tb3);
    c[0][3] = vaddvq_f32(Residual);

    Residual = vmulq_f32(Ta1, Tb0);
    c[1][0] = vaddvq_f32(Residual);

    Residual = vmulq_f32(Ta1, Tb1);
    c[1][1] = vaddvq_f32(Residual);

    Residual = vmulq_f32(Ta1, Tb2);
    c[1][2] = vaddvq_f32(Residual);

    Residual = vmulq_f32(Ta1, Tb3);
    c[1][3] = vaddvq_f32(Residual);

    Residual = vmulq_f32(Ta2, Tb0);
    c[2][0] = vaddvq_f32(Residual);

    Residual = vmulq_f32(Ta2, Tb1);
    c[2][1] = vaddvq_f32(Residual);

    Residual = vmulq_f32(Ta2, Tb2);
    c[2][2] = vaddvq_f32(Residual);

    Residual = vmulq_f32(Ta2, Tb3);
    c[2][3] = vaddvq_f32(Residual);

    Residual = vmulq_f32(Ta3, Tb0);
    c[3][0] = vaddvq_f32(Residual);

    Residual = vmulq_f32(Ta3, Tb1);
    c[3][1] = vaddvq_f32(Residual);

    Residual = vmulq_f32(Ta3, Tb2);
    c[3][2] = vaddvq_f32(Residual);

    Residual = vmulq_f32(Ta3, Tb3);
    c[3][3] = vaddvq_f32(Residual);
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
    neon_mul(a, b, d); 
    auto sp2 = chrono::high_resolution_clock::now();

    cout << "------------------NEON-TENSOR-MUL------------------" << endl;

    auto d1 = chrono::duration_cast<std::chrono::nanoseconds>(sp1 - st1);
     
    cout << "Time taken by normal function: "
         << d1.count() << " nanoseconds" << endl;

    auto d2 = chrono::duration_cast<std::chrono::nanoseconds>(sp2 - st2);
 
    cout << "Time taken by NEON function: "
         << d2.count() << " nanoseconds" << endl;

    const float percent = (float)d1.count()/(float)d2.count() * 100;

    cout << "Speed Uplift: "
        << percent << " %" << endl;

    cout << "---------------------------------------------------" << endl;

    return(0);
}
