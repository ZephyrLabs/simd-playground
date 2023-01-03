/**
 * @file neon_add.cpp
 * @author Sravan Senthilnathan
 * @brief NEON_SIMD implementation of Tensor Addition
 * @version 0.1
 * @date 2023-01-03
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <iostream>
#include <arm_neon.h>

// misc lib:
#include <algorithm>
#include <numeric>
#include <chrono>

using namespace std;

// implicit Tensor Declaration:
using Tensor = float[4][4];

/**
 * @brief Standard Tensor Addition function:
 * 
 * @param a first operand tensor
 * @param b second operand tensor
 * @param c output tensor
 */
void add(const Tensor a, const Tensor b, Tensor &c){
    for(int i = 0; i < 4; ++i){
        for(int j = 0; j < 4; ++j){
            c[i][j] = a[i][j] + b[i][j];
        }
    }
}

/**
 * @brief NEON_SIMD accelerated Tensor Addition function:
 * 
 * @param a first operand tensor
 * @param b second operand tensor
 * @param c output tensor
 */
void neon_add(const Tensor a, const Tensor b, Tensor &c){    
    float32x4_t Ta0 = vld1q_f32(a[0]);
    float32x4_t Ta1 = vld1q_f32(a[1]);
    float32x4_t Ta2 = vld1q_f32(a[2]);
    float32x4_t Ta3 = vld1q_f32(a[3]);

    float32x4_t Tb0 = vld1q_f32(b[0]);
    float32x4_t Tb1 = vld1q_f32(b[1]);
    float32x4_t Tb2 = vld1q_f32(b[2]);
    float32x4_t Tb3 = vld1q_f32(b[3]);

    vst1q_f32(c[0], vaddq_f32(Ta0, Tb0));
    vst1q_f32(c[1], vaddq_f32(Ta1, Tb1));
    vst1q_f32(c[2], vaddq_f32(Ta2, Tb2));
    vst1q_f32(c[3], vaddq_f32(Ta3, Tb3));
}

int main(){
    Tensor a = {{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}, {4, 5, 6, 7}};
    Tensor b = {{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}, {4, 5, 6, 7}};
    Tensor c = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

    // normal implementation:
    auto st1 = chrono::high_resolution_clock::now();
    add(a, b, c); 
    auto sp1 = chrono::high_resolution_clock::now();

    // vectorized implementation:
    auto st2 = chrono::high_resolution_clock::now();
    neon_add(a, b, c); 
    auto sp2 = chrono::high_resolution_clock::now();

    cout << "------------------NEON-TENSOR-ADD------------------" << endl;

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