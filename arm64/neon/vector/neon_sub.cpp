/**
 * @file neon_sub.cpp
 * @author Sravan Senthilnathan
 * @brief NEON_SIMD implementation of parallel floating point subtraction
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

/**
 * @brief Standard Vector Subtraction function:
 * 
 * @param a first operand vector 
 * @param b second operand vector
 * @param c output vector
 */
void sub(const vector<float> a, const vector<float> b, vector<float>& c){
    for(int i = 0; i < a.size(); ++i){
        c[i] = a[i] - b[i];
    }
}

/**
 * @brief NEON accelerated Vector Subtraction function:
 * 
 * @param a first operand vector
 * @param b second operand vector
 * @param c output vector
 */
void neon_sub(const vector<float> a, const vector<float> b, vector<float>& c){

    const int vectorize = (a.size() / 4u) * 4u;

    int i = 0;

    for(; i < vectorize; i += 4u){
        float32x4_t aReg = vld1q_f32(a.data() + i);
        float32x4_t bReg = vld1q_f32(b.data() + i);
        float32x4_t cReg = vsubq_f32(aReg, bReg);
        vst1q_f32(c.data() + i, cReg);
    }
    for(; i < a.size(); ++i){
       c[i] = a[i] - b[i];
    }
}

int main(){
    vector<float> a(100000);
    vector<float> b(100000);
    vector<float> c(100000);

    iota(a.begin(), a.end(), 0.9);
    iota(b.begin(), b.end(), 0.6);

    // normal implementation:
    auto st1 = chrono::high_resolution_clock::now();
    sub(a, b, c); 
    auto sp1 = chrono::high_resolution_clock::now();

    // vectorized implementation:
    auto st2 = chrono::high_resolution_clock::now();
    neon_sub(a, b, c); 
    auto sp2 = chrono::high_resolution_clock::now();

    cout << "------------------NEON-VECTOR-SUB------------------" << endl;

    auto d1 = chrono::duration_cast<std::chrono::microseconds>(sp1 - st1);
 
    cout << "Time taken by normal function: "
         << d1.count() << " microseconds" << endl;

    auto d2 = chrono::duration_cast<std::chrono::microseconds>(sp2 - st2);
 
    cout << "Time taken by NEON function: "
         << d2.count() << " microseconds" << endl;

    const float percent = (float)d1.count()/(float)d2.count() * 100;

    cout << "Speed Uplift: "
        << percent << " %" << endl;

    cout << "---------------------------------------------------" << endl;

    return(0);
}
