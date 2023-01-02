/**
 * @file avx_add.cpp
 * @author Sravan Senthilnathan
 * @brief AVX implementation of parallel floating point addition
 * @version 0.1
 * @date 2023-01-02
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <iostream>
#include <immintrin.h>
#include <vector>

// misc lib:
#include <algorithm>
#include <numeric>
#include <chrono>

using namespace std;

/**
 * @brief Standard Vector Addition function:
 * 
 * @param a first operand vector 
 * @param b second operand vector
 * @param c output vector
 */
void add(const vector<float> a, const vector<float> b, vector<float>& c){
    for(int i = 0; i < a.size(); ++i){
        c[i] = a[i] + b[i];
    }
}

/**
 * @brief AVX accelerated Vector Addition function (uses AVX2):
 * 
 * @param a first operand vector
 * @param b second operand vector
 * @param c output vector
 */
void avx_add(const vector<float> a, const vector<float> b, vector<float>& c){

    const int vectorize = (a.size() / 8u) * 8u;

    int i = 0;

    for(; i < vectorize; i += 8u){
        __m256 aReg = _mm256_loadu_ps(a.data() + i);
        __m256 bReg = _mm256_loadu_ps(a.data() + i);
        __m256 cReg = _mm256_add_ps(aReg, bReg);

        _mm256_storeu_ps(c.data() + i, cReg);
    }
    for(; i < a.size(); ++i){
       c[i] = a[i] + b[i];
    }
}

int main(){
    vector<float> a(1000000);
    vector<float> b(1000000);
    vector<float> c(1000000);

    iota(a.begin(), a.end(), 0.9);
    iota(b.begin(), b.end(), 0.6);

    // normal implementation:
    auto st1 = chrono::high_resolution_clock::now();
    add(a, b, c); 
    auto sp1 = chrono::high_resolution_clock::now();

    // vectorized implementation:
    auto st2 = chrono::high_resolution_clock::now();
    avx_add(a, b, c); 
    auto sp2 = chrono::high_resolution_clock::now();

    auto d1 = chrono::duration_cast<std::chrono::microseconds>(sp1 - st1);
 
    cout << "Time taken by normal function: "
         << d1.count() << " microseconds" << endl;

    auto d2 = chrono::duration_cast<std::chrono::microseconds>(sp2 - st2);
 
    cout << "Time taken by AVX function: "
         << d2.count() << " microseconds" << endl;

    const float diff = d1.count() - d2.count();
    float percent = diff/(float)d1.count() * 100;

    cout << "Speed Uplift: "
        << percent << " %" << endl;

    return(0);
}
