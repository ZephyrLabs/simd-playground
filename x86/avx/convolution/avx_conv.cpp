/**
 * @file avx_conv.cpp
 * @author sravan senthilnathan
 * @brief AVX implementation of convolution operation
 * @version 0.1
 * @date 2023-02-12
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
 * @brief normal function to compute the convolution of a given discrete time signal
 * 
 * @param x discrete time input signal
 * @param h discrete impulse response
 * @param y convoluted response
 */
void conv(const vector<float> x, const vector<float> h, vector<float>& y){
    const auto l1 = x.size();
    const auto l2 = h.size();
    const auto l = l1 + l2 - 1;

    for(int n = 0; n < l; n++){
        for(int k = 0 ;k < l1; k++){
            if((n - k) >= 0 && (n - k) <= l2){
                y[n] = y[n] + (x[k]*h[n - k]);
            }
        }
    }
}

/**
 * @brief AVX accelerated convolution function (uses AVX1):
 * 
 * @param x discrete time input signal
 * @param h discrete impulse response
 * @param y convoluted response
 */
void avx_conv(const vector<float> x, const vector<float> h, vector<float>& y){
    const auto l1 = x.size();
    const auto l2 = h.size();
    const auto l = l1 + l2 - 1;

    int i = 0;
    __m128 a, b, c;

    for(int n = 0; n < l; n++){
        i = 0;
        c = _mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f);
        for(int k = 0 ;k < l1; k++){
            if((n - k) >= 0 && (n - k) <= l2){  
                a[i] = x[k];
                b[i] = h[n - k];
                i++;

                if(i > 3){
                    c = _mm_mul_ps(a, b);
                    c = _mm_hadd_ps(c, c);
                    c = _mm_hadd_ps(c, c);
                    
                    y[n] += c[0];

                    c = _mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f);
                    i = 0;
                }
            }
        }    
        if (i != 0){
            a[i] = 0;

            c = _mm_mul_ps(a, b);
            c = _mm_hadd_ps(c, c);
            c = _mm_hadd_ps(c, c);
            y[n] += c[0];
        }
    }
}

int main(){
    const vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    const vector<float> b = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    vector<float> c  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    vector<float> d  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // normal implementation:
    auto st1 = chrono::high_resolution_clock::now();
    conv(a, b, c); 
    auto sp1 = chrono::high_resolution_clock::now();

    // vectorized implementation:
    auto st2 = chrono::high_resolution_clock::now();
    avx_conv(a, b, d); 
    auto sp2 = chrono::high_resolution_clock::now();

    cout << "------------------AVX-CONVOLUTION------------------" << endl;

    auto d1 = chrono::duration_cast<std::chrono::nanoseconds>(sp1 - st1);
     
    cout << "Time taken by normal function: "
         << d1.count() << " nanoseconds" << endl;

    auto d2 = chrono::duration_cast<std::chrono::nanoseconds>(sp2 - st2);
 
    cout << "Time taken by AVX function: "
         << d2.count() << " nanoseconds" << endl;

    const float percent = (float)d1.count()/(float)d2.count() * 100;

    cout << "Speed Uplift: "
        << percent << " %" << endl;

    cout << "---------------------------------------------------" << endl;

    return(0);
}