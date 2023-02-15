/**
 * @file neon_conv.cpp
 * @author sravan senthilnathan
 * @brief NEON_SIMD implemention of convolution operation
 * @version 0.1
 * @date 2023-02-14
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
 * @brief NEON accelerated convolution function:
 * 
 * @param x discrete time input signal
 * @param h discrete impulse response
 * @param y convoluted response
 */
void neon_conv(const vector<float> x, const vector<float> h, vector<float>& y){
    const auto l1 = x.size();
    const auto l2 = h.size();
    const auto l = l1 + l2 - 1;

    int i = 0;
    float32x4_t a, b, c;

    for(int n = 0; n < l; n++){
        i = 0;
        c = vmovq_n_f32(0.0f);
        for(int k = 0 ;k < l1; k++){
            if((n - k) >= 0 && (n - k) <= l2){  
                a[i] = x[k];
                b[i] = h[n - k];
                i++;

                if(i > 3){
                    c = vmulq_f32(a, b);
                    y[n] = vaddvq_f32(c);

                    c = vmovq_n_f32(0.0f);
                    i = 0;
                }
            }
        }    
        if (i != 0){ 
            a[i] = 0;

            c = vmulq_f32(a, b);
            y[n] = vaddvq_f32(c);
        }
    }
}

/* this part is to be tested on a ARMv8.2 and above platform, ignore for now

void neon_conv_fp16(const vector<float> x, const vector<float> h, vector<float>& y){
    const auto l1 = x.size();
    const auto l2 = h.size();
    const auto l = l1 + l2 - 1;

    int i = 0;
    float16x8_t a, b, c;

    for(int n = 0; n < l; n++){
        i = 0;
        c = vmovq_n_f16(0.0f);
        for(int k = 0 ;k < l1; k++){
            if((n - k) >= 0 && (n - k) <= l2){  
                a[i] = x[k];
                b[i] = h[n - k];
                i++;

                if(i > 7){
                    c = vmulq_f16(a, b);
                    y[n] = c[0] + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7];

                    c = vmovq_n_f16(0.0f);
                    i = 0;
                }
            }
        }    
        if (i != 0){ 
            a[i] = 0;

            c = vmulq_f16(a, b);
            y[n] = c[0] + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7];
        }
    }
}
*/

int main(){
    const vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    const vector<float> b = {1.0f, 2.0f, 3.0f, 4.0f};
    vector<float> c  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    vector<float> d  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // normal implementation:
    auto st1 = chrono::high_resolution_clock::now();
    conv(a, b, c); 
    auto sp1 = chrono::high_resolution_clock::now();

    // vectorized implementation:
    auto st2 = chrono::high_resolution_clock::now();
    neon_conv(a, b, d); 
    auto sp2 = chrono::high_resolution_clock::now();

    for(int i = 0; i < c.size(); i++){
        cout << c[i] << " " << d[i] << endl;
    }

    cout << "-----------------NEON-CONVOLUTION------------------" << endl;

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