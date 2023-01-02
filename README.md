# SIMD playground
 experiments with simd instructions on x86 and ARM platforms

This Repo is a collection of various code examples and implementations using SIMD paralleism.

## Introduction to SIMD:
SIMD instructions provide powerful data crunching capabilities by allowing
operations on Multiple Data using a Single Instruction call.

The power of parallelism dominates against faster single pipeline data processing methods.

This technology aids in places such as Improved HPC, Signal Processing, etc.

## SIMD instruction types:
each architecture has its own implmentation of it.

### x86:
* For x86 it has been **SSE** (Streaming SIMD Extensions) and **AVX** (Advanced Vector Extensions) instructions.

* The x86 simplified library will mostly focus on implmentations using AVX2 (256 
bit wide operations) but may also use AVX (128 bit wide operations)

### ARMv8
* For the Arm implmentation it will be using **NEON** instructions.

## Testing:
setup up the compile environment with:
```
git clone https://github.com/ZephyrLabs/simd-playground
cd simd-playground
make init
make -s all
```

run an example with:
```
make <example name>
```
example: `make vec_add`

### Examples:
Under AVX:
- vec_add
- vec_sub
- vec_mul
- vec_div
- tensor_add
- tensor_sub
- tensor_mul

### Test Results:
```
-------------------AVX-VECTOR-ADD------------------
Time taken by normal function: 5769 microseconds
Time taken by AVX function: 3370 microseconds
Speed Uplift: 171.187 %
---------------------------------------------------
-------------------AVX-TENSOR-SUB------------------
Time taken by normal function: 6238 microseconds
Time taken by AVX function: 3504 microseconds
Speed Uplift: 178.025 %
---------------------------------------------------
-------------------AVX-TENSOR-MUL------------------
Time taken by normal function: 5803 microseconds
Time taken by AVX function: 3404 microseconds
Speed Uplift: 170.476 %
---------------------------------------------------
-------------------AVX-VECTOR-DIV------------------
Time taken by normal function: 5766 microseconds
Time taken by AVX function: 3384 microseconds
Speed Uplift: 170.39 %
---------------------------------------------------
-------------------AVX-TENSOR-ADD------------------
Time taken by normal function: 151 nanoseconds
Time taken by AVX function: 74 nanoseconds
Speed Uplift: 204.054 %
---------------------------------------------------
-------------------AVX-TENSOR-SUB------------------
Time taken by normal function: 142 nanoseconds
Time taken by AVX function: 84 nanoseconds
Speed Uplift: 169.048 %
-------------------AVX-TENSOR-MUL------------------
Time taken by normal function: 321 nanoseconds
Time taken by AVX function: 112 nanoseconds
Speed Uplift: 286.607 %
---------------------------------------------------
```
