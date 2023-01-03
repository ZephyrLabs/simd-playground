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

**note:** you will need to specify the correct arch makefile with: 
> -f avx-Makefile/neon-Makefile
accordingly.

### Examples:
- vec_add
- vec_sub
- vec_mul
- vec_div
- tensor_add
- tensor_sub
- tensor_mul

### Test Results:
Compiler: **g++**
Optimizations: **none**
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

Compiler: **g++**
Optimizations: **O2**
```
-------------------AVX-VECTOR-ADD------------------
Time taken by normal function: 3050 microseconds
Time taken by AVX function: 2664 microseconds
Speed Uplift: 114.489 %
---------------------------------------------------
-------------------AVX-TENSOR-SUB------------------
Time taken by normal function: 2999 microseconds
Time taken by AVX function: 2684 microseconds
Speed Uplift: 111.736 %
---------------------------------------------------
-------------------AVX-TENSOR-MUL------------------
Time taken by normal function: 3158 microseconds
Time taken by AVX function: 2707 microseconds
Speed Uplift: 116.661 %
---------------------------------------------------
-------------------AVX-VECTOR-DIV------------------
Time taken by normal function: 2996 microseconds
Time taken by AVX function: 2612 microseconds
Speed Uplift: 114.701 %
---------------------------------------------------
-------------------AVX-TENSOR-ADD------------------
Time taken by normal function: 101 nanoseconds
Time taken by AVX function: 25 nanoseconds
Speed Uplift: 404 %
---------------------------------------------------
-------------------AVX-TENSOR-SUB------------------
Time taken by normal function: 147 nanoseconds
Time taken by AVX function: 26 nanoseconds
Speed Uplift: 565.385 %
-------------------AVX-TENSOR-MUL------------------
Time taken by normal function: 254 nanoseconds
Time taken by AVX function: 26 nanoseconds
Speed Uplift: 976.923 %
---------------------------------------------------
```

Compiler: **clang++**
Optimizations: **none**
```
-------------------AVX-VECTOR-ADD------------------
Time taken by normal function: 5630 microseconds
Time taken by AVX function: 3450 microseconds
Speed Uplift: 163.188 %
---------------------------------------------------
-------------------AVX-TENSOR-SUB------------------
Time taken by normal function: 6052 microseconds
Time taken by AVX function: 3610 microseconds
Speed Uplift: 167.645 %
---------------------------------------------------
-------------------AVX-TENSOR-MUL------------------
Time taken by normal function: 5716 microseconds
Time taken by AVX function: 3430 microseconds
Speed Uplift: 166.647 %
---------------------------------------------------
-------------------AVX-VECTOR-DIV------------------
Time taken by normal function: 5845 microseconds
Time taken by AVX function: 3372 microseconds
Speed Uplift: 173.339 %
---------------------------------------------------
-------------------AVX-TENSOR-ADD------------------
Time taken by normal function: 125 nanoseconds
Time taken by AVX function: 52 nanoseconds
Speed Uplift: 240.385 %
---------------------------------------------------
-------------------AVX-TENSOR-SUB------------------
Time taken by normal function: 142 nanoseconds
Time taken by AVX function: 49 nanoseconds
Speed Uplift: 289.796 %
-------------------AVX-TENSOR-MUL------------------
Time taken by normal function: 266 nanoseconds
Time taken by AVX function: 149 nanoseconds
Speed Uplift: 178.523 %
---------------------------------------------------
```

