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
- convolution

### sample run:
Compiler: **g++**
Optimizations: **none**
```
-------------------AVX-VECTOR-ADD------------------
Time taken by normal function: 5819 microseconds
Time taken by AVX function: 3437 microseconds
Speed Uplift: 169.305 %
---------------------------------------------------
-------------------AVX-TENSOR-SUB------------------
Time taken by normal function: 6056 microseconds
Time taken by AVX function: 3539 microseconds
Speed Uplift: 171.122 %
---------------------------------------------------
-------------------AVX-TENSOR-MUL------------------
Time taken by normal function: 5886 microseconds
Time taken by AVX function: 3348 microseconds
Speed Uplift: 175.806 %
---------------------------------------------------
-------------------AVX-VECTOR-DIV------------------
Time taken by normal function: 5867 microseconds
Time taken by AVX function: 3416 microseconds
Speed Uplift: 171.751 %
---------------------------------------------------
-------------------AVX-TENSOR-ADD------------------
Time taken by normal function: 168 nanoseconds
Time taken by AVX function: 85 nanoseconds
Speed Uplift: 197.647 %
---------------------------------------------------
-------------------AVX-TENSOR-SUB------------------
Time taken by normal function: 186 nanoseconds
Time taken by AVX function: 75 nanoseconds
Speed Uplift: 248 %
-------------------AVX-TENSOR-MUL------------------
Time taken by normal function: 306 nanoseconds
Time taken by AVX function: 184 nanoseconds
Speed Uplift: 166.304 %
---------------------------------------------------
------------------AVX-CONVOLUTION------------------
Time taken by normal function: 2673 nanoseconds
Time taken by AVX function: 1476 nanoseconds
Speed Uplift: 181.098 %
---------------------------------------------------
```


