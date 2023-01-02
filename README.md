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
```

run any of the examples with:
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
