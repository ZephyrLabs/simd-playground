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