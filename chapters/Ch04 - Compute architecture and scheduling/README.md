# Excercises

### 1. <img width="592" alt="image" src="https://github.com/user-attachments/assets/d5827542-eff4-4871-8137-3a62505b4d40">

a. What is the number of warps per block? 

- "#" of threads per block = 128

- "#" of threads in a wrap = 32

- 128/32 = 4 warps per block 

b. What is the number of warps in the grid?

- "#" of grids = 9 

- 4 wraps per block

- 4 * 9 = 36 wraps in grid 

c. For the statement on line 04:

Wrap 0 = 0-31 thread IDx

Wrap 1 = 32-63 thread IDx

Wrap 2 = 64-95 thread IDx

Wrap 3 = 96-127 thread IDx

- i. How many warps in the grid are active?

  - 2 are active (Wrap 0 & 3 successfully execute the conditional)

- ii. How many warps in the grid are divergent?

  - 1 wrap (Wrap 1) 

- iii. What is the SIMD efficiency (in %) of warp 0 of block 0? iv. What is the SIMD efficiency (in %) of warp 1 of block 0? v. What is the SIMD efficiency (in %) of warp 3 of block 0?

  - Wrap 0 & 3 = 100% 

  - Wrap 1 = 62.5%

  - Wrap 2 = 0%

d. For the statement on line 07:
- i. How many warps in the grid are active?
  - All wraps are active  
- ii. How many warps in the grid are divergent?
  - All wrap are divergent 
- iii. What is the SIMD efficiency (in %) of warp 0 of block 0?
  - 50%
 
e. For the loop on line 09:
- i. How many iterations have no divergence?
  - 1/3 
- ii. How many iterations have divergence?
  - 2/3 

### 2. For a vector addition, assume that the vector length is 2000, each thread calculates one output element, and the thread block size is 512 threads. How many threads will be in the grid?

vector length = 2000
threads per block = 512

2000 / 512 ~ 4 blocks to cover length 

512 * 4 = 2048 threads in the grid 

### 3. For the previous question, how many warps do you expect to have divergence due to the boundary check on vector length?

2000 / 32 ~ 63 wraps

Threads 2000-2017 will be out of bounds 

Wraps 62 & 63 will be divergent 

