# Excercises

### 1. In this chapter we implemented a matrix multiplication kernel that has each thread produce one output matrix element. In this question, you will implement different matrix-matrix multiplication kernels and compare them.

  a. Write a kernel that has each thread produce one output matrix row. Fill in the execution configuration parameters for the design.
```C
  __global__
void matrixMulti_one_row(float* M, float* N, float* P, int width) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < width) {
        float Pvalue = 0;

        // iterate through each column 
        for (int col = 0; col < width; ++col) {
            float Pvalue = 0;

            // Calucalte one matrix P 
            for (int k = 0; k < width; ++k) {
                Pvalue += M[row * width + k] * N[k * width + col];
            }
            P[row * width + col] = Pvalue;
        }
    }
}
```
  b. Write a kernel that has each thread produce one output matrix column.
```C
__global__
void matrixMulti_one_column(float* M, float* N, float* P, int width) {
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width) {
        float Pvalue = 0;

        // iterate through each row 
        for (int row = 0; row < width; ++row) {
            float Pvalue = 0;

            // Calucalte one matrix P 
            for (int k = 0; k < width; ++k) {
                Pvalue += M[row * width + k] * N[k * width + col];
            }
            P[row * width + col] = Pvalue;
        }
    }
}
```


## 3. 
<img width="571" alt="image" src="https://github.com/j2moreno/Programming-Massively-Parallel-Processors/assets/13912964/4dfc208d-41f0-41b0-8426-386a2527354b">

block dim = (16, 32) 
grid dim = (20, 6)

a. What is the number of threads per block? 512

b. What is the number of threads in the grid? 512 * 20 * 6 = 61,440
 
c. What is the number of blocks in the grid? 120

d. What is the number of threads that execute the code on line 05? 45,000


### 4. Consider a 2D matrix with a width of 400 and a height of 500. The matrix is stored as a one-dimensional array. Specify the array index of the matrix element at row 20 and column 10:

a. If the matrix is stored in row-major order. 

row * width + col
-> 20 * 400 + 10 = 8500

b. If the matrix is stored in column-major order.

col * height + row
-> 10 * 500 + 20 = 5020

### 5. Consider a 3D tensor with a width of 400, a height of 500, and a depth of 300. The tensor is stored as a one-dimensional array in row-major order. Specify the array index of the tensor element at x = 10, y = 20, and z = 5.

index = z * (width * height) + y * width + x

5 * (400 * 500) + (20 * 400) + 10 = 1008010

