# Blur colored image 

<img width="607" alt="image" src="https://github.com/j2moreno/Programming-Massively-Parallel-Processors/assets/13912964/373a1b2c-e30f-4c53-8619-f14e2be34a44">

## Input 
![Broccoli-treehouse](https://github.com/j2moreno/Programming-Massively-Parallel-Processors/assets/13912964/d0e3008d-eec7-4280-85d1-fade38699670)

## Output
![blurred_image](https://github.com/j2moreno/Programming-Massively-Parallel-Processors/assets/13912964/1dbb5151-d005-4463-82d2-4785cee7810b)

## Profiler
```
/content# nvprof ./blur_image Broccoli-treehouse.jpg 
Image Width: 575, Image Height: 419, Size: 722775, Channels: 3
==4573== NVPROF is profiling process 4573, command: ./blur_image Broccoli-treehouse.jpg
Block Dimensions: (16, 16, 3)
Grid Dimensions: (36, 27, 1)
==4573== Profiling application: ./blur_image Broccoli-treehouse.jpg
==4573== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   35.15%  66.910us         1  66.910us  66.910us  66.910us  blurKernel(unsigned char*, unsigned char*, int, int)
                   32.75%  62.335us         1  62.335us  62.335us  62.335us  [CUDA memcpy DtoH]
                   32.09%  61.086us         1  61.086us  61.086us  61.086us  [CUDA memcpy HtoD]
      API calls:   98.23%  92.448ms         2  46.224ms  6.9530us  92.441ms  cudaMalloc
                    1.08%  1.0156ms         2  507.78us  193.36us  822.19us  cudaMemcpy
                    0.29%  271.35us         1  271.35us  271.35us  271.35us  cudaLaunchKernel
                    0.22%  205.99us       114  1.8060us     248ns  79.953us  cuDeviceGetAttribute
                    0.15%  138.13us         2  69.065us  20.613us  117.52us  cudaFree
                    0.01%  13.208us         1  13.208us  13.208us  13.208us  cuDeviceGetName
                    0.01%  7.8190us         1  7.8190us  7.8190us  7.8190us  cuDeviceGetPCIBusId
                    0.01%  6.3480us         1  6.3480us  6.3480us  6.3480us  cuDeviceTotalMem
                    0.00%  2.1050us         3     701ns     333ns  1.4030us  cuDeviceGetCount
                    0.00%  1.1520us         2     576ns     306ns     846ns  cuDeviceGet
                    0.00%     838ns         1     838ns     838ns     838ns  cudaGetLastError
                    0.00%     493ns         1     493ns     493ns     493ns  cuModuleGetLoadingMode
                    0.00%     431ns         1     431ns     431ns     431ns  cuDeviceGetUuid
```
