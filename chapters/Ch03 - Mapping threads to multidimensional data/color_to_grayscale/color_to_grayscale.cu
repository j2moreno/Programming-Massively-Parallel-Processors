/*
    To compile and run: 

    apt-get install libopencv-dev

    nvcc -o color_to_grayscale color_to_grayscale.cu $(pkg-config --cflags --libs opencv4) -diag-suppress 611    
    ./color_to_greyscale <path_to_image>
*/

#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

#define NUM_CHANNELS 3

inline void checkError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(err);
    }
}

#define CHECK_ERROR(call) checkError((call), __FILE__, __LINE__)

__global__
void color_to_grayscale_conversion(unsigned char* in, unsigned char* out, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height ) {

        int grey_offset = row * width + col;
        int rgb_offset = grey_offset * NUM_CHANNELS;

        unsigned char r = in[rgb_offset + 0];
        unsigned char g = in[rgb_offset + 1];
        unsigned char b = in[rgb_offset + 2];

        out[grey_offset] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

// input dimensions: 575 × 419 
int main(int argc, char *argv[]) {

    // Check if the correct number of arguments are passed
    if (argc != 2) {
        printf("Usage: %s <image_path>\n", argv[0]);
        return 1;
    }

    char* image_path = argv[1];

    // Load the image using OpenCV
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        printf("Could not open or find the image\n");
        return 1;
    }

    int image_width = image.cols;
    int image_height = image.rows;
    int size = image_width * image_height;

    printf("Image Width: %d, Image Height: %d, Size: %d\n", image_width, image_height, size);

    // Allocate memory for the input and output images on host
    unsigned char* h_input_image = image.data;
    unsigned char* h_output_image = (unsigned char*) malloc(size * sizeof(unsigned char));

     // Allocate memory for the input and output images on device
    unsigned char* d_input_image, *d_output_image;
    CHECK_ERROR(cudaMalloc((void**)&d_input_image, NUM_CHANNELS * size * sizeof(unsigned char)));
    CHECK_ERROR(cudaMalloc((void**)&d_output_image, size * sizeof(unsigned char)));

    // Copy the input image to the device
    CHECK_ERROR(cudaMemcpy(d_input_image, h_input_image, NUM_CHANNELS * size * sizeof(unsigned char), cudaMemcpyHostToDevice));

     // Kernel launch code
    int block_dim = 16; // You can choose different block size
    dim3 dimBlock(block_dim, block_dim, 1);
    dim3 dimGrid((image_width + block_dim - 1) / block_dim, (image_height + block_dim - 1) / block_dim, 1);

    // Print the grid and block dimensions
    printf("Block Dimensions: (%d, %d, %d)\n", dimBlock.x, dimBlock.y, dimBlock.z);
    printf("Grid Dimensions: (%d, %d, %d)\n", dimGrid.x, dimGrid.y, dimGrid.z);

    color_to_grayscale_conversion<<<dimGrid, dimBlock>>>(d_input_image, d_output_image, image_width, image_height);
    CHECK_ERROR(cudaGetLastError());

    // Copy the output back to the host
    CHECK_ERROR(cudaMemcpy(h_output_image, d_output_image, size * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // Create a grayscale image using OpenCV
    cv::Mat gray_image(image_height, image_width, CV_8UC1, h_output_image);

    // Save the grayscale image
    cv::imwrite("grayscale_image.jpg", gray_image);

    // Free the device memory
    cudaFree(d_input_image);
    cudaFree(d_output_image);

    // Free the host memory
    free(h_output_image);

    return 0;
}