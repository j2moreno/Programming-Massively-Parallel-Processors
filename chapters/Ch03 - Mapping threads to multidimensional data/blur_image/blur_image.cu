/*
    To compile and run: 

    apt-get install libopencv-dev

    nvcc -o blur_image blur_image.cu $(pkg-config --cflags --libs opencv4) -diag-suppress 611    
    ./blur_image <path_to_image>
*/

#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

#define BLUR_SIZE 1
#define NUM_CHANNELS 3

inline void checkError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(err);
    }
}

#define CHECK_ERROR(call) checkError((call), __FILE__, __LINE__)

__global__
void blurKernel(unsigned char* in, unsigned char* out, int width, int height) {
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = threadIdx.z;

    if (col < width && row < height ) {
        int pixVal = 0;
        int pixels = 0;

        // Get average of the surrounding BLUR_SIZE X BLUR_SIZE box 
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow){
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol){
                int currRow = row + blurRow;
                int currCol = col + blurCol;

                // Verify we have a valid image pixel 
                if (currRow >= 0 && currRow < height && currCol >= 0 && currCol < width) {
                    pixVal += in[(currRow * width + currCol) * NUM_CHANNELS + channel];
                    ++pixels;
                }
            }
        }
        out[(row * width + col) * NUM_CHANNELS + channel] = (unsigned char)(pixVal / pixels);
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
    int channels = image.channels();
    int size = image_width * image_height * channels;

    printf("Image Width: %d, Image Height: %d, Size: %d, Channels: %d\n", image_width, image_height, size, channels);

    // Allocate memory for the input and output images on host
    unsigned char* h_input_image = image.data;
    unsigned char* h_output_image = (unsigned char*) malloc(size * sizeof(unsigned char));

    // Allocate memory for the input and output images on device
    unsigned char* d_input_image, *d_output_image;
    CHECK_ERROR(cudaMalloc((void**)&d_input_image, size * sizeof(unsigned char)));
    CHECK_ERROR(cudaMalloc((void**)&d_output_image, size * sizeof(unsigned char)));

    // Copy the input image to the device
    CHECK_ERROR(cudaMemcpy(d_input_image, h_input_image, size * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Kernel launch code
    int block_dim = 16; // You can choose different block size

    // We add NUM_CHANNELS to the dimBlock z coordinate so that each thread handles 1 of 3 channels 
    dim3 dimBlock(block_dim, block_dim, NUM_CHANNELS); 
    dim3 dimGrid((image_width + block_dim - 1) / block_dim, (image_height + block_dim - 1) / block_dim, 1);

    // Print the grid and block dimensions
    printf("Block Dimensions: (%d, %d, %d)\n", dimBlock.x, dimBlock.y, dimBlock.z);
    printf("Grid Dimensions: (%d, %d, %d)\n", dimGrid.x, dimGrid.y, dimGrid.z);

    blurKernel<<<dimGrid, dimBlock>>>(d_input_image, d_output_image, image_width, image_height);
    CHECK_ERROR(cudaGetLastError());

    // Copy the output back to the host
    CHECK_ERROR(cudaMemcpy(h_output_image, d_output_image, size * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // Create a blurred image using OpenCV
    cv::Mat blurred_image(image_height, image_width, CV_8UC3, h_output_image);

    // Save the blurred image
    cv::imwrite("blurred_image.jpg", blurred_image);

    // Free the device memory
    cudaFree(d_input_image);
    cudaFree(d_output_image);

    // Free the host memory
    free(h_output_image);

    return 0;
}
