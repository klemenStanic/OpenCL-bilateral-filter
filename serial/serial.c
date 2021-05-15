#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

float gaussian_kernel_2D(int delta_y, int delta_x, int sigma_s){
    return exp(-(pow(delta_x, 2) + pow(delta_y, 2))/(2 * pow(sigma_s, 2)));
}

float gaussian_kernel_1D(int delta, float sigma_v){
    return exp(-(pow(delta, 2)/(2 * pow(sigma_v, 2))));
}

float *precompute_guassian_kernels_1D(float sigma_v){
    float* kernel_1D = calloc(255, sizeof(float));

    for (int i = 0; i < 255; i++){
        kernel_1D[i] = gaussian_kernel_1D(i, sigma_v);
    }
    return kernel_1D;
}

float *precompute_gaussian_kernels_2D(int sigma_s){
    int w = sigma_s * 2;
    float* kernel_2D = calloc((w + 1) * (w + 1), sizeof(float));

    for (int x = 0; x <= w; x++){
        for (int y = 0; y <= w; y++){
            kernel_2D[y * (w + 1) + x] = gaussian_kernel_2D(y, x, sigma_s);
        }
    }

    return kernel_2D;
}

void process(unsigned char *image, unsigned char *image_out, int width, int height, int cpp, int sigma_s, float sigma_v){

    
    printf("width: %d, height: %d, cpp: %d, sigma_s: %d, sigma_v: %f\n", width, height, cpp, sigma_s, sigma_v);
    int w = sigma_s * 2;

    // Precompute kernels
    float* kernel_1D = precompute_guassian_kernels_1D(sigma_v);
    float* kernel_2D = precompute_gaussian_kernels_2D(sigma_s);

    // TODO: fix indexes for edges
    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            float W_r = 0;
            float W_g = 0;
            float W_b = 0;

            float img_r = 0;
            float img_g = 0;
            float img_b = 0;

            for (int r = y - w; r <= y + w; r++){
                for (int s = x - w; s <= x + w; s++ ){
                    if (s < 0 || s >= width || r < 0 || r >= height){
                        continue;
                    }
                    float gaussian = kernel_2D[abs(r-y) * (w + 1) + abs(s-x)];
                    //float gaussian = gaussian_kernel(r - y, s - x, sigma_s);

                    unsigned char delta_im = abs(image[(y * width + x) * cpp] - image[(r * width + s) * cpp]);
                    W_r += gaussian * kernel_1D[delta_im]; 
                    img_r += image[(r * width + s) * cpp] * gaussian * kernel_1D[delta_im];
                    

                    delta_im = abs(image[(y * width + x) * cpp + 1] - image[(r * width + s) * cpp + 1]); 
                    W_g += gaussian * kernel_1D[delta_im]; 
                    img_g += image[(r * width + s) * cpp + 1] * gaussian * kernel_1D[delta_im];
                

                    delta_im = abs(image[(y * width + x) * cpp + 2] - image[(r * width + s) * cpp + 2]); 
                    W_b += gaussian *  kernel_1D[delta_im]; 
                    img_b += image[(r * width + s) * cpp + 2] * gaussian *  kernel_1D[delta_im];
                }
            }
            
            //printf("B4: %f\n", (image_out[(y * width + x) * cpp] / W_r));
            //W_r /= 10;
            //W_g /= 10;
            //W_b /= 10;
            image_out[(y * width + x) * cpp] = (unsigned char) (img_r / W_r);
            image_out[(y * width + x) * cpp + 1] = (unsigned char) (img_g / W_g);
            image_out[(y * width + x) * cpp + 2] = (unsigned char) (img_b / W_b);
            //printf("x: %d, y: %d\n", x, y);
            //printf("W_r: %f\nright_side: %d\n--------------------------------------------------------------------------\n", W_r, image_out[(y * width + x) * cpp]);
        }
    }
}

int main(int argc, char **argv)
{
    char *image_file = argv[1];
    char *output = argv[2];
    
    int sigma_s = 0;
    float sigma_v = 0.f;
 

    if (argc > 1) {
        image_file = argv[1];
        output = argv[2];
        sigma_s = atoi(argv[3]);
        sigma_v = atof(argv[4]) * 255;
    }
    else {
        fprintf(stderr, "Not enough arguments\n");
        fprintf(stderr, "Usage: %s <IMAGE_PATH>\n", argv[0]);
        exit(1);
    }

    // Read image
    int width, height, cpp;
    unsigned char *image = stbi_load(image_file, &width, &height, &cpp, 0);
    unsigned char *image_out = calloc(width * height * cpp, sizeof(unsigned char));



    double proc_start = omp_get_wtime();
    // Process
    process(image, image_out, width, height, cpp, sigma_s, sigma_v);

    double proc_end = omp_get_wtime();
    printf("Time: %.5f s\n", proc_end-proc_start);

    // Write image
    char *filename = output;
    int status_image = stbi_write_jpg(filename, width, height, 3, image_out,  100);
    printf("Image writing status: %d\n", status_image);

}


// gcc serial.c -O2 -lm -fopenmp -o serial.o
// srun --reservation=fri serial.o test_images/lena15.jpg outputs/test_output.jpg 10 10






