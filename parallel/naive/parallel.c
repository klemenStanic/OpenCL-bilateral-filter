#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>
#include <CL/cl.h>


#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"


#define MAX_SOURCE_SIZE	16384
#define WORKGROUP_SIZE  (1024)

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

    // Calculate window size from sigma_s
    int w_size = 2 * sigma_s;

    // Read image
    int width, height, cpp;
    unsigned char *image = stbi_load(image_file, &width, &height, &cpp, 0);
    unsigned char *image_out = calloc(width * height * cpp, sizeof(unsigned char));

    //-------------------------------- Initializing GPU --------------------------------

    // OpenCL initialization
	
    // read kernel source
    FILE* fp;
    char* source_str;
    size_t source_size;

	fp = fopen("bilateral_filter.cl", "r");
	if (!fp)
	{
		fprintf(stderr, " :-(#\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	source_str[source_size] = '\0';
	fclose(fp);

    
    // Get platforms
    cl_uint num_platforms;
    cl_int clStatus;
    clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*num_platforms);
    clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);

    //Get platform devices
    cl_uint num_devices;
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    
    num_devices = 1; // limit to one device
    cl_device_id *devices = (cl_device_id *)malloc(sizeof(cl_device_id)*num_devices);
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);

    // Create a context
	cl_context context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &clStatus);

	// Create a command queue
	cl_command_queue command_queue = clCreateCommandQueue(context, devices[0], 0, &clStatus);

	// create and build a program
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, NULL, &clStatus);
	clStatus = clBuildProgram(program, num_devices, devices, NULL, NULL, NULL);
    
    // Log
	size_t build_log_len;
	char *build_log;
	clStatus = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
    if (build_log_len > 2)
    {
        build_log =(char *)malloc(sizeof(char)*(build_log_len+1));
        clStatus = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 
                                        build_log_len, build_log, NULL);
        printf("%s", build_log);
        free(build_log);
        return 1;
    }
    
    
    // divide work among threads
    int size = width * height * cpp;
    size_t local_item_size = WORKGROUP_SIZE;
    size_t global_item_size = ((int) (size / WORKGROUP_SIZE) + 1) * WORKGROUP_SIZE;	

    
    // Precompute kernels
    //float* kernel_1D = precompute_guassian_kernels_1D(sigma_v);
    //float* kernel_2D = precompute_gaussian_kernels_2D(sigma_s);    
    

    // Alocate device memory
	cl_mem image_in_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * width * height * cpp, image, &clStatus);
	cl_mem image_out_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * width * height * cpp, NULL, &clStatus);
    //cl_mem w_size_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), w_size, &clStatus);
    
    if(clStatus < 0) {
      printf("Error at allocating device memory");
      exit(1);
    }

    // Create kernels
	cl_kernel kernel = clCreateKernel(program, "BilateralFilter", &clStatus);
	clStatus = clSetKernelArg(kernel, 0, sizeof(cl_mem), &image_in_d);
    clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem), &image_out_d);
    clStatus = clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&w_size);
    clStatus = clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&width);
    clStatus = clSetKernelArg(kernel, 4, sizeof(cl_int), (void*)&height);
    clStatus = clSetKernelArg(kernel, 5, sizeof(cl_float), (void*)&sigma_v);

    clFinish(command_queue);
    if(clStatus < 0) {
      printf("Error creating kernel and loading args.");
      exit(1);
    }


    //-------------------------------- Executing on the GPU --------------------------------
    double proc_start = omp_get_wtime();
    
    // trigger kernel execution
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    clFinish(command_queue);
    if (clStatus < 0) {
        printf("%d\n", clStatus);
        printf("Error at executing on GPU.");
        exit(1);
    }



    double proc_end = omp_get_wtime();

    //-------------------------------- Getting image from the GPU --------------------------------

    clStatus = clEnqueueReadBuffer(command_queue, image_out_d, CL_TRUE, 0, sizeof(unsigned char) * width * height * cpp, image_out, 0, NULL, NULL);
    if (clStatus < 0) {
        printf("%d\n", clStatus);
        printf("Error retrieving data from GPU.");
        exit(1);
    }

    clFinish(command_queue);

    

    // Write image
    char *filename = output;
    int status_image = stbi_write_jpg(filename, width, height, 3, image_out,  100);
    printf("Image writing status: %d\n", status_image);
    
    // Print times
    printf("Time: %.5f s\n", proc_end-proc_start);
}
