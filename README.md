# OpenCL Bilateral Filter
Image smoothing with bilateral filter, implemented to run on GPU with OpenCL library.

This repository contains the code implemented as a part of a seminal assignment of the High Performance
Computing Masters studies course at the Faculty of Computer and Information Science, University of Ljubljana.

## Contents:
- Serial implementation resides in folder `serial`. The `serial_naive.c` file contains more naive approach, where gaussian kernel values are computed on each iteration. In `serial.c`, we precompute the kernel values and reuse them through iterations.

To build and execute, run:
```
gcc serial.c -O2 -lm -fopenmp -o serial.o
./serial.o <image_in_path> <image_out_path> sigma_s sigma_v
```

- OpenCL implementation resides in the folder `parallel/final`. The naive version is in `parallel/naive` folder.
To build and execute, run:
```
module load CUDA
gcc parallel.c -O2 -lm -lOpenCL -fopenmp -o parallel.o
srun --gpus=1 parallel.c <image_in_path> <image_out_path> sigma_s sigma_v
```

- Folder `outputs` includes images, processed during testing.
- Folder `test_images` includes some image files, used for testing.
- *.log files, containing test outputs.


## Results:
Running on `lena_small.jpg`, a 256x256 jpg file:

| sigma_s \ sigma_v |                            0.05                            |                             0.2                            |                             0.8                            |                             1.0                            |
|:-----------------:|:----------------------------------------------------------:|:----------------------------------------------------------:|:----------------------------------------------------------:|:----------------------------------------------------------:|
|         4         |  s: 0.85514 s <br />s-n: 17.25884 s <br />p: 0.07422 s <br />p-n: 0.12338 s  |  s: 0.80891 s <br />s-n: 16.81087 s <br />p: 0.07440 s <br />p-n: 0.12357 s  |  s: 0.80182 s  <br />s-n: 16.49050 s <br />p: 0.07420 s <br />p-n: 0.12341 s |  s: 0.80199 s  <br />s-n: 18.00163 s <br />p: 0.07420 s <br />p-n: 0.12327 s |
|         8         |  s: 3.25598 s <br />s-n: 54.25504 s <br />p: 0.27598 s <br />p-n: 0.46052 s  |  s: 2.92055 s <br />s-n: 55.94103 s <br />p: 0.27607 s <br />p-n: 0.46074 s  |  s: 2.89884 s <br />s-n: 52.13937 s <br />p: 0.27602 s <br />p-n: 0.46046 s  |  s: 2.90287 s <br />s-n: 51.66329 s <br />p: 0.27600 s <br />p-n: 0.46032 s  |
|         16        | s: 11.11582 s <br />s-n: 229.01741 s <br />p: 0.98500 s <br />p-n: 1.75122 s | s: 10.87755 s <br />s-n: 235.22470 s <br />p: 0.98483 s <br />p-n: 1.75066 s | s: 10.88048 s <br />s-n: 233.11871 s <br />p: 0.98488 s <br />p-n: 1.75093 s | s: 10.89152 s <br />s-n: 230.59420 s <br />p: 0.98480 s <br />p-n: 1.75098 s |

Running on `lena4K.jpg`, a 3840×2160 jpg file:
| sigma_s \ sigma_v |                             0.05                            |                             0.2                             |                             0.8                            |                             1.0                            |
|:-----------------:|:-----------------------------------------------------------:|:-----------------------------------------------------------:|:----------------------------------------------------------:|:----------------------------------------------------------:|
|         4         |  s: 11.85443 s <br />s-n: 248.80349 s <br />p: 0.86189 s <br />p-n: 1.80670 s |  s: 11.80305 s <br />s-n: 246.05509 s <br />p: 0.86180 s <br />p-n: 1.80688 s | s: 11.80456 s <br />s-n: 241.20787 s <br />p: 0.86189 s <br />p-n: 1.80675 s | s: 11.86722 s <br />s-n: 243.73240 s <br />p: 0.86181 s <br />p-n: 1.80657 s |
|         8         | s: 43.41393 s <br />s-n: 970.30759 s <br />p: 3.38738 s <br />p-n: 6.83974 s | s: 42.47110 s  <br />s-n: 963.65313 s <br />p: 3.38741 s <br />p-n: 6.83875 s | s: 42.79564 s <br />s-n: 897.20665 s <br />p: 3.38744 s <br />p-n: 6.83622 s | s: 42.54512 s <br />s-n: 961.09238 s <br />p: 3.38736 s <br />p-n: 6.83528 s |
|         16        |  s: 174.94544 s <br />s-n: TIMEOUT <br />p: 13.17960 s <br />p-n: 26.49231 s  |  s: 165.13394 s <br />s-n: TIMEOUT <br />p: 13.17988 s <br />p-n: 26.48724 s  |  s: 164.59882 s <br />s-n: TIMEOUT <br />p: 13.18050 s <br />p-n: 26.48660 s | s: 165.02416 s <br />s-n: TIMEOUT  <br />p: 13.18013 s <br />p-n: 26.48193 s |

*s* stands for serial algorithm, running on a single CPU, *s-n* is a naive version of this same algorithm.
Similarily, *p* stands for the parallel algorithm (running on 1 GPU with OpenCL), *p-n* is a naive version of the parallel algorithm.

## Speedups
| sigma_s | s-n / p-n | s / p-n | s / p |
|---------|-----------|---------|-------|
| 4       | 139.88    | 6.93    | 11.52 |
| 8       | 117.81    | 7.07    | 11.79 |
| 16      | 130.77    | 6.34    | 11.28 |
