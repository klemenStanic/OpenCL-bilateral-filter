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

| sigma_v \ sigma_s |                            0.05                            |                             0.2                            |                             0.8                            |                             1.0                            |
|:-----------------:|:----------------------------------------------------------:|:----------------------------------------------------------:|:----------------------------------------------------------:|:----------------------------------------------------------:|
|         4         |  s: 0.85514 s \ s-n: 19.25884 s p: 0.07422 s p-n: 0.12338 s  |  s: 0.80891 s s-n: 15.21087 s p: 0.07440 s p-n: 0.12357 s  |  s: 0.80182 s  s-n: 16.49050 s p: 0.07420 s p-n: 0.12341 s |  s: 0.80199 s  s-n: 19.90163 s p: 0.07420 s p-n: 0.12327 s |
|         8         |  s: 3.25598 s s-n: 72.25504 s p: 0.27598 s p-n: 0.46052 s  |  s: 2.92055 s s-n: 66.94103 s p: 0.27607 s p-n: 0.46074 s  |  s: 2.89884 s s-n: 52.13937 s p: 0.27602 s p-n: 0.46046 s  |  s: 2.90287 s s-n: 51.66329 s p: 0.27600 s p-n: 0.46032 s  |
|         16        | s: 13.11582 s s-n: 276.01741 s p: 0.98500 s p-n: 1.75122 s | s: 10.87755 s s-n: 255.22470 s p: 0.98483 s p-n: 1.75066 s | s: 10.88048 s s-n: 233.11871 s p: 0.98488 s p-n: 1.75093 s | s: 10.89152 s s-n: 230.59420 s p: 0.98480 s p-n: 1.75098 s |

Running on `lena4K.jpg`, a 3840×2160 jpg file:
| sigma_v \ sigma_s |                             0.05                            |                             0.2                             |                             0.8                            |                             1.0                            |
|:-----------------:|:-----------------------------------------------------------:|:-----------------------------------------------------------:|:----------------------------------------------------------:|:----------------------------------------------------------:|
|         4         |  s: 11.85443 s s-n: 248.80349 s p: 0.86189 s p-n: 1.80670 s |  s: 11.80305 s s-n: 256.05509 s p: 0.86180 s p-n: 1.80688 s | s: 11.80456 s s-n: 241.20787 s p: 0.86189 s p-n: 1.80675 s | s: 11.86722 s s-n: 214.73240 s p: 0.86181 s p-n: 1.80657 s |
|         8         | s: 43.41393 s s-n: 1070.30759 s p: 3.38738 s p-n: 6.83974 s | s: 42.47110 s  s-n: 963.65313 s p: 3.38741 s p-n: 6.83875 s | s: 42.79564 s s-n: 897.20665 s p: 3.38744 s p-n: 6.83622 s | s: 42.54512 s s-n: 761.09238 s p: 3.38736 s p-n: 6.83528 s |
|         16        |  s: 174.94544 s s-n: TIMEOUT p: 13.17960 s p-n: 26.49231 s  |  s: 165.13394 s s-n: TIMEOUT p: 13.17988 s p-n: 26.48724 s  |  s: 164.59882 s s-n: TIMEOUT p: 13.18050 s p-n: 26.48660 s | s: 165.02416 s s-n: TIMEOUT  p: 13.18013 s p-n: 26.48193 s |
