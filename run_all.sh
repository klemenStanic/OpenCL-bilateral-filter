#!/bin/bash

sigma_v_all=(0.05 0.2 0.8 1.0)
sigma_s_all=(4 8 16)
img_name="lena4K"

echo "Start"
for sigma_v in ${sigma_v_all[@]}; do
    for sigma_s in ${sigma_s_all[@]}; do
        echo "Running Serial with sigma_s=$sigma_s and sigma_v=$sigma_v"
        # Serial first
        srun --reservation=fri serial/serial.o test_images/"$img_name".jpg outputs/"$img_name"_serial_"$sigma_s"_"$sigma_v".jpg "$sigma_s" "$sigma_v" --wait
        echo "-------------------------------------------------------------------"
        echo "Running Serial Naive with sigma_s=$sigma_s and sigma_v=$sigma_v"
        # Serial first
        srun --reservation=fri serial/serial_naive.o test_images/"$img_name".jpg outputs/"$img_name"_serial_naive_"$sigma_s"_"$sigma_v".jpg "$sigma_s" "$sigma_v" --wait
        echo "-------------------------------------------------------------------"
        echo "Running Parallel with sigma_s=$sigma_s and sigma_v=$sigma_v"
        cd parallel/final
        # Serial first
        srun --reservation=fri --gpus=1 parallel.o ../../test_images/"$img_name".jpg ../../outputs/"$img_name"_parallel_"$sigma_s"_"$sigma_v".jpg "$sigma_s" "$sigma_v" --wait
        echo "-------------------------------------------------------------------"
        cd ../naive
        echo "Running Parallel Naive with sigma_s=$sigma_s and sigma_v=$sigma_v"
        # Serial first
        srun --reservation=fri --gpus=1 parallel.o ../../test_images/"$img_name".jpg ../../outputs/"$img_name"_parallel_naive_"$sigma_s"_"$sigma_v".jpg "$sigma_s" "$sigma_v" --wait
        cd ../../
        echo "*******************************************************************"
    done
done
