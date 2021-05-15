float gaussian_kernel_2D(int delta_y, int delta_x, int sigma_s){
    return exp(-(pow(delta_x, 2.0f) + pow(delta_y, 2.0f))/(2 * pow(sigma_s, 2.0f)));
}

float gaussian_kernel_1D(int delta, float sigma_v){
    float delta_ = (float) delta;
    return exp(-(pow(delta_, 2.0f)/(2.0f * pow(sigma_v, 2.0f))));
}


__kernel void BilateralFilter(__global unsigned char* image_in,
                          __global unsigned char* image_out,
                          int w, int width, int height, float sigma_v)
{

    int gi = get_global_id(0);
    int color_indx = gi % 3;

    int pixl = (int) (gi / 3);
    int x = pixl % width;
    int y = (int) (pixl / width);

    float W_r = 0;
    float W_g = 0;
    float W_b = 0;

    float img_r = 0;
    float img_g = 0;
    float img_b = 0;

    if (gi < width * height * 3){
        for (int r = y - w; r <= y + w; r++){
            for (int s = x - w; s <= x + w; s++ ){
                if (s < 0 || s >= width || r < 0 || r >= height){
                    continue;
                }

                //float gaussian = kernel_2D[abs(r-y) * (w + 1) + abs(s-x)];
                float gaussian = gaussian_kernel_2D(r - y, s - x, w / 2);
                unsigned char delta_im;
                if (color_indx == 0){
                    delta_im = abs(image_in[gi] - image_in[(r * width + s) * 3]);
                    W_r += gaussian * gaussian_kernel_1D(delta_im, sigma_v); 
                    img_r += image_in[(r * width + s) * 3] * gaussian * gaussian_kernel_1D(delta_im, sigma_v);
                } else if (color_indx == 1){
                    delta_im = abs(image_in[gi] - image_in[(r * width + s) * 3 + 1]); 
                    W_g += gaussian * gaussian_kernel_1D(delta_im, sigma_v); 
                    img_g += image_in[(r * width + s) * 3 + 1] * gaussian * gaussian_kernel_1D(delta_im, sigma_v);
                } else {    
                    delta_im = abs(image_in[gi] - image_in[(r * width + s) * 3 + 2]); 
                    W_b += gaussian *  gaussian_kernel_1D(delta_im, sigma_v); 
                    img_b += image_in[(r * width + s) * 3 + 2] * gaussian *  gaussian_kernel_1D(delta_im, sigma_v);
                }
            }
        }
        if (color_indx == 0){
            image_out[gi] = (unsigned char) (img_r / W_r);
        } else if (color_indx == 1){
            image_out[gi] = (unsigned char) (img_g / W_g);
        } else {
            image_out[gi] = (unsigned char) (img_b / W_b);
        }
    }

    
    barrier(CLK_LOCAL_MEM_FENCE);
}