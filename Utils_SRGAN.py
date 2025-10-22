from keras.layers import Lambda
import tensorflow as tf
from skimage import data, io, filters
import numpy as np
from numpy import array
from numpy.random import randint
import os
import sys
import PIL
import imageio
from PIL import Image
import cv2


import matplotlib.pyplot as plt
plt.switch_backend('agg')

# Subpixel Conv will upsample from (h, w, c) to (h/r, w/r, c/r^2)
def SubpixelConv2D(input_shape, scale=4):
    def subpixel_shape(input_shape):
        dims = [input_shape[0],input_shape[1] * scale,input_shape[2] * scale,int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape
    
    def subpixel(x):
        return tf.depth_to_space(x, scale)
        
    return Lambda(subpixel, output_shape=subpixel_shape)
    
    
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)


# Takes LR images and save respective HR images
def plot_test_generated_images(generator, x_test_lr, figsize=(5, 5)):
    
    examples = x_test_lr.shape[0]
    image_batch_lr = x_test_lr
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)
    
    for index in range(examples):
    
        plt.figure(figsize=figsize)
    
        plt.imshow(generated_image[index], interpolation='nearest')
        plt.axis('off')
        
        plt.tight_layout()
        #file_name = output_dir + 'high_res_result_image_%d.png' % index
        #cv2.imwrite(file_name, generated_image[index])
        #plt.savefig(output_dir + 'high_res_result_image_%d.png' % index)
    
        #plt.show()
        
        return generated_image[index]
        
def pad_img(img):
    
    #extract height and width
    h, w, _ = img.shape
    
    #check which is bigger, height or width
    if h > w:
        max_dim = h
    else:
        max_dim = w
    
    #check if the max dimension is even or not
    if max_dim%2 !=0:
        max_dim += 1
    
    pad_top = (max_dim - h)//2
    pad_bottom = (max_dim - h)//2
    pad_left = (max_dim - w)//2
    pad_right = (max_dim - w)//2
    
    img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)
    
    resized_img = cv2.resize(img_padded, (96, 96), interpolation = cv2.INTER_CUBIC)
    
    return resized_img

def LOAD_DATA_TEST(input_low_res_dir):
    
    #read image
    #img = cv2.imread(input_low_res_dir)
    
    #padding image to make it square shaped
    test_image = pad_img(input_low_res_dir)
    
    ##lr
    downscale = 1
    res = (test_image.shape[0]//downscale, test_image.shape[1]//downscale)
    i = test_image
    i = cv2.resize(i, res, interpolation = cv2.INTER_CUBIC)
    images = [np.array(PIL.Image.fromarray(i))]
    images_lr = array(images)

    #normalize
    normalized_img = (images_lr.astype(np.float32) - 127.5)/127.5
    
    return normalized_img
    

