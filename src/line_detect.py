#!/usr/bin/python

import os
import os.path
import time

import cv2
import numpy as np
import math

### Setup ###
image_data_path = os.path.abspath('../CRBD/Images')
gt_data_path = os.path.abspath('../CRBD/GT data')
image_out_path = os.path.abspath('../img')

NUMBER_OF_STRIPS = 10             # How many strips the image is split into
SUM_THRESH = 5                    # How much green in a strip before it's a plant
DIFF_NOISE_THRESH = 4             # How close can two sections be?

HOUGH_RHO = 8                     # Distance resolution of the accumulator in pixels
HOUGH_ANGLE = math.pi/180         # Angle resolution of the accumulator in radians
HOUGH_THRESH = 8                  # Accumulator threshold parameter. Only those lines are returned that get enough votes

ANGLE_THRESH = math.pi*(30.0/180) # How steep angles the crop rows can be in radians


use_camera = False
#view_all_steps = False
save_images = False
strip_to_save = 3
timing = False


def main():
    
    if use_camera == False:
        
        diff_times = []
        
        for image_name in sorted(os.listdir(image_data_path)):
            
            start_time = time.time()
            
            image_path = os.path.join(image_data_path, image_name)
            
            image_in = cv2.imread(image_path)
            crop_lines = crop_row_detect(image_in)
            
            if timing == False:
                cv2.destroyAllWindows()
                #print(crop_lines)
                cv2.imshow(image_name, cv2.addWeighted(image_in, 1, crop_lines, 1, 0.0))
                #cv2.imshow(image_name, image_in)
                #cv2.imshow("detected lines", crop_lines)
                
                print('Press any key to continue...')
                while cv2.waitKey(1) < 0:
                    pass
                
            
            ### Timing ###
            else:
                diff_times.append(time.time() - start_time)
                mean = 0
                for diff_time in diff_times:
                    mean += diff_time
                
                #if timing == True:
                #    print('elapsed time = {0}'.format(diff_times[-1]))
                #    print('average time = {0}'.format(1.0 * mean / len(diff_times)))
            
                    
        ### Display Timing ###
        print('max time = {0}'.format(max(diff_times)))
        print('ave time = {0}'.format(1.0 * mean / len(diff_times)))
        
        while cv2.waitKey(1) < 0:
            pass
            
    else:
        capture = cv2.VideoCapture(0)
        
        while cv2.waitKey(1) < 0:
            _, image_in = capture.read()
            image_out = crop_row_detect(image_in)
            cv2.imshow('webcam crop rows', image_out)
    
        capture.release()
        
    cv2.destroyAllWindows()



def crop_row_detect(image_in):
    
    save_image('0_image_in.jpg', image_in)
    
    ### Grayscale Transform ###
    image_edit = grayscale_transform(image_in)
    save_image('1_image_gray.jpg', image_edit)
        
    ### Binarization ###
    _, image_edit = cv2.threshold(image_edit, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    save_image('2_image_bin.jpg', image_edit)
    
    ### Stripping ###
    crop_points = strip_process(image_edit)
    save_image('8_crop_points.jpg', crop_points)
    
    ### Hough Transform ###
    crop_lines = crop_point_hough(crop_points)
    save_image('9_image_hough.jpg', cv2.addWeighted(image_in, 1, crop_lines, 1, 0.0))
    
    return crop_lines
    
    
def save_image(image_name, image_data):
    if save_images == True:
        cv2.imwrite(os.path.join(image_out_path, image_name), image_data)

   
def grayscale_transform(image_in):
    b, g, r = cv2.split(image_in)
    return 2*g - r - b


def strip_process(image_edit):
    
    height = len(image_edit)
    width = len(image_edit[0])
    
    strip_height = height / NUMBER_OF_STRIPS
    crop_points = np.zeros((height, width), dtype=np.uint8)
    
    for strip_number in range(NUMBER_OF_STRIPS):
        image_strip = image_edit[(strip_number*strip_height):((strip_number+1)*strip_height-1), :]
        
        
        if strip_number == strip_to_save:
            save_image('4_image_strip_4.jpg', image_strip)
        
        v_sum = [0] * width
        v_thresh = [0] * width
        v_diff = [0] * width
        v_mid = [0] * width
        
        diff_start = 0
        diff_end = 0
        diff_end_found = True
        
        for col_number in range(width):
            
            ### Vertical Sum ###
            v_sum[col_number] = sum(image_strip[:, col_number]) / 255
            
            ### Threshold ###
            if v_sum[col_number] >= SUM_THRESH:
                v_thresh[col_number] = 1
            else:
                v_thresh[col_number] = 0
            
            ### Differential with Noise Reduction ###
            if v_thresh[col_number] > v_thresh[col_number - 1]:
                v_diff[col_number] = 1
                if (col_number - diff_end) > DIFF_NOISE_THRESH:
                    diff_start = col_number
                    diff_end_found = False
                
            elif v_thresh[col_number] < v_thresh[col_number - 1]:
                v_diff[col_number] = -1
                
                if (col_number - diff_start) > DIFF_NOISE_THRESH:
                    v_mid[diff_start + (col_number-diff_start)/2] = 1
                    diff_end = col_number
                    diff_end_found = True
        
        if save_images == True:
            if strip_number == strip_to_save:
                print(v_sum)
                print(v_thresh)
                print(v_diff)
                print(v_mid)
        
        crop_points[(strip_number*strip_height), :] = v_mid
        crop_points *= 255
        
        #image_edit[(strip_number*strip_height):((strip_number+1)*strip_height-1), :] = image_strip
        
    return crop_points


def crop_point_hough(crop_points):
    
    height = len(crop_points)
    width = len(crop_points[0])
    
    #crop_line_data = cv2.HoughLinesP(crop_points, 1, math.pi/180, 2, 10, 10)
    crop_line_data = cv2.HoughLines(crop_points, HOUGH_RHO, HOUGH_ANGLE, HOUGH_THRESH)
    
    crop_lines = np.zeros((height, width, 3), dtype=np.uint8)
    
    if crop_line_data != None:
        crop_line_data = crop_line_data[0]
        print(crop_line_data)
        
        if len(crop_line_data[0]) == 2:
            for [rho, theta] in crop_line_data:
                #print(rho, theta)
                if (theta <= ANGLE_THRESH) or (theta >= math.pi-ANGLE_THRESH):
                    a = math.cos(theta)
                    b = math.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    point1 = (int(round(x0+1000*(-b))), int(round(y0+1000*(a))))
                    point2 = (int(round(x0-1000*(-b))), int(round(y0-1000*(a))))
                    cv2.line(crop_lines, point1, point2, (0, 0, 255), 2)
                
        elif len(crop_line_data[0]) == 4:
            for [x0, y0, x1, y1] in crop_line_data:
                cv2.line(crop_lines, (x0, y0), (x1, y1), (0, 0, 255), 2)
    else:
        print("No lines found")
    
    return crop_lines
    
    
main()
