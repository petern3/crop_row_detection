#!/usr/bin/python
"""
line_detect_2.py

Peter Nicholls


This program tests a crop-row detection algorithm developed in 2016.
See README.md for more info
"""

import os
import os.path
import time

import cv2
import numpy as np
import math

### Setup ###
image_data_path = os.path.abspath('../CRBD/Images')
gt_data_path = os.path.abspath('../CRBD/GT data')
image_out_path = os.path.abspath('../img/algorithm_2')

use_camera = False  # whether or not to use the test images or camera
images_to_save = [] # which test images to save
timing = False      # whether to time the test images

curr_image = 0 # global counter

HOUGH_RHO = 2                      # Distance resolution of the accumulator in pixels
HOUGH_ANGLE = math.pi*4.0/180     # Angle resolution of the accumulator in radians
HOUGH_THRESH_MAX = 100             # Accumulator threshold parameter. Only those lines are returned that get enough votes
HOUGH_THRESH_MIN = 10
HOUGH_THRESH_INCR = 1

NUMBER_OF_ROWS = 3  # how many crop rows to detect

THETA_SIM_THRESH = math.pi*(6.0/180)   # How similar two rows can be
RHO_SIM_THRESH = 8   # How similar two rows can be
ANGLE_THRESH = math.pi*(30.0/180) # How steep angles the crop rows can be in radians


### Functions ###
def main():

    if use_camera == False:

        diff_times = []

        for image_name in sorted(os.listdir(image_data_path)):
            global curr_image
            curr_image += 1

            start_time = time.time()

            image_path = os.path.join(image_data_path, image_name)
            image_in = cv2.imread(image_path)

            crop_lines = crop_row_detect(image_in)

            if timing == False:
                cv2.imshow(image_name, cv2.addWeighted(image_in, 1, crop_lines, 1, 0.0))

                print('Press any key to continue...')
                cv2.waitKey()
                cv2.destroyAllWindows()


            ### Timing ###
            else:
                diff_times.append(time.time() - start_time)
                mean = 0
                for diff_time in diff_times:
                    mean += diff_time

        ### Display Timing ###
        print('max time = {0}'.format(max(diff_times)))
        print('ave time = {0}'.format(1.0 * mean / len(diff_times)))

        cv2.waitKey()

    else:  # use camera. Hasn't been tested on a farm.
        capture = cv2.VideoCapture(0)

        while cv2.waitKey(1) < 0:
            _, image_in = capture.read()
            crop_lines = crop_row_detect(image_in)
            cv2.imshow("Webcam", cv2.addWeighted(image_in, 1, crop_lines, 1, 0.0))

        capture.release()

    cv2.destroyAllWindows()



def crop_row_detect(image_in):
    '''Inputs an image and outputs the lines'''

    save_image('0_image_in', image_in)

    ### Grayscale Transform ###
    image_edit = grayscale_transform(image_in)
    save_image('1_image_gray', image_edit)

    ### Skeletonization ###
    skeleton = skeletonize(image_edit)
    save_image('2_image_skeleton', skeleton)

    ### Hough Transform ###
    (crop_lines, crop_lines_hough) = crop_point_hough(skeleton)

    save_image('3_image_hough', cv2.addWeighted(image_in, 1, crop_lines_hough, 1, 0.0))
    save_image('4_image_lines', cv2.addWeighted(image_in, 1, crop_lines, 1, 0.0))

    return crop_lines


def save_image(image_name, image_data):
    '''Saves image if user requests before runtime'''
    if curr_image in images_to_save:
        image_name_new = os.path.join(image_out_path, "{0}_{1}.jpg".format(image_name, str(curr_image) ))

        cv2.imwrite(image_name_new, image_data)


def grayscale_transform(image_in):
    '''Converts RGB to Grayscale and enhances green values'''
    b, g, r = cv2.split(image_in)
    return 2*g - r - b

def skeletonize(image_in):
    '''Inputs and grayscale image and outputs a binary skeleton image'''
    size = np.size(image_in)
    skel = np.zeros(image_in.shape, np.uint8)

    ret, image_edit = cv2.threshold(image_in, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    done = False

    while not done:
        eroded = cv2.erode(image_edit, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(image_edit, temp)
        skel = cv2.bitwise_or(skel, temp)
        image_edit = eroded.copy()

        zeros = size - cv2.countNonZero(image_edit)
        if zeros == size:
            done = True

    return skel


def crop_point_hough(crop_points):
    '''Iterates though Hough thresholds until optimal value found for
       the desired number of crop rows. Also does filtering.
    '''

    height = len(crop_points)
    width = len(crop_points[0])

    hough_thresh = HOUGH_THRESH_MAX
    rows_found = False

    while hough_thresh > HOUGH_THRESH_MIN and not rows_found:
        crop_line_data = cv2.HoughLines(crop_points, HOUGH_RHO, HOUGH_ANGLE, hough_thresh)

        crop_lines = np.zeros((height, width, 3), dtype=np.uint8)
        crop_lines_hough = np.zeros((height, width, 3), dtype=np.uint8)

        if crop_line_data != None:

            # get rid of duplicate lines. May become redundant if a similarity threshold is done
            crop_line_data_1 = tuple_list_round(crop_line_data[0], -1, 4)
            crop_line_data_2 = []

            crop_lines_hough = np.zeros((height, width, 3), dtype=np.uint8)
            for (rho, theta) in crop_line_data_1:

                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a*rho
                y0 = b*rho
                point1 = (int(round(x0+1000*(-b))), int(round(y0+1000*(a))))
                point2 = (int(round(x0-1000*(-b))), int(round(y0-1000*(a))))
                cv2.line(crop_lines_hough, point1, point2, (0, 0, 255), 2)


            for curr_index in range(len(crop_line_data_1)):
                (rho, theta) = crop_line_data_1[curr_index]

                is_faulty = False
                if ((theta >= ANGLE_THRESH) and (theta <= math.pi-ANGLE_THRESH)) or (theta <= 0.0001):
                    is_faulty = True

                else:
                    for (other_rho, other_theta) in crop_line_data_1[curr_index+1:]:
                        if abs(theta - other_theta) < THETA_SIM_THRESH:
                            is_faulty = True
                        elif abs(rho - other_rho) < RHO_SIM_THRESH:
                            is_faulty = True

                if not is_faulty:
                    crop_line_data_2.append( (rho, theta) )



            for (rho, theta) in crop_line_data_2:

                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a*rho
                y0 = b*rho
                point1 = (int(round(x0+1000*(-b))), int(round(y0+1000*(a))))
                point2 = (int(round(x0-1000*(-b))), int(round(y0-1000*(a))))
                cv2.line(crop_lines, point1, point2, (0, 0, 255), 2)


            if len(crop_line_data_2) >= NUMBER_OF_ROWS:
                rows_found = True


        hough_thresh -= HOUGH_THRESH_INCR

    if rows_found == False:
        print(NUMBER_OF_ROWS, "rows_not_found")


    return (crop_lines, crop_lines_hough)


def tuple_list_round(tuple_list, ndigits_1=0, ndigits_2=0):
    '''Rounds each value in a list of tuples to the number of digits
       specified
    '''
    new_list = []
    for (value_1, value_2) in tuple_list:
        new_list.append( (round(value_1, ndigits_1), round(value_2, ndigits_2)) )

    return new_list

main()
