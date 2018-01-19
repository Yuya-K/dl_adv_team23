# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division

import cv2
import os, sys
import better_exceptions
from math import floor, ceil
from argparse import ArgumentParser



#CASCADE_PATH = "./haarcascades/haarcascade_frontalface_default.xml"
CASCADE_PATH = "./haarcascades/haarcascade_frontalface_alt.xml"
#CASCADE_PATH = "./haarcascades/haarcascade_frontalface_alt2.xml"
#CASCADE_PATH = "./haarcascades/haarcascade_frontalface_alt_tree.xml"

SIZE = (128, 128)

parser = ArgumentParser()
parser.add_argument("-i", "--input_path", default="./input_images")
parser.add_argument("-o", "--output_path", default="./output_images")
parser.add_argument("-c", "--cascade_path", default=CASCADE_PATH)
parser.add_argument("-r", "--crop_ratio", type=float, default=0.85)
parser.add_argument("-m", "--max_num_image", type=int, default=0)
args = parser.parse_args()


def crop_face(image_path, cascade_path=CASCADE_PATH):
    """
    crop face from the image in image_path
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # convert image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # obtain the features of cascade
    cascade = cv2.CascadeClassifier(cascade_path)

    # get bounding boxes
    facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(50, 50))

    croped_image_list = []
    for rect in facerect:
        x, y, w, h = rect
        # move y down
        y += int(0.1 * h)
        croped_image_list.append(image[y:y+h, x:x+w])
    return croped_image_list


def crop_more(cropped_image, crop_ratio):
    """
    crop the cropped image cropped_image more
    """
    size = cropped_image.shape[0]
    start_ratio = (1. - crop_ratio) / 2.
    end_ratio = crop_ratio + (1. - crop_ratio) / 2.
    assert abs((end_ratio - start_ratio) - crop_ratio) < 1e-3
    start_hori_index, end_hori_index = int(ceil(size * start_ratio)), int(floor(size * end_ratio))
    start_vert_index, end_vert_index = int(ceil(size * (start_ratio + (1 - crop_ratio)/3))), int(floor(size * (end_ratio + (1 - crop_ratio)/3)))
    
    more_cropped_image = cropped_image[start_vert_index:end_vert_index, start_hori_index:end_hori_index]
    return more_cropped_image


def main():
    input_path = args.input_path
    output_path = args.output_path
    if os.path.exists(output_path):
        os.mkdir(output_path)
    image_name_list = os.listdir(input_path)
    if args.max_num_image != 0:
        image_name_list = image_name_list[:args.max_num_image]
    for image_name in image_name_list:
        image_path = os.path.join(input_path, image_name)
        croped_image_list = crop_face(image_path)
        for idx, croped_image in enumerate(croped_image_list):
            more_cropped_image = crop_more(croped_image, args.crop_ratio)
            resized_image = cv2.resize(more_cropped_image, SIZE)
            image_name_body, image_name_extension = os.path.splitext(image_name)
            output_image_path = os.path.join(output_path, image_name_body + "_" + str(idx) + image_name_extension)
            print("Save", output_image_path)
            cv2.imwrite(output_image_path, resized_image)


if __name__ == "__main__":
    main()
