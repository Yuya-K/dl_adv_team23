# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division

import cv2
import os, sys
from argparse import ArgumentParser



#CASCADE_PATH = "./haarcascades/haarcascade_frontalface_default.xml"
CASCADE_PATH = "./haarcascades/haarcascade_frontalface_alt.xml"
#CASCADE_PATH = "./haarcascades/haarcascade_frontalface_alt2.xml"
#CASCADE_PATH = "./haarcascades/haarcascade_frontalface_alt_tree.xml"

SIZE = (200, 200)

parser = ArgumentParser()
parser.add_argument("-i", "--input_path", default="./input_images")
parser.add_argument("-o", "--output_path", default="./output_images")
parser.add_argument("-c", "--cascade_path", default=CASCADE_PATH)
args = parser.parse_args()


def crop_face(image_path, cascade_path=CASCADE_PATH):
    """
    crop face from the image in image_path
    """
    image = cv2.imread(image_path)
    
    # convert image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # obtain the features of cascade
    cascade = cv2.CascadeClassifier(cascade_path)

    # get bounding boxes
    facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(50, 50))

    crop_image_list = []
    for rect in facerect:
        x, y, w, h = rect
        crop_image_list.append(image[y:y+h, x:x+w])
    return crop_image_list



def main():
    input_path = args.input_path
    output_path = args.output_path
    image_name_list = os.listdir(input_path)
    for image_name in image_name_list:
        image_path = os.path.join(input_path, image_name)
        crop_image_list = crop_face(image_path)
        print(len(crop_image_list))
        for idx, crop_image in enumerate(crop_image_list):
            resized_image = cv2.resize(crop_image, SIZE)
            image_name_body, image_name_extension = os.path.splitext(image_name)
            output_image_path = os.path.join(output_path, image_name_body + "_" + str(idx) + image_name_extension)
            print(output_image_path)
            cv2.imwrite(output_image_path, crop_image)




if __name__ == "__main__":
    main()
