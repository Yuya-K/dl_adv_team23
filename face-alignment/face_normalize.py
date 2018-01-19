# USAGE
# python align_faces.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from multiprocessing import Pool
import multiprocessing as multi
import os
import argparse
import imutils
import dlib
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False,
	help="path to facial landmark predictor", default="./shape_predictor_68_face_landmarks.dat")
ap.add_argument("-i", "--image", required=False,
	help="path to input image dir ./path/to/input_dir/", default="./input/")
ap.add_argument("-o", "--output", required=False,
	help="output directory", default="./results/")
ap.add_argument("-m", "--margin", required=False,
	help="mergin for crop(lager number means smaller face region, adjust crop-ratio basically)", default=10)
ap.add_argument("-r", "--crop-ratio", required=False,
	help="crop ratio([0,1]): larger number means smaller face region", default=0.8)
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=128)

#load images in input dir
images = []
gray_images = []
files = os.listdir(args["image"])
file_list = [f for f in files if os.path.isfile(os.path.join(args["image"], f))]
assert len(file_list) != 0
for i in range(len(file_list)):
    name, ext = os.path.splitext(file_list[i]) #get extention
    if ext == u'.png' or u'.jpeg' or u'.jpg':
        #resize and convert to grayscale
        images.append(cv2.imread(args["image"] + file_list[i])) #imread needs path to input file, not only filename
        images[-1] = imutils.resize(images[-1], width=800)
        gray_images.append(cv2.cvtColor(images[-1], cv2.COLOR_BGR2GRAY)) #imread reads images as BGR numpy array

for i in range(len(images)):
    print("processing image " + str(i))
    #detect the number of faces
    rects = detector(gray_images[i], 2)
    for j in range(len(rects)):
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(rects[j])
        faceOrig = imutils.resize(images[i][y:y + h, x:x + w], width=128)
        faceAligned = fa.align(images[i], gray_images[i], rects[j])
        margin = int(args["margin"])
        ratio = float(args["crop_ratio"])
        faceAligned = faceAligned[int(margin*ratio):-int(margin*ratio)+128,int(margin*ratio):-int(margin*ratio)+128]
        
        name, ext = os.path.splitext(file_list[i])
        cv2.imwrite(args["output"] + name + "_face" + str(j) + "_cropped" + ext, faceAligned)