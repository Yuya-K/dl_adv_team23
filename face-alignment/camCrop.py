#crop images from camera input
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from multiprocessing import Pool
import multiprocessing as multi
from datetime import datetime
import threading
import argparse
import imutils
import dlib
import cv2
import os

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



#image processing in another thread
class ProcessThread(threading.Thread):
    def __init__(self, frame, index):
        super(ProcessThread, self).__init__()
        self.image = frame
        self.index = index

    def run(self):
        self.image = imutils.resize(self.image, width=800)
        self.grayImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        rects = detector(self.grayImage, 2)

        if len(rects) == 0:
            print("can't detect any face")
        else:   
            for j in range(len(rects)):
                # extract the ROI of the *original* face, then align the face
                # using facial landmarks
                (x, y, w, h) = rect_to_bb(rects[j])
                faceOrig = imutils.resize(self.image[y:y + h, x:x + w], width=128)
                faceAligned = fa.align(self.image, self.grayImage, rects[j])
                margin = int(args["margin"])
                ratio = float(args["crop_ratio"])
                faceAligned = faceAligned[int(margin*ratio):-int(margin*ratio)+128,int(margin*ratio):-int(margin*ratio)+128]

                cv2.imwrite("./output"+str(self.index)+".jpg", faceAligned);
                print("saved image" + str(self.index))


# start video capture
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
ret, frame = cap.read()

while(cap.isOpened()):
    ret, frame = cap.read()

    cv2.imshow('camera capture', frame)

    #wating key input for 10msec
    k = cv2.waitKey(10)
    #exit with esc or q key
    if k == 27:
        break
    if k == ord('q'):
        break
    if k == 49 and len(frame) != 0:
        #49 means num 1
        cv2.imshow('image1', frame)
        th = ProcessThread(frame,1)
        th.start()
    if k == 50 and len(frame) != 0:
        cv2.imshow('image2', frame)
        th = ProcessThread(frame,2)
        th.start()
#exit
cap.release()
cv2.destroyAllWindows()