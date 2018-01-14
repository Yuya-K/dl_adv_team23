# USAGE
# python align_faces.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import uuid
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False,
	help="path to facial landmark predictor", default="shape_predictor_68_face_landmarks.dat")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-o", "--output", required=False,
	help="additional filename for output", default="result")
ap.add_argument("-m", "--margin", required=False,
	help="mergin for crop(lager number means smaller face region, adjust crop-ratio basically)", default=60)
ap.add_argument("-r", "--crop-ratio", required=False,
	help="crop ratio([0,1]): larger number means smaller face region", default=0.6)
args = vars(ap.parse_args())

filename = os.path.basename(args["image"])

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
face_width = 256+int(args["margin"])
fa = FaceAligner(predictor, desiredFaceWidth=face_width)

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# show the original input image and detect faces in the grayscale
# image

# cv2.imshow("Input", image)
rects = detector(gray, 2)

# loop over the face detections
for rect in rects:
	# extract the ROI of the *original* face, then align the face
	# using facial landmarks
	(x, y, w, h) = rect_to_bb(rect)
	# faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
	faceAligned = fa.align(image, gray, rect)
	margin = int(args["margin"])
	ratio = float(args["crop_ratio"])
	# faceAligned = faceAligned[int(margin/2):int(margin/2)+256,margin:margin+256]
	# faceAligned = faceAligned[margin:margin+256,int(margin/2):int(margin/2)+256]
	faceAligned = faceAligned[int(margin*ratio):int(margin*ratio)+256,int(margin*ratio):int(margin*ratio)+256]

	f = str(uuid.uuid4())
	cv2.imwrite(args["output"] + "_"+ filename, faceAligned)

	# display the output images

	# cv2.imshow("Original", faceOrig)
	# cv2.imshow("Aligned", faceAligned)
	cv2.waitKey(0)