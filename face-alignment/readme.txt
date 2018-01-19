Face crop and alignment

Usage:
python face_normalize.py
	-i --image [[path/to/input_dir]] 

	-o --output [[path/to/output_dir]]

	-p --shape-predictor [[path/to/predictor]] (optional)





example:
python face_normalize.py --image images/test.jpg --output results/result




dependencies:
openCV(python), dlib, boost.python(for dlib), imutils(available from pip)

