Face crop and alignment

Usage:
python align_faces.py --image [[path/to/input_image]] 
	--output [[output_text(add to original name)]]
	--shape-predictor [[path/to/predictor]] (optional)


example:
python align_faces.py --image images/test.jpg --output results/result


dependencies:
openCV(python), dlib, boost.python(for dlib), imutils(available from pip)

