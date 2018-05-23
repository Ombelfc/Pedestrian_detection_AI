# Usage
# python3 gui.py

# import the necessary packages
from __future__ import print_function
import argparse
import imutils
import numpy as np
import tkinter.filedialog as fd
import cv2
from tkinter import *
from PIL import Image
from PIL import ImageTk
from imutils.object_detection import non_max_suppression
from imutils import paths
 
# original image panel 
panelA = None
# error-prone image panel
panelB = None
# final image panel
panelC = None

def select_image():

	# grab a reference to the image panels
	global panelA, panelB, panelC
 
	# open a file chooser dialog and allow the user 
	# to select an input image
	path = fd.askopenfilename()

	# check if the path is valid
	if len(path) > 0:

		# initialize the HOG descriptor/person detector
		hog = cv2.HOGDescriptor()
		hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

		# load the image and resize it to fit the panel
		image = cv2.imread(path)
		image = imutils.resize(image, width=min(400, image.shape[1]))
		# create copies of the original to process
		original = image.copy()
		initial = image.copy()
		
		# detect people in the image
		(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

		# Draw the initial bounding boxes
		for(x, y, w, h) in rects:
			cv2.rectangle(initial, (x, y), (x + w, y + h), (0, 0, 255), 2)

		# Apply non-maxima-suppression to tthe bounding boxes using a fairly large
		# overlap threshold to try to maintain overlaping boxes that are still people
		rects = np.array([[x, y, x + w, y + h] for(x, y, w, h) in rects])
		pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

		# Draw the final bounding boxes
		for(xA, yA, xB, yB) in pick:
			cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
		
		# Show information on the number of bounding boxes
		filename = path[path.rfind("/") + 1:]
		print("[INFO] {}: {} initial boxes, {} after suppression".format(filename, len(rects), len(pick)))

		# change the color model of the images from BGR 2 RGB to allow further processing
		original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
		initial = cv2.cvtColor(initial, cv2.COLOR_BGR2RGB)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# loads the image to memory
		original = Image.fromarray(original)
		initial = Image.fromarray(initial)
		image = Image.fromarray(image)
		
		original = ImageTk.PhotoImage(original)
		initial = ImageTk.PhotoImage(initial)
		image = ImageTk.PhotoImage(image)

	if panelA is None or panelB is None or panelC is None:
		
		# place the panels on the main gui frame
		panelA = Label(image=original)
		panelA.image = original
		panelA.pack(side="left", padx=10, pady=10)

		panelB = Label(image=initial)
		panelB.image = initial
		panelB.pack(side="left", padx=10, pady=10)

		panelC = Label(image=image)
		panelC.image = image
		panelC.pack(side="right", padx=10, pady=10)
	else:
		# place the final images on the panels
		panelA.configure(image=original)
		panelB.configure(image=initial)
		panelC.configure(image=image)
		
		panelA.image = original
		panelB.image = initial
		panelC.image = image

def main():

	# construct the argument parser and parse the arguments
	#ap = argparse.ArgumentParser()
	#ap.add_argument("-i", "--images", required=True, help="path to images directory")
	#args = vars(ap.parse_args())

	# initialize the HOG descriptor/person detector
	#hog = cv2.HOGDescriptor()
	#hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	# loop over the image paths
	#imagePaths = list(paths.list_images(args["images"]))

	# the main gui fram
	root = Tk()

	# button allowing selection of the image
	btn = Button(root, text="Select an image", command=select_image)
	# add the button to the main frame
	btn.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

	# run the main gui loop
	root.mainloop()

if __name__ == "__main__":
	main()