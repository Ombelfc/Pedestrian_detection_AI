# Usage
# python3 gui.py

# import the necessary packages
from __future__ import print_function
from tkinter import *
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog as fd
import cv2
from imutils.object_detection import non_max_suppression
from imutils import paths
import argparse
import imutils
import numpy as np
 
panelA = None
panelB = None
panelC = None

def select_image():
	# grab a reference to the image panels
	global panelA, panelB, panelC
 
	# open a file chooser dialog and allow the user to select an input
	# image
	path = fd.askopenfilename()

	if len(path) > 0:

		hog = cv2.HOGDescriptor()
		hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

		image = cv2.imread(path)
		image = imutils.resize(image, width=min(400, image.shape[1]))
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

		original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
		initial = cv2.cvtColor(initial, cv2.COLOR_BGR2RGB)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		original = Image.fromarray(original)
		initial = Image.fromarray(initial)
		image = Image.fromarray(image)
		
		original = ImageTk.PhotoImage(original)
		initial = ImageTk.PhotoImage(initial)
		image = ImageTk.PhotoImage(image)

	if panelA is None or panelB is None or panelC is None:

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
		panelA.configure(image=original)
		panelB.configure(image=initial)
		panelC.configure(image=image)
		panelA.image = original
		panelB.image = initial
		panelC.image = image

def main():

	#ap = argparse.ArgumentParser()
	#ap.add_argument("-i", "--images", required=True, help="path to images directory")
	#args = vars(ap.parse_args())

	#hog = cv2.HOGDescriptor()
	#hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	#imagePaths = list(paths.list_images(args["images"]))

	root = Tk()

	btn = Button(root, text="Select an image", command=select_image)
	btn.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

	root.mainloop()

if __name__ == "__main__":
	main()