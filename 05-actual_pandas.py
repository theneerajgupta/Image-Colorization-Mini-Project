import numpy as np
import cv2
import os
import pickle

files = os.listdir("panda")
ds = []

for panda in files :
	img = cv2.imread(os.path.join("panda", panda))
	img = cv2.resize(img, (1024, 512))
	print(img.shape)
	ds.append(img)


pickle.dump(ds, open(os.path.join("datasets", "panda.pickle"), "wb"))