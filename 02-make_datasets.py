import numpy as np
import cv2
import os
import pickle
import random

DIRECTORY = "capture"
OUTPUT = "datasets"
CLUSTERS = 12

print("list file in folder...")
images = os.listdir(DIRECTORY)




print("creating empty array to store images")
rgb_ds = []
kmeans_ds = []
final_ds = []



def K_Means(Images, K) :
	if(len(Images.shape) < 3) :
		Z = Images.reshape((-1, 1))
	elif(len(Images.shape) == 3) :
		Z = Images.reshape((-1, 3))

	# convert array to np.float32
	Z = np.float32(Z)

	# define criteria, number of clusters(k) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

	# now convert back into uint8, and make original image
	center = np.uint8(center)
	res = center[label.flatten()]
	Clustered_Image = res.reshape((Images.shape))
	return Clustered_Image




def make_rgb() :
	print("loading images for color datasets...")
	for img in images :
		try :
			rgb = cv2.imread(os.path.join(DIRECTORY, img))
			rgb_ds.append(rgb)

			# kmeans = K_Means(rgb, CLUSTERS)
			# kmeans_ds.append(kmeans)
			# print(images.index(img), end=" ")

		except Exception as e :
			print(e)
			break
	# random.shuffle(means_ds)
	print("exporting color database...")
	pickle.dump(rgb_ds, open(os.path.join(OUTPUT, "color.pickle"), "wb"))
	print("database color exported...")





def make_means() :
	print("loading images for color datasets...")
	for img in images :
		try :
			rgb = cv2.imread(os.path.join(DIRECTORY, img))
			rgb_ds.append(rgb)

			kmeans = K_Means(rgb, CLUSTERS)
			kmeans_ds.append(kmeans)

			b1, g1, r1 = cv2.split(rgb)
			b2, g2, r2 = cv2.split(kmeans)

			final = np.stack([b1, g1, r1, b2, g2, r1], axis=2)
			final_ds.append(final)

		except Exception as e :
			print(e)
			break

	# random.shuffle(kmeans_ds)
	print("exporting database...")
	pickle.dump(rgb_ds, open(os.path.join(OUTPUT, "color.pickle"), "wb"))
	pickle.dump(kmeans_ds, open(os.path.join(OUTPUT, "kmeans.pickle"), "wb"))
	pickle.dump(final_ds, open(os.path.join(OUTPUT, "combined.pickle"), "wb"))
	print("export completed...")


make_means()