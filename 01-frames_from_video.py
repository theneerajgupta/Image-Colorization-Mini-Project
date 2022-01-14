import cv2
import os

VIDEO = "videos/video3.mp4"
CAPTURE = "capture/"

def frame_cap(input, output) :
	try :
		video_obj = cv2.VideoCapture(input)
		count = 0
		success = 1
		while success :
			success, image = video_obj.read()
			if count%6 == 0 :
				image = cv2.resize(image, (1024, 512))
				cv2.imwrite(os.path.join(output, "frame%d.jpg"%count), image)
			count = count + 1
	except Exception as e :
		pass

if __name__ == "__main__" :
	frame_cap(VIDEO, CAPTURE)