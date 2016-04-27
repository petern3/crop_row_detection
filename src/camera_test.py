#!/usr/bin/python

import cv2

def main():
	capture = cv2.VideoCapture(0)
	_, image = capture.read()
	previous = image.copy()
	
	
	while (cv2.waitKey(1) < 0):
		_, image = capture.read()
		diff = cv2.absdiff(image, previous)
		#image = cv2.flip(image, 3)
		#image = cv2.norm(image)
		_, diff = cv2.threshold(diff, 32, 0, cv2.THRESH_TOZERO)
		_, diff = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY)
		
		diff = cv2.medianBlur(diff, 5)
		
		cv2.imshow('video', diff)
		previous = image.copy()
		
	capture.release()
	cv2.destroyAllWindows()
	
main()
