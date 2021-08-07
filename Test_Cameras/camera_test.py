import cv2
import numpy as np
import os

cap = cv2.VideoCapture(3) # change number to get the right web cam
for i in range(10):
    ret, frame = cap.read()
print(ret) # whether or not frame was captured
cv2.imwrite("camera_test.jpg", frame)

print("Finished.")

'''
run this script to determine camera numbers for top and side camera
'''