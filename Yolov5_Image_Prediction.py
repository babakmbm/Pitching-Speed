import cv2
import numpy as np
import os
import pytesseract as pt

# Model Configuration
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

img = cv2.imread('static/dataset/Prediction/212561.jpg')

# cv2.namedWindow('TEST IMAGE', cv2.WINDOW_KEEPRATIO)
# print(cv2.getBuildInformation())
# cv2.imshow('TEST IMAGE', img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# LOAD YOLO MODEL
net = cv2.dnn.readNetFromONNX('yolov5/runs/train/Model3/weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# CONVERT IMAGE TO YOLO FORMAT
image = img.copy()
row, col, d = image.shape
# print(row, col, d)

max_rc = max(row, col)
# print(max_rc)
input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
# print(input_image)
input_image[0:row, 0:col] = image

# GET PREDICTIONS FROM YOLO MODEL
blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
net.setInput(blob)
preds = net.forward()
print(preds.shape)
print(preds[0][0])

boxes = []
confidences = []
image_w, image_h = input_image.shape[:2]
x_factor = image_w/INPUT_WIDTH
y_factor = image_h/INPUT_HEIGHT

for i in range(len(preds[0])):
    row = preds[0][i]
    confidence = row[-2]
    #print(confidence)
    if confidence > 0.04:
        class_score = row[-1]  # probability score of detecting
        # print(confidence)
        # print(class_score)
        # print("\n")
        if class_score > 0.025:
            cx, cy, w, h = row[0:4]
            print(cx, cy, w, h)
            left = int((cx-0.5*w)*x_factor)
