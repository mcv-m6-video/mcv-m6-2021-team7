import numpy as np
import cv2
from dataReader import ReadData

def findBBOX(mask,task):

    if task == 'task1' or task == 'task2' or task == 'task3' or task == 'task4':
        minH = 50
        maxH =  1080/2 # 1080--> height frame
        minW = 100#120
        maxW =1920/2# 1920--> width frame

        contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        box = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if minW < w < maxW and minH < h < maxH:
                if 0.2 < w/h < 10:
                    box.append([x, y, x + w, y + h])

    return box

def prepareBBOXdata(predicted,frame):
    Info = []
    num_boxes = 0
    for i in range(len(predicted)):
        boxes = predicted[i]
        f = frame[i]
        Info.append({"frame": f, "bbox": np.array(boxes)})
        num_boxes = num_boxes + len(boxes)

    return Info,num_boxes



def removeNoise(mask,task):

    if task == 'task2':
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    elif task == 'task3':
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    elif task == 'task1' or task == 'task4':
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)#remove small white dots
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)#define the cars

    return closing



