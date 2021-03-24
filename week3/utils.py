import numpy as np

def euclid_dist(point1, point2):
    d = np.linalg.norm(np.array(point1) - np.array(point2))
    return d

def centroid(bboxGT):  # x, y, w, h
    x1 = bboxGT[0]
    y1 = bboxGT[1]
    x2 = bboxGT[2] + x1
    y2 = bboxGT[3] + y1
    xCenter = (x1 + x2) / 2
    yCenter = (y1 + y2) / 2
    return xCenter, yCenter

def iou(boxA, boxB):
    # For each prediction, compute its iou over all the boxes in that frame
    x11, y11, x12, y12 = np.split(boxA, 4, axis=0)
    x21, y21, x22, y22 = np.split(boxB, 4, axis=0)

    # Calculate the intersection in the bboxes
    xmin = np.maximum(x11, np.transpose(x21))
    ymin = np.maximum(y11, np.transpose(y21))
    xmax = np.minimum(x12, np.transpose(x22))
    ymax = np.minimum(y12, np.transpose(y22))
    w = np.maximum(xmax - xmin + 1.0, 0.0)
    h = np.maximum(ymax - ymin + 1.0, 0.0)
    intersection = w * h

    # Union
    areaboxA = (x12 - x11 + 1.0) * (y12 - y11 + 1.0)
    areaboxB = (x22 - x21 + 1.0) * (y22 - y21 + 1.0)
    union = areaboxA + np.transpose(areaboxB) - intersection

    iou = intersection / union

    return iou
