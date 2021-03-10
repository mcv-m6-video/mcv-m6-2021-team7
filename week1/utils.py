import numpy as np
import cv2

# 3.1 MSEN & PEPN
def read_flow(path_to_img):
    img = cv2.cvtColor(cv2.imread(path_to_img, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype(np.double)

    flow_u = (img[:, :, 0] - 2**15)/64
    flow_v = (img[:, :, 1] - 2**15)/64

    #_,flow_valid = cv2.threshold(img[:,:,2], 1, 1, cv2.THRESH_BINARY)
    flow_valid = img[:,:,2]
    flow_valid[flow_valid>1] = 1

    flow_u[flow_valid == 0] = 0
    flow_v[flow_valid == 0] = 0

    flow_img = np.dstack((flow_u, flow_v, flow_valid))
    #print(flow_img.shape)
    return flow_img
