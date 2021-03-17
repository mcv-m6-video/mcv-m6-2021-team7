import cv2
from tqdm import trange
from bboxdetection import *
import numpy
import imageio
import os

def get_opencv_bgsubs(alg):
    # Inicialize bg segmentation class (from opencv)
    if alg == 'KNN':
        bgsubs = cv2.createBackgroundSubtractorKNN()
    elif alg == 'MOG2':
        bgsubs = cv2.createBackgroundSubtractorMOG2()
    elif alg == 'CNT':
        bgsubs = cv2.bgsegm.createBackgroundSubtractorCNT()
    elif alg == 'GMG':
        bgsubs = cv2.bgsegm.createBackgroundSubtractorGMG()
    elif alg == 'GSOC':
        bgsubs = cv2.bgsegm.createBackgroundSubtractorGSOC()
    elif alg == 'LSBP':
        bgsubs = cv2.bgsegm.createBackgroundSubtractorLSBP()
    elif alg == 'MOG':
        bgsubs = cv2.bgsegm.createBackgroundSubtractorMOG()

    return bgsubs

class BgsModel:
    def __init__(self,path,color_space, alg):
        self.path = path
        self.vCapture = cv2.VideoCapture(path)
        self.width = int(self.vCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.length = int(self.vCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.color_space = color_space
        self.vidLen = int(self.vCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.backSub = get_opencv_bgsubs(alg)

    def retVidLen(self):
        return self.vidLen

    def _color_space_prep(self,addDim,i):
        self.vCapture.set(cv2.CAP_PROP_POS_FRAMES, i)
        im = self.vCapture.read()[1]
        if self.color_space == 'gray':
            imC = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            num_channels = 1
        elif self.color_space == 'rgb':
            imC = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            num_channels = 3
        elif self.color_space == 'hsv':
            imC = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            num_channels = 3
        elif self.color_space == 'lab':
            imC = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
            num_channels = 3
        elif self.color_space == 'yuv':
            imC = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
            num_channels = 3
        else:
            print('Color space is not in the dictionary')
            return
        if addDim:
            imC = imC.reshape(im.shape[0],im.shape[1],num_channels)
        return imC,num_channels


    def foreground_extraction(self,showVid, gt, use_postprocessing, outpath=None):
        initFrame = int(self.vidLen*0.25)
        endFrame = int(self.vidLen)

        if outpath:
            if not os.path.exists(outpath):
                os.makedirs(outpath)

        for i in trange(initFrame, desc='Background computation'):
            imC, _ = self._color_space_prep(addDim=True, i=i)
            self.backSub.apply(imC)

        predictedBBOX = []
        predictedFrames = []
        count = 0
        for i in trange(initFrame,endFrame, desc='Foreground extraction'):
            imC,_ = self._color_space_prep(addDim=True,i=i)
            mask = self.backSub.apply(imC)
            mask = np.ceil(mask/255.0)

            # Detect white patches (cars)
            if use_postprocessing:
                denoised_m = removeNoise(mask,'task3')
            else:
                denoised_m = mask
            bboxFrame = findBBOX(denoised_m,'task3')
            predictedBBOX.append(bboxFrame)
            predictedFrames.append(i)

            if showVid:
                denoised_m = denoised_m*255
                gtBoxes = gt[count]['bbox']
                mRGB = np.zeros((denoised_m.shape[0],denoised_m.shape[1],3))
                mRGB[:, :, 0] = denoised_m
                mRGB[:, :, 1] = denoised_m
                mRGB[:, :, 2] = denoised_m

                for k in range(len(gtBoxes)):
                    gbox=gtBoxes[k]
                    if gbox != None:
                        cv2.rectangle(mRGB, (int(gbox[0]), int(gbox[1])), (int(gbox[2]), int(gbox[3])), (0,0,255), 3)
                for b in bboxFrame:
                    cv2.rectangle(mRGB, (b[0], b[1]), (b[2], b[3]), (100, 255, 0), 3)
                mRGB = mRGB.astype('uint8')


                if outpath:
                    cv2.imwrite(os.path.join(outpath, str(i) + '.jpg'), mRGB)


            count += 1
        predictionsInfo,num_bboxes = prepareBBOXdata(predictedBBOX, predictedFrames)

        return predictionsInfo,num_bboxes


if __name__ == '__main__':
    backSub = get_opencv_bgsubs('KNN')

    capture = cv2.VideoCapture("D:\MCV\M6\AICity_data\\train\S03\c010\\vdo.avi")

    if not capture.isOpened():
        print('Unable to open: D:\MCV\M6\AICity_data\\train\S03\c010\\vdo.avi')
        exit(0)
    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        fgMask = backSub.apply(frame)

        cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv2.imshow('Frame', frame)
        cv2.imshow('FG Mask', fgMask)

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break