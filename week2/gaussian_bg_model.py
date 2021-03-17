import cv2
import xmltodict
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from bboxdetection import *
from dataReader import ReadData
import pickle


class GaussianModel:
    def __init__(self,path,color_space):
        self.path = path
        self.vCapture = cv2.VideoCapture(path)
        self.width = int(self.vCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.length = int(self.vCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.color_space = color_space
        self.vidLen = int(self.vCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.previous_mask = None
        self.mean = []
        self.std = []

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

    def mean_std_Welford(self):
        mean_is_loaded = True
        std_is_loaded = True
        try:
            self.mean = pickle.load(open('mean.pkl', "rb"))
            # print("MEAN: ", str(self.mean))
            print("MEAN LOADED")

        except Exception:
            mean_is_loaded = False
            print("MEAN NOT LOADED")
        try:
            self.std = pickle.load(open('std.pkl', "rb"))
            print("STD LOADED")
        except Exception:
            std_is_loaded = False
            print("STD NOT LOADED")
        # Implementation of a running average (Welford’s algorithm)
        if not mean_is_loaded or not std_is_loaded:
            _, num_channels = self._color_space_prep(addDim=False, i=None)
            mean = np.zeros((self.height, self.width, num_channels))
            M2 = np.zeros((self.height, self.width, num_channels))
            count = 0
            len_25 = int(self.vidLen * 0.25)
            for i in trange(len_25, desc='GaussianModelling background'):
                imC, _ = self._color_space_prep(addDim=True, i=i)
                count += 1
                delta = imC - mean
                mean += delta / count
                delta2 = imC - mean
                M2 += delta * delta2
            self.mean = mean
            self.std = np.sqrt(M2 / count)

            with open('mean.pkl', 'wb') as handle:
                pickle.dump(self.mean, handle)
            with open('std.pkl', 'wb') as handle:
                pickle.dump(self.std, handle)

    def foreground_extraction_task2(self, showVid, gt, adaptive=False, alpha=2, rho=0.001):
        initFrame = int(self.vidLen * 0.25)
        endFrame = int(self.vidLen)

        predictedBBOX = []
        predictedFrames = []
        count = 0
        for i in trange(initFrame, endFrame, desc='Foreground extraction'):
            imC, _ = self._color_space_prep(addDim=True, i=i)

            if adaptive:
                if self.previous_mask is not None:
                    background = (1 - self.previous_mask)
                    # print("adapting mean and std...")
                    # print("previous mean: ", str(self.mean))
                    self.mean = rho * imC * (background) + (1 - rho) * self.mean
                    self.std = np.sqrt(rho * (imC * (background) - self.mean) ** 2 + (1 - rho) * self.std ** 2)

            mask = abs(imC - self.mean) >= (alpha * (self.std + 2))
            mask = mask * 1.0
            self.previous_mask = mask

            # Detect white patches (cars)
            denoised_m = removeNoise(mask,'task2')
            bboxFrame = findBBOX(denoised_m,'task2')
            predictedBBOX.append(bboxFrame)
            predictedFrames.append(i)

            if not showVid:
                gtBoxes = gt[count]['bbox']
                mRGB = np.zeros((denoised_m.shape[0], denoised_m.shape[1], 3))
                mRGB[:, :, 0] = denoised_m
                mRGB[:, :, 1] = denoised_m
                mRGB[:, :, 2] = denoised_m

                for k in range(len(gtBoxes)):
                    gbox = gtBoxes[k]
                    if gbox != None:
                        cv2.rectangle(mRGB, (int(gbox[0]), int(gbox[1])), (int(gbox[2]), int(gbox[3])), (0, 0, 255), 2)
                for b in bboxFrame:
                    cv2.rectangle(mRGB, (b[0], b[1]), (b[2], b[3]), (100, 255, 0), 2)
                cv2.imshow('mask', mRGB)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            count += 1
        predictionsInfo, num_bboxes = prepareBBOXdata(predictedBBOX, predictedFrames)

        return predictionsInfo, num_bboxes

    # Foreground extraction task 1
    def foreground_extraction(self,showVid,gt,alpha,noiseRemoval):
        initFrame = int(self.vidLen*0.25)
        endFrame = int(self.vidLen)
        predictedBBOX = []
        predictedFrames = []
        count = 0
        for i in trange(initFrame,endFrame, desc='Foreground extraction'):
            imC,_ = self._color_space_prep(addDim=True,i=i)
            mask = abs(imC-self.mean)>=(alpha*(self.std+2))
            mask = mask*1.0
            m = mask.reshape(mask.shape[0], mask.shape[1])

            # Detect white patches (cars)
            if noiseRemoval:
                denoised_m = removeNoise(m,'task1')
            else:
                denoised_m = m
            bboxFrame = findBBOX(denoised_m,'task1')
            predictedBBOX.append(bboxFrame)
            predictedFrames.append(i)

            if showVid:
                gtBoxes = gt[count]['bbox']
                mRGB = np.zeros((denoised_m.shape[0],denoised_m.shape[1],3))
                mRGB[:, :, 0] = denoised_m
                mRGB[:, :, 1] = denoised_m
                mRGB[:, :, 2] = denoised_m

                for k in range(len(gtBoxes)):
                    gbox=gtBoxes[k]
                    if gbox != None:
                        cv2.rectangle(mRGB, (int(gbox[0]), int(gbox[1])), (int(gbox[2]), int(gbox[3])), (0,0,255), 2)
                for b in bboxFrame:
                    cv2.rectangle(mRGB, (b[0], b[1]), (b[2], b[3]), (100, 255, 0), 2)
                cv2.imshow('mask', mRGB)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            count += 1
        predictionsInfo,num_bboxes = prepareBBOXdata(predictedBBOX, predictedFrames)

        return predictionsInfo,num_bboxes





