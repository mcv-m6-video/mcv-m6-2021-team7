import matplotlib.pyplot as plt
import pickle
import numpy as np
import cv2


class PlotCreator:
    def __init__(self):
        super(PlotCreator, self).__init__()
    def plotCurve(self,mapx,mapy,iouy,labelx,name):

        plt.plot(mapx, mapy, label='mAP')
        plt.plot(mapx, iouy, label="mIoU")
        plt.xlabel(labelx)
        plt.legend(loc="upper left")
        plt.show()
        plt.savefig(name+'.png')

    def plotMIOU(self):

        with open('miouretina.pkl', 'rb') as f:
            ret = pickle.load(f)
        with open('mioufasterRCNN.pkl', 'rb') as f:
            fast = pickle.load(f)
        with open('mioumaskRCNN.pkl', 'rb') as f:
            mask = pickle.load(f)
        print(ret== fast)
        plt.plot(ret, label='RetinaNet')
        plt.plot(fast, label="FasterRCNN")
        plt.plot(mask, label="MaskRCNN")
        plt.xlabel('Frames')
        plt.ylabel('mIoU')
        plt.legend(loc="upper left")
        plt.show()

    def plotVid(self,init_frame, end_frame, cap, gtInfo, predictionsInfo):

        count = 0
        for i in range(init_frame, end_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            im = cap.read()[1]

            gtBoxes = gtInfo[count]['bbox']

            for k in range(len(gtBoxes)):
                gbox = gtBoxes[k]
                if gbox.all() != None:
                    cv2.rectangle(im, (int(gbox[0]), int(gbox[1])), (int(gbox[2]), int(gbox[3])), (0, 0, 255), 2)
            for b in predictionsInfo[count]['bbox']:
                if b is not None:
                    b = b.cpu().numpy()
                    cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), (100, 255, 0), 2)
            cv2.imshow('Video', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            count += 1

