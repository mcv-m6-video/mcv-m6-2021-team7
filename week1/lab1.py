from evaluationOF import  *
from dataReader import ReadData
import numpy as np
from createPlots import PlotCreator
from plotsOF import PlotOF
from numpy import random
import cv2
import matplotlib.pyplot as plt
from evaluation import *
from utils import *
from videos import *
from mpl_toolkits.mplot3d import axes3d


def task11():
    #Read gt file
    path = 'ai_challenge_s03_c010-full_annotation.xml'
    reader = ReadData(path)
    gt, num_iter = reader.getGTfromXML()
    sortedFrames, sortedBBOX = reader.bboxInFrame(gt)
    gtInfo = reader.joinBBOXfromFrame(sortedFrames, sortedBBOX, isGT=True)

    summary = []
    addNoise = True
    addDrop = False
    if addNoise:
        stdPixels = np.linspace(0, 100, 11)
        for n in range(len(stdPixels)):
            std = int(stdPixels[n])
            predictedBBOX = []
            for i in range(len(sortedBBOX)):
                noisyBBOX = sortedBBOX[i] + np.random.normal(0, std, 4)  # mean 0 , std = stdPixels
                predictedBBOX.append(noisyBBOX)
            predictionsInfo = reader.joinBBOXfromFrame(sortedFrames, predictedBBOX, isGT=False)
            rec, prec, ap, meanIoU = ap_score(gtInfo, predictionsInfo, num_bboxes=len(sortedFrames), ovthresh=0.5)

            print('Noise std:', std)
            print('mAP:', np.mean(ap))
            print('Mean IOU:', meanIoU)
            gtInfo = reader.resetGT(gtInfo)
            summary.append({"std": std, "prec": np.mean(prec), "rec": np.mean(rec), "iou":meanIoU,"mAP":np.mean(ap)})

    if addDrop:
        dropThr = np.linspace(0, 0.9, 11)
        for n in range(len(dropThr)):
            predictedBBOX = []
            for i in range(len(sortedBBOX)):
                drop_box = random.rand()<dropThr[n]
                if not drop_box:
                    predictedBBOX.append(sortedBBOX[i])
                else:
                    predictedBBOX.append(None)
            predictionsInfo = reader.joinBBOXfromFrame(sortedFrames, predictedBBOX, isGT=False)

            rec, prec, ap, meanIoU = ap_score(gtInfo, predictionsInfo, num_bboxes=len(sortedFrames), ovthresh=0.5)
            print('Thr:', dropThr[n])
            print('mAP:', np.mean(ap))
            print('Mean IOU:', meanIoU)
            gtInfo = reader.resetGT(gtInfo)
            summary.append(
                {"dropThr": dropThr[n], "prec": np.mean(prec), "rec": np.mean(rec), "iou": meanIoU, "mAP": np.mean(ap)})


    plots = True
    if plots and addNoise:
        graph = PlotCreator()
        graph.plotCurve(datax=[dict['std'] for dict in summary],datay=[dict['iou'] for dict in summary],labelx='Noise std (pixels)',labely='IoU')
        graph.plotCurve(datax=[dict['std'] for dict in summary],datay=[dict['mAP'] for dict in summary],labelx='Noise std (pixels)',labely='mAP')

    elif plots and addDrop:
        graph = PlotCreator()
        graph.plotCurve(datax=[dict['dropThr'] for dict in summary],datay=[dict['iou'] for dict in summary],labelx='Drop Thr',labely='IoU')
        graph.plotCurve(datax=[dict['dropThr'] for dict in summary],datay=[dict['mAP'] for dict in summary],labelx='Drop Thr',labely='mAP')


# 1.2. Compute the mAP on the detections from mask_rcnn, ssd512 i yolo3
def task12():
    path = 'ai_challenge_s03_c010-full_annotation.xml'

    pred_paths = ['AICity_data/train/S03/c010/det/det_mask_rcnn.txt',
                  'AICity_data/train/S03/c010/det/det_ssd512.txt',
                  'AICity_data/train/S03/c010/det/det_yolo3.txt']

    pred_nets = ['mask_rcnn', 'det_ssd512', 'det_yolo3']

    for pc in range(len(pred_nets)):
        reader = ReadData(path)
        gt, num_iter = reader.getGTfromXML()
        sortedFrames_gt, sortedBBOX_gt = reader.bboxInFrame(gt)
        gtInfo = reader.joinBBOXfromFrame(sortedFrames_gt, sortedBBOX_gt, isGT=True)

        reader = ReadData(pred_paths[pc])
        pred, _ = reader.getPredfromTXT()
        sortedFrames_pred, sortedBBOX_pred, sortedScore_pred = reader.bboxInFrame_Score(pred)
        predictionsInfo = reader.joinBBOXfromFrame_Score(sortedFrames_pred, sortedBBOX_pred, sortedScore_pred,isGT=False)

        rec, prec, ap, meanIoU = VOC_ap_score(gtInfo,predictionsInfo,num_bboxes=len(sortedFrames_gt),ovthresh=0.5)
        print('Inference with',pred_nets[pc])
        print('mAP:',np.mean(ap))
        print('meanIoU:', np.mean(meanIoU))


# 2. IoU vs time
def task2():
    path = 'ai_challenge_s03_c010-full_annotation.xml'
    video_path = 'AICity_data/train/S03/c010/vdo.avi'
    # Networks mask_rcnn, det_ssd512 and det_yolo3 (task 1.2)
    pred_paths = ['AICity_data/train/S03/c010/det/det_mask_rcnn.txt',
                  'AICity_data/train/S03/c010/det/det_ssd512.txt',
                  'AICity_data/train/S03/c010/det/det_yolo3.txt']

    # Import GT
    reader = ReadData(path)
    gt, num_iter = reader.getGTfromXML()
    sortedFrames_gt, sortedBBOX_gt = reader.bboxInFrame(gt)
    gtInfo = reader.joinBBOXfromFrame(sortedFrames_gt, sortedBBOX_gt, isGT=True)

    # Noisy GT (task 1.1)
    stdPixels =  10.0
    predictedBBOX = []
    for i in range(len(sortedFrames_gt)):
        noisyBBOX = sortedBBOX_gt[i] + np.random.normal(0, stdPixels, 4)#mean 0 , std = stdPixels
        predictedBBOX.append(noisyBBOX)

    predictionsInfo = reader.joinBBOXfromFrame(sortedFrames_gt,predictedBBOX,isGT=False)
    rec, prec, ap, meanIoU = ap_score(gtInfo,predictionsInfo,num_bboxes=len(sortedFrames_gt),ovthresh=0.5)

    video_with_bbox(video_path,'noisyGT', gtInfo, predictionsInfo)
    meanIoUvideoplot('noisyGT', meanIoU)

    pred_nets = ['mask_rcnn', 'det_ssd512', 'det_yolo3']

    for pc in range(len(pred_nets)):
        reader = ReadData(pred_paths[pc])
        pred, _ = reader.getPredfromTXT()
        sortedFrames_pred, sortedBBOX_pred, sortedScore_pred = reader.bboxInFrame_Score(pred)
        predictionsInfo = reader.joinBBOXfromFrame_Score(sortedFrames_pred, sortedBBOX_pred, sortedScore_pred,isGT=False)

        rec, prec, ap, meanIoU = VOC_ap_score(gtInfo,predictionsInfo,num_bboxes=len(sortedFrames_gt),ovthresh=0.5)

        video_with_bbox(video_path, pred_nets[pc], gtInfo, predictionsInfo)
        meanIoUvideoplot(pred_nets[pc], meanIoU)

def task3():
    sequences = ['045', '157']
    plotOF = PlotOF()
    for idx, sequence in enumerate(sequences):
        pred_flow = read_flow('../results_opticalflow_kitti/results/LKflow_000' + sequence + '_10.png')
        gt_flow = read_flow('../data_stereo_flow/training/flow_noc/000' + sequence + '_10.png')

        sq_error, msen, pepn = msen_pepn(pred_flow, gt_flow)
        print("#### SEQUENCE ", sequence)
        print("> MSEN = ", str(msen))
        print("> PEPN = ", str(pepn))

        plotOF.visualise_error_histogram(gt_flow, sq_error, msen, title='- sequence ' + sequence)

        xx, yy = np.mgrid[0:sq_error.shape[0], 0:sq_error.shape[1]]

        # Create the figure
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(xx,yy,sq_error,rstride=1,cstride=1,cmap=plt.cm.get_cmap("Reds"),linewidth=1)

        plt.show()


def task4():
    plotOF = PlotOF()
    sequences = ['045', '157']
    step = 10
    for idx, sequence in enumerate(sequences):
        pred_flow = read_flow('../results_opticalflow_kitti/results/LKflow_000' + sequence + '_10.png')
        gt_flow = read_flow('../data_stereo_flow/training/flow_noc/000' + sequence + '_10.png')
        path_img = '../data_stereo_flow/training/image_0/000'+ sequence +'_10.png'
        plotOF.magnitudeOP(gt_flow,path_img)
        plotOF.magnitudeOP(pred_flow, path_img)
        plotOF.plotArrowsOP(gt_flow, step, path_img)
        plotOF.plotArrowsOP(pred_flow, step, path_img)


if __name__ == '__main__':

    task11()
    #task12()
    #task2()
    #task3()
    #task4()


