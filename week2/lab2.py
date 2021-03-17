from dataReader import ReadData
from gaussian_bg_model import GaussianModel
from evaluation import *
from createPlots import PlotCreator
import pickle
from soa_bgsubs import BgsModel


def task11_12():
    # Read gt file
    path = 'ai_challenge_s03_c010-full_annotation.xml'
    reader = ReadData(path)

    # Compute the mean and variance for each of the pixels along the 25% of the video
    gaussModel = GaussianModel(path='/home/mar/Desktop/M6/Lab1/AICity_data/train/S03/c010/vdo.avi',color_space='gray')
    vidLen = gaussModel.retVidLen()

    # Load gt for plot
    gt, num_iter = reader.getGTfromXML()
    gt = reader.preprocessGT(gt)
    sortedFrames, sortedBBOX, numBBOX = reader.bboxInFrame(gt, int(vidLen*0.25))
    gtInfo = reader.joinBBOXfromFrame(sortedFrames, sortedBBOX, isGT=True)

    # Compute mean and std
    gaussModel.mean_std_Welford()
    # Separate foreground from background
    alpha = [6.25]
    summary = []
    for a in alpha:
        print('Alpha:', a)
        predictionsInfo,num_bboxes = gaussModel.foreground_extraction(showVid=True,gt=gtInfo,alpha=a,noiseRemoval=True)

        # --------------------------------------TASK 1.2-------------------------------------------------
        gtInfo = reader.resetGT(gtInfo)
        rec, prec, ap, meanIoU = ap_score(gtInfo, predictionsInfo, num_bboxes=num_bboxes, ovthresh=0.5)
        print('mAP:',ap)
        print('Mean IoU:', meanIoU)
        summary.append(
            {"alpha": a,"iou": meanIoU, "mAP": np.mean(ap)})

    plots = False
    if plots:
        graph = PlotCreator()
        graph.plotCurve(datax=[dict['alpha'] for dict in summary], datay=[dict['iou'] for dict in summary],
                        labelx='Alpha', labely='IoU',name='iouAlpha')
        graph.plotCurve(datax=[dict['alpha'] for dict in summary], datay=[dict['mAP'] for dict in summary],
                        labelx='Alpha', labely='mAP',name='mAPAlpha')

def task2():
    # Read gt file
    path = 'ai_challenge_s03_c010-full_annotation.xml'
    reader = ReadData(path)

    # Compute the mean and variance for each of the pixels along the 25% of the video
    gaussModel = GaussianModel(path='/home/mar/Desktop/M6/Lab1/AICity_data/train/S03/c010/vdo.avi',color_space='gray')
    vidLen = gaussModel.retVidLen()

    # Load gt for plot
    gt, num_iter = reader.getGTfromXML()
    gt = reader.preprocessGT(gt)
    sortedFrames, sortedBBOX, numBBOX = reader.bboxInFrame(gt, int(vidLen * 0.25))
    gtInfo = reader.joinBBOXfromFrame(sortedFrames, sortedBBOX, isGT=True)


    # Separate foreground from background
    alpha = [3,4,5,6]
    rho_values = [0.001,0.01,0.1,0.5]
    configurations = []
    ap_results = []
    iou_results = []
    for a in alpha:
        for r in rho_values:
            # Compute mean and std
            gaussModel.mean_std_Welford()

            print('Alpha:', a)
            predictionsInfo, num_bboxes = gaussModel.foreground_extraction_task2(showVid=True, gt=gtInfo, alpha=a, rho=r, adaptive=True)

            gtInfo = reader.resetGT(gtInfo)
            rec, prec, ap, meanIoU = ap_score(gtInfo, predictionsInfo, num_bboxes=num_bboxes, ovthresh=0.5)
            configurations.append([a,r])
            ap_results.append(ap)
            iou_results.append(meanIoU)
            print("")
            print("Alpha: ", a)
            print("Rho: ", r)
            print('mAP:', ap)
            print('Mean IoU:', meanIoU)

    with open('ap_results_2.pkl', 'wb') as handle:
        pickle.dump(ap_results, handle)
    with open('iou_results_2".pkl', 'wb') as handle:
        pickle.dump(iou_results, handle)
    with open('configurations_2.pkl', 'wb') as handle:
        pickle.dump(configurations, handle)

def task3():

    algorithms = ['KNN','MOG2','CNT','GMG','GSOC','MOG'] #'LSBP'

    for i in range(len(algorithms)):
        # Read gt file
        path = "ai_challenge_s03_c010-full_annotation.xml"
        reader = ReadData(path)

        model = BgsModel(path='/home/mar/Desktop/M6/Lab1/AICity_data/train/S03/c010/vdo.avi', color_space='gray',alg=algorithms[i])
        vidLen = model.retVidLen()

        gt, num_iter = reader.getGTfromXML()
        gt = reader.preprocessGT(gt)
        sortedFrames, sortedBBOX, numBBOX = reader.bboxInFrame(gt, int(vidLen * 0.25))
        gtInfo = reader.joinBBOXfromFrame(sortedFrames, sortedBBOX, isGT=True)

        predictionsInfo, num_bboxes = model.foreground_extraction(showVid=True, gt=gtInfo, use_postprocessing=True)

        # Uncomment if original video with bboxes is needed:
        # video_with_bbox("D:\MCV\M6\AICity_data\train\S03\c010\vdo.avi", algorithms[i], gtInfo, predictionsInfo, int(vidLen*0.25), int(vidLen))
        rec, prec, ap, meanIoU = ap_score(gtInfo, predictionsInfo, num_bboxes=num_bboxes, ovthresh=0.5)
        print('Method: ', algorithms[i])
        print('mAP:',ap)
        print('Mean IoU:', meanIoU)


if __name__ == '__main__':

    #task11_12()
    #task2()
    task3()


