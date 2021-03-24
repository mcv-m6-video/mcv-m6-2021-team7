from dataReader import ReadData
from video_init import VideoModel
from torchvision.models import detection
from torchvision.transforms import transforms
import cv2
import torch
from createPlots import PlotCreator
from evaluation import *
from tqdm import tqdm



def torchModel(model_name, video_path, xml_path,init_frame,end_frame):

    reader = ReadData(xml_path)
    gt, num_iter = reader.getGTfromXML()
    vid = VideoModel(path=video_path,color_space='gray')
    vidLen = vid.retVidLen()
    sortedFrames, sortedBBOX, numBBOX = reader.bboxInFrame(gt, init_frame,end_frame)
    gtInfo = reader.joinBBOXfromFrame(sortedFrames, sortedBBOX, isGT=True)

    if model_name == 'maskRCNN':
        model = detection.maskrcnn_resnet50_fpn(pretrained=True)

    vidCapture = cv2.VideoCapture(video_path)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    with torch.no_grad():
        thr = [0.3]
        tensor = transforms.ToTensor()
        count = 0
        for minConf in thr:
            labels = []
            boxes = []
            scores = []
            frames = []
            for fr in tqdm(range(init_frame, end_frame)):

                vidCapture.set(cv2.CAP_PROP_POS_FRAMES, fr)
                im = vidCapture.read()[1]

                x = [tensor(im).to(device)]
                bbox_pred = model(x)[0]

                ordered_bbox = list(zip(bbox_pred['labels'], bbox_pred['scores'], bbox_pred['boxes']))
                # get car bboxes
                car_bbox = []
                for pred in ordered_bbox:
                    if pred[0] == 3 and pred[1]>minConf:
                        car_bbox.append(pred)


                for box in car_bbox:
                    labels.append('car')
                    scores.append(box[1])
                    boxes.append(box[2])
                    frames.append(fr)

                    count += 1

            predictionsInfo = reader.fixFormat(frames, boxes, labels, scores, False)
            gtInfo = reader.resetGT(gtInfo)
            rec, prec, ap, meanIoU, meanIoUF = ap_score(gtInfo, predictionsInfo, num_bboxes=len(boxes), ovthresh=0.5)



        showVid = True

        if showVid:
            graph = PlotCreator()
            graph.plotVid(init_frame,end_frame,vidCapture,gtInfo,predictionsInfo)

    return rec,prec,ap,meanIoU,meanIoUF

