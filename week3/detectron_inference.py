import os, json, cv2, random
import matplotlib.pyplot as plt 

from tqdm import tqdm
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from dataReader import ReadData
from evaluation import *
import pickle
from createPlots import PlotCreator




def detectronModels(model_name,video_path,xml_path,init_frame, end_frame):
    # set up configuration
    cfg = get_cfg()
    model_name = 'fasterRCNN'
    # Adding the configuration to a desired model
    if model_name == 'retinaNet':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
    elif model_name == 'fasterRCNN':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    ############### INFERENCE
    cap = cv2.VideoCapture(video_path)

    if model_name == 'retinaNet':
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.3
    else:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    predictor = DefaultPredictor(cfg)

    frames = []
    scores = []
    labels = []
    bboxes = []
    boxesScore_pkl=[]

    for frame_idx in tqdm(range(init_frame,end_frame)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        im = cap.read()[1]
        output = predictor(im)

        car_instances = output["instances"][output["instances"].pred_classes==2].to("cpu")
        box_pkl = []
        for idx in range(len(car_instances.pred_boxes)):
            bbox = car_instances.pred_boxes[idx]
            score = car_instances.scores[idx]

            bboxes.append(bbox.tensor[0])
            frames.append(frame_idx)
            scores.append(score)
            labels.append('Car')

            box_pkl.append([bbox.tensor[0].cpu().numpy(),score.cpu().numpy()])
        boxesScore_pkl.append(box_pkl)

    with open('boxesScores.pkl', 'wb') as f:
        pickle.dump(boxesScore_pkl, f)

    prediction_results = [frames,scores,labels,bboxes]
    with open('prediction_results_retina.pkl', 'wb') as f:
        pickle.dump(prediction_results, f)

    reader = ReadData(xml_path)
    predictionsInfo = reader.fixFormat(frames, bboxes, labels, scores, False)
    gt, num_iter = reader.getGTfromXML()
    sortedFrames, sortedBBOX, numBBOX = reader.bboxInFrame(gt=gt,initFrame=init_frame,endFrame=end_frame-1)
    gtInfo = reader.joinBBOXfromFrame(sortedFrames, sortedBBOX, isGT=True)

    gtInfo = reader.resetGT(gtInfo)
    rec, prec, ap, meanIoU,meanIoUF = ap_score(gtInfo, predictionsInfo, num_bboxes=len(bboxes), ovthresh=0.5)


    showVid = True
    if showVid:
        graph = PlotCreator()
        graph.plotVid(init_frame, end_frame, cap, gtInfo, predictionsInfo)

    return rec,prec,ap,meanIoU,meanIoUF