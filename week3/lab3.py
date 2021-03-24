from createPlots import PlotCreator
from evaluation import *
import detectron_inference
import torchNets
from video_init import VideoModel
from tqdm import trange
import os, json, cv2, random
import matplotlib.pyplot as plt
import pickle
import random

import motmetrics as mm
import time
from dataReader import ReadData
from sort import Sort
from utils import *

from tqdm import tqdm
from detectron2.utils.logger import setup_logger

import pycocotools
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LVISEvaluator
from detectron2.data import build_detection_test_loader
from LossEvalHook import LossEvalHook
from detectron2.data import DatasetMapper

def task11():
    # Read gt file
    model_name = 'retinaNet'
    path = 'ai_challenge_s03_c010-full_annotation.xml'
    video_path = '/home/mar/Desktop/M6/Lab1/AICity_data/train/S03/c010/vdo.avi'

    #Return video length
    vid = VideoModel(video_path,'rgb')
    vidLen = vid.retVidLen()

    # INIT FRAME and END FRAME
    init_frame = int(vidLen*0.25)
    end_frame = int(vidLen)

    summary = []

    if model_name == 'fasterRCNN' or model_name == 'retinaNet':
        rec,prec,ap,meanIoU,meanIoUF=detectron_inference.detectronModels(model_name,video_path,path,init_frame,end_frame)
    elif model_name == 'maskRCNN':
        rec,prec,ap,meanIoU,meanIoUF=torchNets.torchModel(model_name,video_path,path,init_frame,end_frame)
    else:
        print('Model not implemented!')

    print('Model name:',model_name)
    print('mAP:', ap)
    print('Mean IoU:', meanIoU)
    summary.append(
        {"iou": meanIoU, "mAP": np.mean(ap),'iouFrame':meanIoUF})

    with open('miou'+model_name+'.pkl', 'wb') as f:
        pickle.dump(meanIoUF, f)

    plotsPKL = False
    if plotsPKL:
        graph = PlotCreator()
        graph.plotMIOU()

def task12_B():
    lr = 0.0025
    batch_size = 256
    n_iter = 300
    EXPERIMENT_NAME = 'K1_' + str(lr) + 'lr_' + str(batch_size) + 'bsize_' + str(n_iter) + 'iter'

    def get_aicity_dataset(frame_idx_list):
        path = '/home/group09/code/week6/datasets/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml'
        video_path = '/home/group09/code/week6/datasets/AICity_data/train/S03/c010/vdo.avi'

        reader = ReadData(path)
        gt, num_iter = reader.getGTfromXML()

        sortedFrames, sortedBBOX, numBBOX = reader.bboxInFrame(gt, 0,2141)
        gtInfo = reader.joinBBOXfromFrame(sortedFrames, sortedBBOX, isGT=True)

        dataset_dicts = []
        directory = '/home/group09/code/week6/datasets/AICity_data/AICity_frames'
        for frame_idx in tqdm(frame_idx_list):
            filename = str(frame_idx).zfill(4) + '.png'
            record = {}
            im_path = os.path.join(directory, filename)
            im = cv2.imread(im_path)
            print(filename)
            height, width = im.shape[:2]

            record["file_name"] = im_path
            record["image_id"] = str(frame_idx).zfill(4)
            record["height"] = height
            record["width"] = width

            classes = ['Car']

            objs = []
            for [x1, y1, x2, y2] in gtInfo[frame_idx]['bbox']:  # for every bbox in a frame's gt
                class_id = 0
                obj = {
                    "type": 'Car',
                    "bbox": [x1, y1, x2, y2],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": 0
                }

                objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)

        return dataset_dicts

    # K-FOLD SPLITS
    k_train = 0  # take 0th k-fold for train
    k_step = int(np.floor(2141 * 0.25))
    frame_idx = [i for i in range(2141)]
    ini_frame_train = k_step * k_train

    # train dataset
    train_frame_idx = frame_idx[ini_frame_train:(ini_frame_train + k_step)]
    print("==== TRAIN SPLIT")
    print("")
    for d in ['train']:
        DatasetCatalog.register('train_retina', lambda d=d: get_aicity_dataset(train_frame_idx))
        MetadataCatalog.get('train_retina').set(thing_classes=['Car'])

    # val dataset
    val_frame_idx = [x for x in frame_idx if x not in train_frame_idx]
    print("==== VALIDATION SPLIT")
    print("")
    for d in ['val']:
        DatasetCatalog.register('val_retina', lambda d=d: get_aicity_dataset(val_frame_idx))
        MetadataCatalog.get('val_retina').set(thing_classes=['Car'])

    train_metadata = MetadataCatalog.get("train_retina")
    dataset_dicts = get_aicity_dataset(train_frame_idx)

    OUTPUT_DIR = '/home/group09/code/week6/models_retina/' + EXPERIMENT_NAME

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    cfg = get_cfg()
    cfg.OUTPUT_DIR = OUTPUT_DIR
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("train_retina",)
    cfg.DATASETS.TEST = ("val_retina",)
    # cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/retinanet_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = lr  # pick a good LR
    cfg.SOLVER.MAX_ITER = n_iter  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (car)

    cfg.TEST.EVAL_PERIOD = 100

    class MyTrainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            return COCOEvaluator(dataset_name, cfg, True, output_folder)

        def build_hooks(self):
            hooks = super().build_hooks()
            hooks.insert(-1, LossEvalHook(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                build_detection_test_loader(
                    self.cfg,
                    self.cfg.DATASETS.TEST[0],
                    DatasetMapper(self.cfg, True)
                )
            ))
            return hooks

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

def task12_C():
    lr = 0.0025
    batch_size = 256
    n_iter = 300
    EXPERIMENT_NAME = 'Random_' + str(lr) + 'lr_' + str(batch_size) + 'bsize_' + str(n_iter) + 'iter'

    def get_aicity_dataset(frame_idx_list):
        path = '/home/group09/code/week6/datasets/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml'
        video_path = '/home/group09/code/week6/datasets/AICity_data/train/S03/c010/vdo.avi'
        reader = ReadData(path)
        gt, num_iter = reader.getGTfromXML()

        sortedFrames, sortedBBOX, numBBOX = reader.bboxInFrame(gt, 0,2141)
        gtInfo = reader.joinBBOXfromFrame(sortedFrames, sortedBBOX, isGT=True)

        dataset_dicts = []
        directory = '/home/group09/code/week6/datasets/AICity_data/AICity_frames'
        for frame_idx in tqdm(frame_idx_list):
            filename = str(frame_idx).zfill(4) + '.png'
            record = {}
            im_path = os.path.join(directory, filename)
            im = cv2.imread(im_path)

            height, width = im.shape[:2]

            record["file_name"] = im_path
            record["image_id"] = str(frame_idx).zfill(4)
            record["height"] = height
            record["width"] = width

            classes = ['Car']

            objs = []
            for [x1, y1, x2, y2] in gtInfo[frame_idx]['bbox']:  # for every bbox in a frame's gt
                class_id = 0
                obj = {
                    "type": 'Car',
                    "bbox": [x1, y1, x2, y2],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": 0
                }

                objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)

        return dataset_dicts

    # K-FOLD SPLITS
    k_step = int(np.floor(2141 * 0.25))
    frame_idx = [i for i in range(2141)]

    # train dataset
    train_frame_idx = random.sample(frame_idx, k_step)
    print("==== TRAIN SPLIT")
    print("")
    for d in ['train']:
        DatasetCatalog.register('train_retina', lambda d=d: get_aicity_dataset(train_frame_idx))
        MetadataCatalog.get('train_retina').set(thing_classes=['Car'])

    # val dataset
    val_frame_idx = [x for x in frame_idx if x not in train_frame_idx]
    print("==== VALIDATION SPLIT")
    print("")
    for d in ['val']:
        DatasetCatalog.register('val_retina', lambda d=d: get_aicity_dataset(val_frame_idx))
        MetadataCatalog.get('val_retina').set(thing_classes=['Car'])

    train_metadata = MetadataCatalog.get("train_retina")
    dataset_dicts = get_aicity_dataset(train_frame_idx)

    OUTPUT_DIR = '/home/group09/code/week6/models_retina/' + EXPERIMENT_NAME

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    cfg = get_cfg()
    cfg.OUTPUT_DIR = OUTPUT_DIR
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("train_retina",)
    cfg.DATASETS.TEST = ("val_retina",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/retinanet_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = lr  # pick a good LR
    cfg.SOLVER.MAX_ITER = n_iter  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (car)

    cfg.TEST.EVAL_PERIOD = 100

    class MyTrainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            return COCOEvaluator(dataset_name, cfg, True, output_folder)

        def build_hooks(self):
            hooks = super().build_hooks()
            hooks.insert(-1, LossEvalHook(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                build_detection_test_loader(
                    self.cfg,
                    self.cfg.DATASETS.TEST[0],
                    DatasetMapper(self.cfg, True)
                )
            ))
            return hooks

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def task21():

    pkl_path = "boxesScores.pkl"
    video_path = 'AICity_data/train/S03/c010/vdo.avi'
    gt_path = 'ai_challenge_s03_c010-full_annotation.xml'
    threshold = 0.5  # minimum iou to consider the tracking between consecutive frames
    kill_time = 90  # nº of frames to close the track of an object
    video = True
    showVid = True
    compute_score = True

    # Get the bboxes
    frame_bboxes = []
    with (open(pkl_path, "rb")) as openfile:
        while True:
            try:
                frame_bboxes.append(pickle.load(openfile))
            except EOFError:
                break
    frame_bboxes = frame_bboxes[0]
    # correct the data to the desired format
    aux_frame_boxes = []
    for frame_b in frame_bboxes:
        auxiliar,_ = zip(*frame_b)
        aux_frame_boxes.append(list(auxiliar))
    frame_bboxes = aux_frame_boxes

    # Once we have done the detection we can start with the tracking
    bbox_per_frame = []
    id_per_frame = []
    frame = frame_bboxes[0]  # load the bbox for the first frame
    # Since we evaluate the current frame and the consecutive, we loop for range - 1
    for Nframe in trange(len(frame_bboxes) - 1,desc="Tracking"):
        next_frame = frame_bboxes[Nframe + 1]

        # assign a new ID to each unassigned bbox
        for i in range(len(frame)):
            new_bbox = frame[i]

            # append the bbox to the list
            bbox_per_id = []
            bbox_per_id.append(list(new_bbox))
            bbox_per_frame.append(bbox_per_id)
            # append the id to the list
            index_per_id = []
            index_per_id.append(Nframe)
            id_per_frame.append(index_per_id)

        # we loop for each track and we compute the iou with each detection of the next frame
        for id in range(len(bbox_per_frame)):
            length = len(bbox_per_frame[id])
            bbox_per_id = bbox_per_frame[id]  # bboxes of a track
            bbox1 = bbox_per_id[length - 1]  # last bbox stored of the track
            index_per_id = id_per_frame[id]  # list of frames where the track appears

            # don't do anything if the track is closed
            if index_per_id[-1] == -1:
                continue

            # get the list of ious, one with each detection of the next frame
            iou_list = []
            for detections in range(len(next_frame)):
                bbox2 = next_frame[detections]  # detection of the next frame
                iou_list.append(iou(np.array(bbox1), bbox2))

            # break the loop if there are no more bboxes in the frame to track
            if len(next_frame) == 0:
                # kill_time control
                not_in_scene = Nframe - index_per_id[-1]  # nº of frames that we don't track this object
                if not_in_scene > kill_time:  # if it surpasses the kill_time, close the track by adding a -1
                    index_per_id.append(-1)
                break

            # assign the bbox to the closest track
            best_iou = max(iou_list)
            # if the mas iou is lower than 0.5, we assume that it doesn't have a correspondence
            if best_iou > threshold:
                best_detection = [j for j, k in enumerate(iou_list) if k == best_iou]
                best_detection = best_detection[0]

                # append to the list the bbox of the next frame
                bbox_per_id.append(list(next_frame[best_detection]))
                index_per_id.append(Nframe + 1)

                # we delete the detection from the list in order to speed up the following comparisons
                del next_frame[best_detection]
            else:
                # kill_time control
                not_in_scene = Nframe - index_per_id[-1]   # nº of frames that we don't track this object
                if not_in_scene > kill_time:  # if it surpasses the kill_time, close the track by adding a -1
                    index_per_id.append(-1)

        frame = next_frame  # the next frame will be the current

    if video:
        # Generate colors for each track
        id_colors = []
        for i in range(len(id_per_frame)):
            color = list(np.random.choice(range(256), size=3))
            id_colors.append(color)

        # Define the codec and create VideoWriter object
        vidCapture = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('task2_1.avi', fourcc, 10.0, (1920,  1080))
        # for each frame draw rectangles to the detected bboxes
        for i in trange(len(frame_bboxes),desc="Video"):
            vidCapture.set(cv2.CAP_PROP_POS_FRAMES, i)
            im = vidCapture.read()[1]
            for id in range(len(id_per_frame)):
                ids = id_per_frame[id]
                if i in ids:
                    id_index = ids.index(i)
                    bbox = bbox_per_frame[id][id_index]
                    color = id_colors[id]
                    cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                  (int(color[0]), int(color[1]), int(color[2])), 2)
                    cv2.putText(im, 'ID: ' + str(id), (int(bbox[0]), int(bbox[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (int(color[0]), int(color[1]), int(color[2])), 2)
            if showVid:
                cv2.imshow('Video', im)
            out.write(im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vidCapture.release()
        out.release()
        cv2.destroyAllWindows()

    if compute_score:
        # Load gt for plot
        reader = ReadData(gt_path)
        gt, num_iter = reader.getGTfromXML()

        # init accumulator
        acc = mm.MOTAccumulator(auto_id=True)

        # Loop for all frames
        for Nframe in trange(len(frame_bboxes),desc="Score"):

            # get the ids of the tracks from the ground truth at this frame
            gt_list = [item[1] for item in gt if item[0] == Nframe]
            gt_list = np.unique(gt_list)

            # get the ids of the detected tracks at this frame
            pred_list = []
            for ID in range(len(id_per_frame)):
                aux = np.where(np.array(id_per_frame[ID]) == Nframe)[0]
                if len(aux) > 0:
                    pred_list.append(int(ID))

            # compute the distance for each pair
            distances = []
            for i in range(len(gt_list)):
                dist = []
                # compute the ground truth bbox
                bboxGT = gt_list[i]
                bboxGT = [item[3:7] for item in gt if (item[0] == Nframe and item[1] == bboxGT)]
                bboxGT = list(bboxGT[0])
                # compute centroid GT
                centerGT = centroid(bboxGT)
                for j in range(len(pred_list)):
                    # compute the predicted bbox
                    bboxPR = pred_list[j]
                    aux_id = id_per_frame[bboxPR].index(Nframe)
                    bboxPR = bbox_per_frame[bboxPR][aux_id]
                    # compute centroid PR
                    centerPR = centroid(bboxPR)
                    d = euclid_dist(centerGT, centerPR)  # euclidean distance
                    dist.append(d)
                distances.append(dist)

            # update the accumulator
            acc.update(gt_list, pred_list, distances)

        # Compute and show the final metric results
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=['idf1'], name='IDF1:')
        strsummary = mm.io.render_summary(summary, formatters={'idf1': '{:.2%}'.format}, namemap={'idf1': 'idf1'})
        print(strsummary)


def task22():

    pkl_path = "boxesScores.pkl"
    video_path = 'AICity_data/train/S03/c010/vdo.avi'
    gt_path = 'ai_challenge_s03_c010-full_annotation.xml'
    video = True
    showVid = True

    # Get the bboxes
    frame_bboxes = []
    with (open(pkl_path, "rb")) as openfile:
        while True:
            try:
                frame_bboxes.append(pickle.load(openfile))
            except EOFError:
                break
    frame_bboxes = frame_bboxes[0]

    # Get the GT
    reader = ReadData(gt_path)
    gt, num_iter = reader.getGTfromXML()

    out = []
    out_id = []
    out_bbox = []
    total_time = 0.0
    total_frames = 0

    # create instance of the SORT tracker
    mot_tracker = Sort()
    # init accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    # Loop for all frames
    for Nframe in trange(len(frame_bboxes),desc="Tracking and Score"):

        # prepare detection format for update the tracker (Kalman Filter)
        trans_dets = []
        for bbox in frame_bboxes[Nframe]:
            # convert from x,y,w,h to x1,y1,x2,y2
            dets = bbox[0]
            x1 = dets[0]
            y1 = dets[1]
            x2 = dets[2] + x1
            y2 = dets[3] + y1
            dets = np.array([x1,y1,x2,y2,bbox[1]])
            trans_dets.append(dets)
        total_frames += 1

        # mot tracker
        start_time = time.time()
        trackers = mot_tracker.update(np.array(trans_dets))
        cycle_time = time.time() - start_time
        total_time += cycle_time

        out.append(trackers)

        # IDF1 evaluation
        # get ground truth centroids
        id_gt = [item[1] for item in gt if item[0] == Nframe]
        id_gt = np.unique(id_gt)
        gt_list = [item[3:7] for item in gt if item[0] == Nframe]
        gt_centroids = [centroid(item) for item in gt_list]

        # get predictions centroids
        pr_list = trackers[:, 0:4]
        out_bbox.append(pr_list)
        id_pr = trackers[:, 4]
        out_id.append(id_pr)
        pr_centroids = [centroid([item[0], item[1], item[2] - item[0], item[3] - item[1]]) for item in pr_list]

        # Compute euclidean distance for each pair
        distances = []
        for i in range(len(gt_list)):
            dist = []
            centerGT = gt_centroids[i]
            for j in range(len(pr_list)):
                centerPR = pr_centroids[j]
                d = euclid_dist(centerGT, centerPR)
                dist.append(d)
            distances.append(dist)

        # update the accumulator
        acc.update(id_gt, id_pr, distances)

    print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

    # Compute and show the final metric results
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['idf1'], name='IDF1:')
    strsummary = mm.io.render_summary(summary, formatters={'idf1': '{:.2%}'.format}, namemap={'idf1': 'idf1'})
    print(strsummary)

    if video:
        # Generate colors for each track
        id_colors = []
        max_id = max([max(list(k)) for k in out_id])
        for i in range(int(max_id)+1):
            color = list(np.random.choice(range(256), size=3))
            id_colors.append(color)

        # Define the codec and create VideoWriter object
        vidCapture = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('task2_2.avi', fourcc, 10.0, (1920,  1080))
        # for each frame draw rectangles to the detected bboxes
        for i in trange(len(frame_bboxes),desc="Video"):
            vidCapture.set(cv2.CAP_PROP_POS_FRAMES, i)
            im = vidCapture.read()[1]
            for id in range(len(out_id[i])):
                id_index = int(out_id[i][id])
                bbox = out_bbox[i][id]
                color = id_colors[id_index]
                cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])),
                              (int(color[0]), int(color[1]), int(color[2])), 2)
                cv2.putText(im, 'ID: ' + str(id_index), (int(bbox[0]), int(bbox[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (int(color[0]), int(color[1]), int(color[2])), 2)
            if showVid:
                cv2.imshow('Video', im)
            out.write(im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vidCapture.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    task11()
    #task12_B()
    #task12_C()
    #task21()
    #task22()



