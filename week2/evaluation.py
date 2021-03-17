import numpy as np


def iou(boxA,boxB):
    # For each prediction, compute its iou over all the boxes in that frame
    boxA = list(filter(None, boxA))#remove the parked cars bbox
    if boxA == []:
        return None
    boxA = np.array(boxA)
    x11, y11, x12, y12 = np.split(boxA, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxB, 4, axis=1)

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



def ap_score(gt,pred,num_bboxes,ovthresh=0.5):


    # go down dets and mark TPs and FPs
    num_frames = gt[-1]['frame']
    nd = num_bboxes
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    idx_bbox = 0
    iouList = []
    initFrame = pred[0]['frame']
    #gt = cutGT(gt, initFrame)
    for f in range(num_frames+1-initFrame):
        bbgt = gt[f]['bbox']
        bboxes_pred = pred[f]['bbox']
        for box in bboxes_pred:
            bbpred = np.array([box])
            if bbpred[0] is not None:
                iouScore = iou(bbgt,bbpred)
                if iouScore is None:
                    fp[idx_bbox] = 1.0
                else:
                    maxScore = max(iouScore)
                    index = np.argmax(iouScore)
                    iouList.append(maxScore)

                    if maxScore > ovthresh:
                        if not gt[f]['is_detected'][index]:
                            # We have detected an existing bbox in the gt
                            gt[f]['is_detected'][index] = True
                            tp[idx_bbox] = 1.0
                        else:
                            fp[idx_bbox] = 1.0
                    else:
                        fp[idx_bbox] = 1.0
            else:
                iouScore = iou(bbgt, bbpred)
                if iouScore is None:
                    nobbox = 1
                else:
                    fp[idx_bbox] = 1.0

            idx_bbox += 1
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(idx_bbox)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    # compute AP with the 11 point metric
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.0

    return rec, prec, ap, np.mean(iouList)



def VOC_ap_score(gt,pred,num_bboxes,ovthresh=0.5):


    def sort_key_def(elem):
        return elem[2]

    # Sort by confidence
    pred_BB = []
    for i in range(len(pred)):
        for i_bb in range(len(pred[i]['bbox'])):
            pred_BB.append([i, pred[i]['bbox'][i_bb], pred[i]['score'][i_bb]])
    pred_bb_sorted = sorted(pred_BB, reverse=True, key=sort_key_def)

    meanIoU = np.zeros(len(pred))

    # go down dets and mark TPs and FPs
    nd = num_bboxes
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for br in range(len(pred_bb_sorted)):
    #for box in bboxes_pred:
        frame_id = pred_bb_sorted[br][0]
        bbpred = np.array([pred_bb_sorted[br][1]]).astype(float)
        bbgt = gt[frame_id]['bbox'].astype(float)

        iouScore = iou(bbgt,bbpred)
        maxScore = max(iouScore)
        meanIoU[frame_id] += maxScore[0]
        index = np.argmax(iouScore)

        if maxScore > ovthresh:
            if not gt[frame_id]['is_detected'][index]:
                # We have detected an existing bbox in the gt
                gt[frame_id]['is_detected'][index] = True
                tp[br] = 1.0
            else:
                fp[br] = 1.0
        else:
            fp[br] = 1.0

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    # compute precision recall
    rec = tp / float(num_bboxes)
    # avoid divide by zero in case the first detection matches a difficult
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    # compute AP with the 11 point metric
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.0

    for i in range(len(meanIoU)):
        meanIoU[i] = meanIoU[i] / len(pred[i]['bbox'])

    return rec, prec, ap, meanIoU
# # Hem de modificar la funció aquesta per a que usar-la amb les nostres dades (és el mAP calculat amb Detectron2)
# #https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/pascal_voc_evaluation.py
# def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False):
#     """rec, prec, ap = voc_eval(detpath,
#                                 annopath,
#                                 imagesetfile,
#                                 classname,
#                                 [ovthresh],
#                                 [use_07_metric])
#     Top level function that does the PASCAL VOC evaluation.
#     detpath: Path to detections
#         detpath.format(classname) should produce the detection results file.
#     annopath: Path to annotations
#         annopath.format(imagename) should be the xml annotations file.
#     imagesetfile: Text file containing the list of images, one image per line.
#     classname: Category name (duh)
#     [ovthresh]: Overlap threshold (default = 0.5)
#     [use_07_metric]: Whether to use VOC07's 11 point AP computation
#         (default False)
#     """
#     # assumes detections are in detpath.format(classname)
#     # assumes annotations are in annopath.format(imagename)
#     # assumes imagesetfile is a text file with each line an image name
#
#     # first load gt
#     # read list of images
#     with PathManager.open(imagesetfile, "r") as f:
#         lines = f.readlines()
#     imagenames = [x.strip() for x in lines]
#
#     # load annots
#     recs = {}
#     for imagename in imagenames:
#         recs[imagename] = parse_rec(annopath.format(imagename))
#
#     # extract gt objects for this class
#     class_recs = {}
#     npos = 0
#     for imagename in imagenames:
#         R = [obj for obj in recs[imagename] if obj["name"] == classname]
#         bbox = np.array([x["bbox"] for x in R])
#         difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
#         # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
#         det = [False] * len(R)
#         npos = npos + sum(~difficult)
#         class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}
#
#     # read dets
#     detfile = detpath.format(classname)
#     with open(detfile, "r") as f:
#         lines = f.readlines()
#
#     splitlines = [x.strip().split(" ") for x in lines]
#     image_ids = [x[0] for x in splitlines]
#     confidence = np.array([float(x[1]) for x in splitlines])
#     BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)
#
#     # sort by confidence
#     sorted_ind = np.argsort(-confidence)
#     BB = BB[sorted_ind, :]
#     image_ids = [image_ids[x] for x in sorted_ind]
#
#     # go down dets and mark TPs and FPs
#     nd = len(image_ids)
#     tp = np.zeros(nd)
#     fp = np.zeros(nd)
#     for d in range(nd):
#         R = class_recs[image_ids[d]]
#         bb = BB[d, :].astype(float)
#         ovmax = -np.inf
#         BBGT = R["bbox"].astype(float)
#
#         if BBGT.size > 0:
#             # compute overlaps
#             # intersection
#             ixmin = np.maximum(BBGT[:, 0], bb[0])
#             iymin = np.maximum(BBGT[:, 1], bb[1])
#             ixmax = np.minimum(BBGT[:, 2], bb[2])
#             iymax = np.minimum(BBGT[:, 3], bb[3])
#             iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
#             ih = np.maximum(iymax - iymin + 1.0, 0.0)
#             inters = iw * ih
#
#             # union
#             uni = (
#                 (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
#                 + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
#                 - inters
#             )
#
#             overlaps = inters / uni
#             ovmax = np.max(overlaps)
#             jmax = np.argmax(overlaps)
#
#         if ovmax > ovthresh:
#             if not R["difficult"][jmax]:
#                 if not R["det"][jmax]:
#                     tp[d] = 1.0
#                     R["det"][jmax] = 1
#                 else:
#                     fp[d] = 1.0
#         else:
#             fp[d] = 1.0
#
#     # compute precision recall
#     fp = np.cumsum(fp)
#     tp = np.cumsum(tp)
#     rec = tp / float(npos)
#     # avoid divide by zero in case the first detection matches a difficult
#     # ground truth
#     prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
#     ap = voc_ap(rec, prec, use_07_metric)
#
#     return rec, prec, ap
