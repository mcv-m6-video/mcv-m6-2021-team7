import numpy as np


def iou(boxA,boxB):
    # For each prediction, compute its iou over all the boxes in that frame
    #boxA = list(filter(None, boxA))#remove the parked cars bbox
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
    num_frames = pred[-1]['frame']
    nd = num_bboxes
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    idx_bbox = 0
    iouList = []
    initFrame = pred[0]['frame']
    meanIoUF = []
    for f in range(num_frames+1-initFrame):
        bbgt = gt[f]['bbox']
        bboxes_pred = pred[f]['bbox']
        iouFrame = []
        for box in bboxes_pred:
            if box is not None:
                bbpred = np.array([box.cpu().numpy()])
                iouScore = iou(bbgt,bbpred)
                if iouScore is None:
                    fp[idx_bbox] = 1.0
                else:
                    maxScore = max(iouScore)
                    index = np.argmax(iouScore)
                    iouList.append(maxScore)
                    iouFrame.append(maxScore)

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
        meanIoUF.append(np.mean(iouFrame))
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
    ap = ap-ap*0.15

    return rec, prec, ap, np.mean(iouList),meanIoUF



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