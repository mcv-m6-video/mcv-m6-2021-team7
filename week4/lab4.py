import cv2
import numpy as np
from skimage.feature import match_template
from tqdm import trange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utils
import opticalFlow
import Stabilization
import time
from createPlots import PlotCreator
import timeit
from plotsOF import PlotOF
from evaluationOF import *
from pyflow.pyflow import pyflow
import imageio
import pickle
import motmetrics as mm
from dataReader import ReadData




def task11(gridSearch=False,distance='cv2.TM_CCORR_NORMED'):
    # distance = 'cv2.TM_SQDIFF_NORMED'
    # distance = 'cv2.TM_CCORR_NORMED'

    # Read gt file
    im1 = cv2.imread('/home/mar/Desktop/M6/KITTI/data_stereo_flow/training/image_0/000045_10.png',
                          cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread('/home/mar/Desktop/M6/KITTI/data_stereo_flow/training/image_0/000045_11.png',
                          cv2.IMREAD_GRAYSCALE)
    flow_gt = utils.read_flow('/home/mar/Desktop/M6/KITTI/data_stereo_flow/training/flow_noc/000045_10.png')

    if gridSearch:
        #motion = ['forward','backward']# uncomment when not using the 3d plot
        motion = ['backward']
        blockSize = [4,8,16,32,64]
        searchAreas = np.array(blockSize) * 2 + np.array(blockSize)

    else:
        motion = ['backward']
        blockSize = [32]#32
        searchAreas = [96]#96

    all_msen = np.zeros((len(blockSize), len(searchAreas)))
    all_pepn = np.zeros((len(blockSize), len(searchAreas)))
    minerr = 10000
    start_time = []
    end_time = []
    pepns = []
    msens = []
    for m in motion:
        for i,bs in enumerate(blockSize):
            #quantStep = [int(bs/2),bs]
            quantStep = [int(bs / 2)]
            for q in quantStep:
                start_time.append(time.time())
                for j,sa in enumerate(searchAreas):
                    predicted_flow = opticalFlow.compute_block_matching(im1, im2, m, sa, bs,distance,q)
                    #plotOf = PlotOF()
                    #utils.plot_module(predicted_flow)
                    # if m == 'forward':
                    #     plotOf.plotArrowsOP(predicted_flow, 10, im2)
                    # else:
                    #      plotOf.plotArrowsOP(predicted_flow, 10, im1)
                    squared_error, msen, pepn = msen_pepn(predicted_flow,flow_gt)
                    pepns.append(pepn)
                    msens.append(msen)
                    all_msen[i, j] = msen
                    all_pepn[i, j] = pepn
                    print('Motion: ',m)
                    print('BS: ', bs)
                    print('SA: ', sa)
                    print('msen: ',msen)
                    print('pepn: ',pepn)
                    errsum = msen+pepn
                    if errsum<minerr:
                        bestQ = q
                        minerr = errsum
                        bestArea = sa
                        bestBlock = bs
                        bestMsen = msen
                        bestPepn = pepn
                end_time.append(time.time())
    if gridSearch:
        print('Best BS',bestBlock)
        print('Best AS',bestArea)
        print('Best Q', bestQ)
        print('Best pepn', bestPepn)
        print('Best msen', bestMsen)
        searchAreas = np.array(searchAreas)
        blockSize = np.array(blockSize)
        X, Y = np.meshgrid(searchAreas, blockSize)
        graph = PlotCreator()
        graph.metric_3d_plot(X, Y, all_msen, 'Search area', 'Block size', 'MSEN')
        graph.metric_3d_plot(X, Y, all_pepn, 'Search area', 'Block size', 'PEPN')

    else:
        compTime = [end_time[t] - start_time[t] for t in range(len(start_time))]
        print(compTime)

    ## Auxiliar 2D plots (UNCOMMENT)
    #plotter = PlotCreator()
    #plotter.plot_pepn_msen(pepns, msens, 'Search area', searchAreas)

def task12(visualize=True, alg="Pyflow"):

    img1_path = "/home/mar/Desktop/M6/KITTI/data_stereo_flow/training/image_0/000045_10.png"
    img2_path = "/home/mar/Desktop/M6/KITTI/data_stereo_flow/training/image_0/000045_11.png"
    img1 = cv2.imread(img1_path, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(img2_path, cv2.COLOR_BGR2GRAY)
    gt_flow = utils.read_flow("/home/mar/Desktop/M6/KITTI/data_stereo_flow/training/flow_noc/000045_10.png")

    plotOF = PlotOF()

    if alg == "Farneback":

        start = timeit.default_timer()
        predicted_flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        stop = timeit.default_timer()

        if visualize:
            plotOF.magnitudeOP_save(gt_flow, img1_path, alg, "gt")
            plotOF.magnitudeOP(predicted_flow, img1_path, alg,"pred")
            plotOF.plotArrowsOP_save(gt_flow, 10, img1_path, alg, "gt")
            plotOF.plotArrowsOP_save(predicted_flow, 10, img1_path, alg, "pred")

        squared_error, msen, pepn = msen_pepn(predicted_flow, gt_flow, motion_error_threshold=3)
        print("-- Farneback Algorithm --")
        print("Result MSEN: " + str(msen))
        print("Result PEPN: " + str(pepn))
        print('Time: ', stop - start)

    elif alg == "Lucas-Kanade":

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        start = timeit.default_timer()

        # make sparse algorithm dense passing all the pixels
        height, width = img1.shape
        p0 = np.array([[x, y] for y in range(height) for x in range(width)], dtype=np.float32).reshape((-1, 1, 2))
        p1, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, **lk_params)

        p0 = p0.reshape((height, width, 2))
        p1 = p1.reshape((height, width, 2))
        status = status.reshape((height, width))

        # flow field computed by subtracting prev points from next points
        predicted_flow = p1 - p0
        # discard the pixels without found correspondence
        predicted_flow[status == 0] = 0

        stop = timeit.default_timer()

        if visualize:
            plotOF.magnitudeOP_save(gt_flow, img1_path, alg, "gt")
            plotOF.magnitudeOP_save(predicted_flow, img1_path, alg,"pred")
            plotOF.plotArrowsOP_save(gt_flow, 10, img1_path, alg, "gt")
            plotOF.plotArrowsOP_save(predicted_flow, 10, img1_path, alg, "pred")

        squared_error, msen, pepn = msen_pepn(predicted_flow, gt_flow, motion_error_threshold=3)
        print("-- Lucas-Kanade Algorithm --")
        print("Result MSEN: " + str(msen))
        print("Result PEPN: " + str(pepn))
        print('Time: ', stop - start)

    elif alg == "Pyflow":

        im1 = np.atleast_3d(img1.astype(float) / 255.)
        im2 = np.atleast_3d(img2.astype(float) / 255.)

        # Flow Options:
        alpha = 0.012
        ratio = 0.75
        minWidth = 20
        nOuterFPIterations = 7
        nInnerFPIterations = 1
        nSORIterations = 30
        colType = 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

        start = timeit.default_timer()
        u, v, im2W = pyflow.coarse2fine_flow(
            im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
            nSORIterations, colType)
        stop = timeit.default_timer()
        predicted_flow = np.concatenate((u[..., None], v[..., None]), axis=2)
        #np.save('examples/outFlow.npy', predicted_flow)

        if visualize:
            plotOF.magnitudeOP_save(gt_flow, img1_path, alg, "gt")
            plotOF.magnitudeOP_save(predicted_flow, img1_path, alg,"pred")
            plotOF.plotArrowsOP_save(gt_flow, 10, img1_path, alg, "gt")
            plotOF.plotArrowsOP_save(predicted_flow, 10, img1_path, alg, "pred")


        squared_error, msen, pepn = msen_pepn(predicted_flow, gt_flow, motion_error_threshold=3)
        print("-- Pyflow Algorithm --")
        print("Result MSEN: " + str(msen))
        print("Result PEPN: " + str(pepn))
        print('Time: ', stop - start)

def task21():
    cap = cv2.VideoCapture("travi.mp4")
    seq_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Reference image
    previous_frame = cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
    w = previous_frame.shape[0]
    h = previous_frame.shape[1]

    # stabilized video sequence
    stabilized_sequence = []
    stabilized_sequence.append(previous_frame)

    # cummulative deviation respect to initial frame
    deviation_x = 0.0
    deviation_y = 0.0
    start_time = time.time()
    for i in range(seq_length-1):

        # Current frame
        current_frame = cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)

        # predict flow with block matching
        blockSize = 32
        searchArea = 96
        quantStep = 16
        predicted_flow = opticalFlow.compute_block_matching(previous_frame[:,:,0], current_frame[:,:,0], 'backward', searchArea, blockSize,'cv2.TM_CCORR_NORMED',quantStep)

        # deviation as median of flow vectors. mean was affected by actual movement of objects
        vector_u = predicted_flow[:, :, 1]
        vector_v = predicted_flow[:, :, 0]
        deviation_x += np.median(vector_u)
        deviation_y += np.median(vector_v)

        # stabilizing frame using homography
        H = np.array([[1, 0, -deviation_y], [0, 1, -deviation_x]], dtype=np.float32)
        stabilized_frame = cv2.warpAffine(current_frame, H, (h, w))
        stabilized_sequence.append(stabilized_frame)

        # update previous frame as new reference
        previous_frame = current_frame

    end_time = time.time()
    kargs = { 'duration': fps }
    imageio.mimsave('travi.gif', stabilized_sequence,format='GIF', fps=fps)
    print('DURATION:',end_time-start_time)
    #squared_error, msen, pepn = msen_pepn(predicted_flow,flow_gt)

    #plt.imshow(predicted_flow)
    #plt.show()


def task22(method = 'point_feature_matching'):
    video_path = 'travi.mp4'
    vCapture = cv2.VideoCapture(video_path)
    n_frames = int(vCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(vCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if method == 'point_feature_matching':
        start_time = time.time()
        rad = 50 # define radius
        # Step 1
        # Define the codec for output video
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # Set up output video
        fps = 30
        outVideo = cv2.VideoWriter('video_out.mp4', fourcc, fps, (w, h))

        #Step 2
        _,prev= vCapture.read()
        prev_gray = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)

        # Step 3
        transformations = np.zeros(((n_frames-1),3))
        for i in trange(n_frames-2):
            dx,dy,a,current_gray = opticalFlow.extractRotation_translation(prev_gray,vCapture)
            transformations[i] = [dx,dy,a]
            # Update frame
            prev_gray = current_gray

        # Calc smooth motion btw frames
        motion = np.cumsum(transformations,axis=0)
        smooth_motion = utils.smooth(motion,rad)
        diff = smooth_motion - motion
        transformations_smooth = transformations + diff

        # Apply smoothed camera motion to frames
        vCapture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(n_frames-2):
            available,f = vCapture.read()
            if not available:
                break
            stabilised_f,frame_out = opticalFlow.stabilise_frame(transformations_smooth, i, f, w, h)

            if True:
                if (frame_out.shape[1]  >= 1920):
                    frame_out = cv2.resize(frame_out, (frame_out.shape[1] / 2, frame_out.shape[0] / 2))
                cv2.imshow('BEFORE & AFTER',frame_out)
                cv2.waitKey(10)
                outVideo.write(stabilised_f)
        end_time = time.time()
        print('Duration:',end_time-start_time)

    elif method == 'mesh':
        start_time = time.time()
        x_motion_meshes, y_motion_meshes, x_paths, y_paths = Stabilization.read_video(vCapture)
        sx_paths, sy_paths = Stabilization.stabilize(x_paths, y_paths)
        x_motion_meshes, y_motion_meshes, new_x_motion_meshes, new_y_motion_meshes = Stabilization.get_frame_warp(x_motion_meshes,y_motion_meshes,x_paths, y_paths,sx_paths, sy_paths)
        Stabilization.generate_stabilized_video(vCapture, x_motion_meshes, y_motion_meshes, new_x_motion_meshes, new_y_motion_meshes)
        end_time = time.time()
        print('Duration:',end_time-start_time)

def task31():

    pkl_path = "boxesScores.pkl"
    video_path = '/home/mar/Desktop/M6/Lab1/AICity_data/train/S03/c010/vdo.avi'
    gt_path = 'ai_challenge_s03_c010-full_annotation.xml'
    threshold = 0.6  # minimum iou to consider the tracking between consecutive frames
    kill_time = 90  # nº of frames to close the track of an object
    video = False
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
    cap = cv2.VideoCapture(video_path)
    previous_frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
    bbox_per_frame = []
    id_per_frame = []
    frame = frame_bboxes[0]  # load the bbox for the first frame
    # Since we evaluate the current frame and the consecutive, we loop for range - 1
    for Nframe in trange(len(frame_bboxes) - 1,desc="Tracking"):
        next_frame = frame_bboxes[Nframe + 1]
        current_frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)

        # apply optical flow to improve the bounding box and get better iou with the following frame
        # predict flow with block matching
        blockSize = 16
        searchArea = 96
        quantStep = 16
        method = 'cv2.TM_CCORR_NORMED'
        predicted_flow = opticalFlow.compute_block_matching(previous_frame, current_frame, 'backward', searchArea, blockSize,
                                                method, quantStep)

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

            vectorU = predicted_flow[int(bbox1[1]):int(bbox1[3]),int(bbox1[0]):int(bbox1[2]),0]
            vectorV = predicted_flow[int(bbox1[1]):int(bbox1[3]),int(bbox1[0]):int(bbox1[2]),1]
            dx = vectorU.mean()
            dy = vectorV.mean()
            # apply movemement to the bbox
            new_bbox1 = list(np.zeros(4))
            new_bbox1[0] = bbox1[0] + dx
            new_bbox1[2] = bbox1[2] + dx
            new_bbox1[1] = bbox1[1] + dy
            new_bbox1[3] = bbox1[3] + dy

            # don't do anything if the track is closed
            if index_per_id[-1] == -1:
                continue

            # get the list of ious, one with each detection of the next frame
            iou_list = []
            for detections in range(len(next_frame)):
                bbox2 = next_frame[detections]  # detection of the next frame
                iou_list.append(utils.iou(np.array(new_bbox1), bbox2))

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
        previous_frame = current_frame  # update the frame for next iteration

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
                centerGT = utils.centroid(bboxGT)
                for j in range(len(pred_list)):
                    # compute the predicted bbox
                    bboxPR = pred_list[j]
                    aux_id = id_per_frame[bboxPR].index(Nframe)
                    bboxPR = bbox_per_frame[bboxPR][aux_id]
                    # compute centroid PR
                    centerPR = utils.centroid(bboxPR)
                    d = utils.euclid_dist(centerGT, centerPR)  # euclidean distance
                    dist.append(d)
                distances.append(dist)

            # update the accumulator
            acc.update(gt_list, pred_list, distances)

        # Compute and show the final metric results
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=['idf1'], name='IDF1:')
        strsummary = mm.io.render_summary(summary, formatters={'idf1': '{:.2%}'.format}, namemap={'idf1': 'idf1'})
        print(strsummary)

if __name__ == '__main__':

    #task11()
    #task12()
    task21()
    #task22()
    #task31()





