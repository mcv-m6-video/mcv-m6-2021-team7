import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import tqdm as trange


def video_with_bbox(path, netw_name, gtInfo, predictionsInfo, initFrame, endFrame):

    cap = cv.VideoCapture(path)
    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(netw_name + '.avi', fourcc, 10.0, (1920,  1080))
    count = 0
    for fn in range(initFrame,endFrame):
        cap.set(cv.CAP_PROP_POS_FRAMES, fn)
        frame = cap.read()[1]

        aux1 = gtInfo[count].get('bbox')
        for bb in aux1:
            if bb is not None:
                bb = np.round(bb).astype(int)
                frame = cv.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 3) # GT - green
        aux2 = predictionsInfo[count].get('bbox').astype(int)
        for bb in aux2:
            np.round(bb).astype(int)
            frame = cv.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1) # Predicted - red

        # draw legend
        sub_img = frame[15:85, 15:230]
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 235
        res = cv.addWeighted(sub_img, 0.2, white_rect, 0.9, 1.0)
        frame[15:85, 15:230] = res
        frame = cv.line(frame,(30,45),(210,45),(0,255,0),2)
        frame = cv.line(frame,(30,75),(210,75),(0,0,255),2)
        cv.putText(frame,'Ground Truth', (30,40), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 1)
        cv.putText(frame,'Detection', (30,70), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 1)
        cv.imshow('frame', frame)

        if cv.waitKey(1) == ord('q'):
            break

        out.write(frame)
        count = count + 1

    cap.release()
    out.release()
    cv.destroyAllWindows()



def meanIoUvideoplot(netw_name,miou):
    fig = plt.figure()
    ax = plt.axes(xlim=(390, 650), ylim=(0, 1), xlabel='Frames', ylabel='IoU Score', title='IoU over time: '+netw_name)

    def animate(i):
        ax.plot(np.linspace(390, i + 390, i), miou[390:i + 390], 'b', linewidth=1.0)

    a = anim.FuncAnimation(fig, animate, frames=260, repeat=False)
    a.save(netw_name + '_IoU.gif')
    plt.savefig(netw_name + '_IoU.png')
    plt.show()