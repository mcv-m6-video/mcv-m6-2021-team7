import matplotlib.pyplot as plt

class PlotCreator:
    def __init__(self):
        super(PlotCreator, self).__init__()
    def plotCurve(self,datax,datay,labelx,labely):
        plt.plot(datax, datay)
        plt.xlabel(labelx)
        plt.ylabel(labely)
        plt.show()

    def plotPR(self, rec, prec, labelx, labely):
        for iou_t in range(0, 10):
            iou_step = 0.5 + (iou_t * 0.05)
            # plt.plot(recall[iou_t], precision[iou_t], label='P-R Thres:' + '{0:.2f}'.format(iou_step)+'(mAP:''{0:.2f}'.format(map[iou_t])+')')
            plt.plot(rec[iou_t], prec[iou_t],
                     label='P-R Thres:' + '{0:.2f}'.format(iou_step))

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title('Precision-Recall Curve')
        plt.legend(shadow=True)
        plt.grid()
        plt.show()
