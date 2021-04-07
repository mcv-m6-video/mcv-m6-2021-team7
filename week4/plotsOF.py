import matplotlib.pyplot as plt
import cv2
import numpy as np
import flow_vis
from matplotlib import colors, mlab
from matplotlib.ticker import PercentFormatter

class PlotOF:
    def __init__(self):
        super(PlotOF, self).__init__()

    def visualise_error_histogram(self,gt_flow, sq_error, msen, title=""):
        flow_valid = gt_flow[:, :, 2]
        err_non_occluded = sq_error[flow_valid != 0]

        x = err_non_occluded
        n_bins = 25

        fig, axs = plt.subplots(tight_layout=True)

        N, bins, patches = axs.hist(x, bins=n_bins, density=True)  # N is the count in each bin

        # [mean_mse, alpha=opacity, y_max=length]
        axs.axvline(msen, alpha=1, ymax=10, linestyle=":", label='MSEN')

        axs.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        axs.set_title('Density of Optical Flow Error ' + title)
        axs.set_xlabel('Optical Flow square error')
        axs.set_ylabel('Percentage of Pixels')

        plt.legend()

        # color code by height
        fracs = N / N.max()

        # we need to normalize the data to 0..1 for the full range of the colormap
        norm = colors.Normalize(fracs.min(), fracs.max())

        # Now, we'll loop through our objects and set the color of each accordingly
        for thisfrac, thispatch in zip(fracs, patches):
            color = plt.cm.viridis(norm(thisfrac))
            thispatch.set_facecolor(color)

        plt.savefig('op_flow_error_hist_' + title + '.png')
        plt.show()

    def plotArrowsOP_save(self,flow_img, step, path, alg, type):
        img = plt.imread(path)
        flow_img = cv2.resize(flow_img, (0, 0), fx=1. / step, fy=1. / step)
        u = flow_img[:, :, 0]
        v = flow_img[:, :, 1]
        x = np.arange(0, np.shape(flow_img)[0] * step, step)
        y = np.arange(0, np.shape(flow_img)[1] * step, step)
        U, V = np.meshgrid(y, x)
        M = np.hypot(u, v)
        plt.quiver(U, V, u, -v, M, color='g')
        plt.imshow(img, alpha=0.5, cmap='gray')
        plt.title('Orientation OF - ' + alg + ' - ' + type)
        plt.xticks([])
        plt.yticks([])
        plt.show()

        plt.savefig('plotArrowsOP - ' + alg + ' - ' + type)

    def plotArrowsOP(flow_img, step, img):
        flow_img = cv2.resize(flow_img, (0, 0), fx=1. / step, fy=1. / step)
        u = flow_img[:, :, 0]
        v = flow_img[:, :, 1]
        x = np.arange(0, np.shape(flow_img)[0] * step, step)
        y = np.arange(0, np.shape(flow_img)[1] * step, step)
        U, V = np.meshgrid(y, x)
        M = np.hypot(u, v)
        plt.quiver(U, V, u, -v, M, color='g')
        plt.imshow(img, alpha=0.5, cmap='gray')
        # plt.colorbar(cmap='Pastel2')
        plt.title('Orientation OF')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def magnitudeOP_save(self,flow_img, path, alg, type):
        img = plt.imread(path)
        flow_color = flow_vis.flow_to_color(flow_img[:, :, :2], convert_to_bgr=False)
        plt.imshow(flow_color)
        plt.imshow(img, alpha=0.2, cmap='gray')
        plt.title('Magnitude OF - ' + alg + ' - ' + type)
        plt.xticks([])
        plt.yticks([])
        plt.show()

        plt.savefig('magnitudeOP - ' + alg + ' - ' + type)