import numpy as np



def msen_pepn(predicted_flow, gt_flow, motion_error_threshold=3):

    # MSEN
    u_difference = gt_flow[:, :, 0] - predicted_flow[:, :, 0]
    v_difference = gt_flow[:, :, 1] - predicted_flow[:, :, 1]
    squared_error = np.sqrt(u_difference ** 2 + v_difference ** 2)
    squared_error_non_occluded = squared_error[gt_flow[:, :, 2] != 0]

    msen = np.mean(squared_error_non_occluded)

    # PEPN
    n_wrong_pixels = np.sum(squared_error_non_occluded > motion_error_threshold)
    n_pixels_non_occ = len(squared_error_non_occluded)

    pepn = (n_wrong_pixels / n_pixels_non_occ) * 100

    return squared_error, msen, pepn