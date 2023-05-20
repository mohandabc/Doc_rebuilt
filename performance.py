from utils import _prepare_images
import numpy as np
from skimage import exposure


def get_true_positives(seg_res, g_truth):
    # return np.count_nonzero(seg_res & g_truth)
    return np.count_nonzero(g_truth[seg_res >= 125]>125)

def get_false_positives(seg_res, g_truth):
    return np.count_nonzero(g_truth[seg_res >= 125]<125)

def get_false_negatives(seg_res, g_truth):
    return np.count_nonzero(g_truth[seg_res < 125]>=125)

def  get_true_negatives(seg_res, g_truth):
    TP = get_true_positives(seg_res, g_truth)
    FP = get_false_positives(seg_res, g_truth)
    FN = get_false_negatives(seg_res, g_truth)
    w, l = seg_res.shape
    N=w*l
    return  N - TP - FP - FN


def compute_performance(segmentation_result, ground_truth):
    
    return {'sensitivity' : compute_sensitivity(segmentation_result, ground_truth),
            'Specificity' : compute_specificity(segmentation_result, ground_truth),
            'Accuracy':compute_accuracy(segmentation_result, ground_truth),
            'jaccard_index':compute_jaccard_index(segmentation_result, ground_truth),
            'dice_coef':compute_dice_coefficient(segmentation_result, ground_truth)}

def compute_accuracy(segmentation_result, ground_truth):
    seg_res, g_truth = _prepare_images(segmentation_result, ground_truth)
    w, l = seg_res.shape
    N=w*l

    TP = get_true_positives(seg_res, g_truth)
    TN = get_true_negatives(seg_res, g_truth)
    return (TP+TN)/N

def compute_specificity(segmentation_result, ground_truth):
    seg_res, g_truth = _prepare_images(segmentation_result, ground_truth)
    TN = get_true_negatives(seg_res, g_truth)
    FP = get_false_positives(seg_res, g_truth)
    return TN/(TN+FP)

def compute_sensitivity(segmentation_result, ground_truth):
    seg_res, g_truth = _prepare_images(segmentation_result, ground_truth)
    TP = get_true_positives(seg_res, g_truth)
    FN = get_false_negatives(seg_res, g_truth)
    return TP/(TP+FN)

def compute_jaccard_index(segmentation_result, ground_truth):
    seg_res, g_truth = _prepare_images(segmentation_result, ground_truth)
    TP = get_true_positives(seg_res, g_truth)
    FN = get_false_negatives(seg_res, g_truth)
    FP = get_false_positives(seg_res, g_truth)

    return TP / (TP + FN + FP)

def compute_dice_coefficient(segmentation_result, ground_truth):
    seg_res, g_truth = _prepare_images(segmentation_result, ground_truth)
    TP = get_true_positives(seg_res, g_truth)
    FN = get_false_negatives(seg_res, g_truth)
    FP = get_false_positives(seg_res, g_truth)

    return (2*TP) / ((2*TP) + FP + FN)

def restore_cut_parts(image, i, j):
    d = 1
    if len(image.shape) == 3:
        w, l, d = image.shape
    else:
        w, l = image.shape
    

    left_block = np.zeros((w, i, d))
    right_block = np.zeros((w, abs(j), d))

    top_block = np.zeros((i, l+i+abs(j), d))
    bottom_block = np.zeros((abs(j), l+i+abs(j), d))

    line = np.concatenate((left_block,image, right_block), axis=1)
    res = np.concatenate((top_block, line, bottom_block), axis=0)
    
    return res

def Histogram_equalization(image):
    image = exposure.equalize_hist(image)
    return image

def contrast_stretching(image):
    # Apply contrast stretching
    p2, p98 = np.percentile(image, (2, 98))
    image = exposure.rescale_intensity(image, in_range=(p2, p98))
    return image