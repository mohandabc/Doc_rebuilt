import numpy as np
from skimage.segmentation import felzenszwalb,  slic, quickshift, watershed
from skimage.measure import label
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import sobel
from skimage import exposure
from time import time
import cv2

def getLargestCC(segmentation):
    labels, _ = label(segmentation, background=0, return_num=True, connectivity=1)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    return largestCC.astype(np.uint8)

def remove_hair(img):
    image = img.copy()
    if image.dtype not in (np.float64, np.float32, np.uint8):
        raise TypeError('Provided image is not in a supported data type : float64, float32 or uint8')

    if len(image.shape) < 3 or image.shape[2] <3:
        return image.astype(np.float64)
    
    if image.dtype in (np.float64, np.float32):
        # cause image come normalized
        image = image * 255
        image = image.astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY )
    kernel = cv2.getStructuringElement(1,(17,17))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    ret,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    dst = cv2.inpaint(image,thresh2,1,cv2.INPAINT_TELEA)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    
    dst = normalize(dst).astype(np.float64)
    return dst 

def expand_img(img):
    h_flip = img.copy()[:,::-1]
    line = np.concatenate((h_flip ,img, h_flip), axis=1)

    v_flip = line.copy()[::-1, :]
    final = np.concatenate((v_flip, line, v_flip), axis=0)
    return final

# remove black edges
def remove_black_corners(img):
    """Remove the black corners of dermoscopic images.
    
    This function converts the RGB image into a grayscale image then 
    finds indecies where the image should be cropped.
    It starts from the top left corners and scans pixels diagonally 
    and deletes according lines and columns until one of two conditions 
    is met ; either the pixel value is above a predifined threshold, or a 
    predefined number of lines and columns have already been deleted.
    This last condition ensures not cropping the whole image in case of 
    large lesion that touches the borders.

    INPUTS:
    - img : and RGB image.

    OUTPUTS:
    - i, j : Indecies; crop image from column i to j and line i to j.
    """

    if img.dtype not in (np.uint8, np.float32, np.float64):
        raise TypeError("Image data type should be unit8, float32 or float64")
    
    if img.dtype == np.uint8:
        img = (img / 255).astype(np.float64)
    #convert to grayscal image
    gray = img.copy().astype(np.float64)
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = rgb2gray(img)
      
    threshold = 0.3
    #@BUG this makes no sens, why shape 0 then 1, why not the smallest, and 60% is too much 
    i_limit = img.shape[0]*0.6
    j_limit = img.shape[1]*0.6 
    stop = False
    i = 1
    while not stop:
        if (gray[i, i]>threshold).all() or i >= i_limit:
            stop = True
        else:
            i+=2
    stop = False
    j = -2
    while not stop:
        if (gray[j, j]>threshold).all() or j <= -j_limit:
            stop = True
        else:
            j-=2
    
    return i, j


def superpixelate(img, method, config=None):
    """Takes an image and generates a map of superpixels using the methode 
    specified in argument

    Args:
        img (numpy array): numpy array representing the input image
        method : method to use to superpixelate image

    Returns:
        numpy array: image devided into superpixels
    """

    configurations = {
    "slic": {'n_segments' : 200, 'compactness' : 5, 'sigma': 50, 'start_label': 1},
    "quickshift": {'kernel_size' : 3, 'max_dist' : 6, 'ratio' : 0.5},
    "felzenszwalb": {'scale' : 50, 'sigma': 0.5, 'min_size': 50},
    "watershed" : {'markers':250, 'compactness':0.001}
        }
    
    cfg = configurations[method]
    if config != None:
        cfg = config


    try:
        if method == 'slic':
            segments = slic(img, n_segments=cfg['n_segments'], compactness=cfg['compactness'], sigma=cfg['sigma'], start_label=cfg['start_label'])
        elif method == 'felzenszwalb':
            segments = felzenszwalb(img, scale=cfg['scale'], sigma=cfg['sigma'], min_size=cfg['min_size'])
        elif method =='quickshift':
            segments = quickshift(img, kernel_size=cfg['kernel_size'], max_dist=cfg['max_dist'], ratio=cfg['ratio'])
        elif method == 'watershed':
            segments = watershed(sobel(rgb2gray(img)), markers = cfg['markers'], compactness=cfg['compactness'])
    except:
        raise Exception("Super pixel method failed")
    return segments

def find_borders(map, superpixel):
    """"Finds borders of the window that contains a superpixel
    
    INPUTS: 
    - map : the map of superpixels generated
    - superpixel : the superpixel considered

    OUTPUTS:
    - first_x, last_x, first_y, last_y : Coordinates of the window that contains a superpixel
    """
    W = map.shape[0]
    L = map.shape[1]
    first_y = L
    last_y = 0

    first_x = W
    last_x = 0
    
    found_line = False
    for i in range(0, W, 3):
        where = np.where(map[i] == superpixel)
        len_where = len(where[0])
        
        if(len_where>0):
            found_line= True
            min_y = where[0][0] #np.min(where)
            max_y = where[0][-1] #np.max(where)
            if min_y < first_y : first_y = min_y
            if max_y > last_y : last_y = max_y

            if first_x == W : first_x = i
            if i > last_x : last_x = i
        else:
            if found_line == True:
                break
    return first_x, last_x, first_y, last_y

def find_borders_depricated(mask):
    """"
    Depricated, precise but slower
    Finds borders of the window that contains a superpixel
    
    INPUTS: 
    - mask : the mask of superpixels generated

    OUTPUTS:
    - first_x, last_x, first_y, last_y : Coordinates of the window that contains a superpixel
    """


    first_x = -1
    last_x = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]: 
                last_x = i
                if first_x == -1:
                    first_x = i
                    
    first_y = -1
    last_y = 0
    for i in range(mask.shape[1]):
        for j in range(mask.shape[0]):
            if mask[j, i]: 
                last_y = i
                if first_y == -1:
                    first_y = i
    return first_x, last_x, first_y, last_y


def timer(function):
    def wrapper(*args, **kwargs):
        start = time()
        res = function(*args, **kwargs)
        end = time()
        print(f'{function.__name__}: {end - start} s')
        return res
    return wrapper

def normalize(img):
    return img / 255
def sRGB_to_linear(img, normalize = False):
    if normalize == True:
        return normalize(img)**2.2
    return (img)**2.2
def sRGB_to_XYZ(img):
    res = img.copy()
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.072175 ],
        [0.0193339, 0.119192 , 0.9503041]
        ])
    for i in range(img.shape[0]):
        for j in range (img.shape[1]):
            res[i,j] = np.dot(M, (img[i,j])**2.2)*100
    return res

def convert_data(img, data_type):
    """
    input shape (x, y, 3)
    output shape (x, y, 1)
    """
    def rgb2r(img):
        return img[:,:,0]
    def rgb2g(img):
        return img[:,:,1]
    def rgb2b(img):
        return img[:,:,2]
    def rgb2rg(img):
        return img[:,:,0:2]
    def rgb2gb(img):
        return img[:,:,1:3]
    def rgb2rb(img):
        return img[:,:,0:3:2]

    operations = {
        'gray' : rgb2gray,
        'R' : rgb2r,
        'G' : rgb2g,
        'B' : rgb2b,
        'RG' : rgb2rg,
        'GB' : rgb2gb,
        'RB' : rgb2rb,
        'HSV' : rgb2hsv,
        'XYZ' : sRGB_to_XYZ,
    }
    return operations[data_type](img)

def _prepare_images(segmentation_result, ground_truth):
    seg_res = segmentation_result
    if len(segmentation_result.shape)>2:
        seg_res = segmentation_result[:,:,0]

    g_truth = ground_truth
    if len(ground_truth.shape)>2:
        g_truth = ground_truth[:,:,0]
    return seg_res, g_truth


def Histogram_equalization(image):
    image = exposure.equalize_hist(image)
    return image

def contrast_stretching(image):
    # Apply contrast stretching
    p2, p98 = np.percentile(image, (2, 98))
    image = exposure.rescale_intensity(image, in_range=(p2, p98))
    return image

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