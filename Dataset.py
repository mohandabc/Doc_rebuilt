from random import random
import os
from skimage import io, img_as_ubyte
import numpy as np
from skimage.segmentation import mark_boundaries
from pathlib import Path

from utils import remove_black_corners, superpixelate, find_borders, expand_img
CONFIG = {

    "nb_processed_imgs" : -1,
    "n_levels" : 3, 
    "levels":[
             (31, 31), 
             (63, 63), 
             (127, 127), 
             (255, 255), 
            ],
    "train_split": 0.8,  #70% off all images will be in train folder, 30% in test folder
    "batch_size" : 64,
        }

class Dataset():

    """This class prprocesses images to create datasets to feed CNN"""

    current_image_name = 0
    current_image = []
    current_lesion = []
    
    def __init__(self, base_folder = "", output_folder="Built_dataset\\", images_folder = "Images", masks_folder = "GroundTruth"):
        self.base_path = Path(base_folder)
        self.output_folder = Path(output_folder)
        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.config = CONFIG

        self.output_folder.mkdir(exist_ok=True)

    def set_config(self, key, value):
        self.config[key] = value
 
    def process(self):
        # get images to process (limit : nb_processed_imgs)
        # images = (next(os.walk(self.base_path))[1])[0:self.config['nb_processed_imgs']]
        images = os.listdir(self.base_path / self.images_folder)[0:self.config['nb_processed_imgs']]
        ground_truth = os.listdir(self.base_path / self.masks_folder)[0:self.config['nb_processed_imgs']]

        for image, mask in zip(images, ground_truth):
            print('\nprocessing ', image)
            image_path = self.base_path / self.images_folder / image
            mask_path = self.base_path / self.masks_folder / mask

            # Read the image and the correspondant mask
            img = io.imread(image_path,plugin='matplotlib')
            lesion_mask = io.imread(mask_path, plugin='matplotlib')

            # Remove the black corners
            # Resize the image and the mask to ignore the black corners
            i, j = remove_black_corners(img)
            k=i
            l=j
            self.current_image = img[i:j, k:l, :]
            self.current_lesion = lesion_mask[i:j, k:l]
            if len(self.current_lesion.shape)>2:
                self.current_lesion = self.current_lesion[:,:,0]

            self.expanded_img = expand_img(self.current_image)

            options = {'n_segments' : 500, 'compactness' : 5, 'sigma': 50, 'start_label': 1}
            segments = superpixelate(self.current_image, 'slic', config=options)

            # Create the output directory and the files names
            image_name = image.split(".")[0]
            
            save_extra = False
            if(save_extra):
                try:
                    os.makedirs(self.output_folder/image_name)
                except:
                    pass
                output_image_name = image
                output_mask_name = 'lesion_'+image

                # Save the new resized image, mask and a visual representation of the superpixels
                
                io.imsave(self.output_folder / image_name / output_image_name, img_as_ubyte(self.current_image))
                io.imsave(str(self.output_folder / image_name / output_mask_name), img_as_ubyte(self.current_lesion))
                io.imsave(str(self.output_folder / image_name / f'spx_{output_image_name}'), img_as_ubyte(mark_boundaries(self.current_image, segments)))
                io.imsave(self.output_folder / image_name / f'expand_{output_image_name}', img_as_ubyte(self.expanded_img))
 
            self.save_windows(segments)
            # print("number of windows : " , len(np.unique(segments)))
    
    def save_windows(self, segments):

        n_levels = self.config['n_levels']+1
        for level in range(1, n_levels):
            try:
                base_path = self.output_folder / f"data{str(level)}"
                os.makedirs(base_path / 'train' / '0')
                os.makedirs(base_path / 'train' / '1')
                os.makedirs(base_path / 'test' / '0')
                os.makedirs(base_path / 'test' / '1')
            except:
                pass
        
        unique_segments = np.unique(segments)
        for c in unique_segments:
            rand = random()
            folder = 'test'
            if rand < self.config['train_split']:
                folder = 'train'
            sp_class = 0
            self.current_image_name += 1
            
            # these variables are used to save the start x, y and end x,y 
            # of the first window (ref window)
            base_first_x = 0
            base_last_x = 0
            base_first_y = 0
            base_last_y = 0

         
            for level in range(1, n_levels):
                if level == 1:
                    #LEVEL 1
                    # mask = segments==c
                    first_x, last_x, first_y, last_y = find_borders(segments, c)
                    # The length (last_x - first_x) should be odd in order to have a center
                    # Same for (last_y - first_y)
                    if((last_x - first_x +1)%2) == 0:
                        last_x -= 1
                    if((last_y - first_y +1)%2) == 0:
                        last_y -= 1
                    base_first_x, base_last_x, base_first_y, base_last_y = first_x, last_x, first_y, last_y

                    # Find the classification of the center
                    sp_class = self.current_lesion[first_x + (last_x-first_x)//2, first_y + (last_y-first_y)//2]#//255
                else:
                    # FIXME: Manage the border oveflow
                    # FIXME: Make sur legths are odd
                    first_x = int(base_first_x - ((base_last_x-base_first_x)/2*(level-1)))
                    # if first_x < 0 : first_x = 0

                    last_x = base_last_x + ((base_last_x-base_first_x)/2*(level-1))
                    # if last_x > self.current_image.shape[0] : last_x = self.current_image.shape[0]-1

                    first_y = int(base_first_y - ((base_last_y-base_first_y)/2*(level-1)))
                    # if first_y < 0 : first_y = 0
                    last_y = base_last_y + ((base_last_y-base_first_y)/2*(level-1))
                    # if last_y > self.current_image.shape[1] : last_y = self.current_image.shape[1]-1
                    
                    # first_x = first_x.item()
                    last_x = int(last_x)
                    # first_y = first_y.item()
                    last_y = int(last_y)

                shift_x = self.current_image.shape[0]
                shift_y = self.current_image.shape[1]

                    # print(f'   -------{first_x+shift_x}-------')
                    # print(f'   {first_y+shift_y}-------{last_y+shift_y}')
                    # print(f'   -------{last_x+shift_x}-------\n')

                res = self.expanded_img[(first_x+shift_x):(last_x+shift_x+1), (first_y+shift_y):(last_y+shift_y+1), :]
                # res = self.current_image[first_x:last_x+1, first_y:last_y+1, :]
                io.imsave(self.output_folder / f'data{level}' / folder / f'{round(sp_class)}' / f'{self.current_image_name}.png', img_as_ubyte(res), check_contrast=False)

            
    