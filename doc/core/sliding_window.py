from skimage.color import rgb2gray
import numpy as np
from openslide import open_slide, __library_version__ as openslide_version
from typing import Dict
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from typing import Dict,Tuple
import argparse
from PIL import Image
import re
import time
os.chdir("C:\\Users\\aroni\\Documents\\adl-f22\\adl-fall2022-prj")
from doc.core.utils import custom_logger

def read_slide(slide, pos, level, dimension, as_float=False):
    im = slide.read_region(pos, level, dimension)
    im = im.convert('RGB') # drop the alpha channel
    if as_float:
        im = np.asarray(im, dtype=np.float32)
    else:
        im = np.asarray(im)
    return im

class SlideWindow:

    def __init__(self,
    slide,
    tumor_mask,
    windows_dim : Tuple[int],
    level : int,
    stride : int,
    threshold : Dict[str,int]
    ) -> None:
        self.slide = slide
        self.tumor_mask = tumor_mask
        self.windows_dim = windows_dim
        self.level = level
        self.stride = stride
        self.threshold = threshold
        level_dimension = slide.level_dimensions[level]
        tot_slide = read_slide(self.slide,np.array((0,0)),self.level,level_dimension)
        self.gray_slide = rgb2gray(tot_slide)
        self.windows =[
            (x,y) for y in range(0,level_dimension[1],stride) for x in range(0,level_dimension[0],stride)
            ]
        self.num_windows = len(self.windows)
        self.windows = iter(self.windows)

    
    def move_window(self):
            x,y = next(self.windows)
            pixels_in_window = self.gray_slide[y:(y+self.stride),x:(x+self.stride)]
            percentage_gray_pixel = np.mean(pixels_in_window<=self.threshold["gray"])
            if percentage_gray_pixel>self.threshold["percentage_tissues"]:
                self.pos = np.array((x,y)) + np.array((self.stride,self.stride))//2 - np.array(self.windows_dim)//2
                return True
            else :
                return False
    
    def get_zoomed_imgs(
        self,
        levels_zoom,
        patch_name,
        file
        )->Dict:
        for zoom in levels_zoom:
            self.zoomed_pos = self.pos + np.array(self.windows_dim)//2
            self.zoomed_pos -= np.array(self.windows_dim)//(2**(zoom+1))
            self.zoomed_pos = (self.zoomed_pos*(2**(self.level))).astype("int32")
            img = read_slide(
                self.slide,
                self.zoomed_pos,
                self.level-zoom,
                self.windows_dim
                )
            img = Image.fromarray(img)
            dir_path = os.path.join("./data/patches","zoom_x"+str(2**zoom))
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            file_name = os.path.join(dir_path,patch_name)
            img.save(file_name)
            mask_image = read_slide(
                self.tumor_mask,
                self.zoomed_pos,
                self.level-zoom,
                self.windows_dim
                )
            mask_image = mask_image[:,:,0]
            file.write(file_name+",")
        file.write(str(int((mask_image>0).any()))+"\n")

    
def build_training_patches(logger,slide_name,tumor_mask_name,args,file):
    logger.info(f"Building the training patches for the slide {slide_name}")
    slide_path = os.path.join('./data',slide_name) # only this file is available
    tumor_mask_path = os.path.join('./data', tumor_mask_name) # only this file is available

    slide_url = 'https://storage.googleapis.com/adl2022-slides/%s' % slide_name
    mask_url = 'https://storage.googleapis.com/adl2022-slides/%s' % tumor_mask_name

    # Download the whole slide image
    if not os.path.exists(slide_path):
        logger.info(f"Missing slide downloading it from {slide_url}")
        os.chdir("./data/")
        os.system(f"curl -O {slide_url}")
        os.chdir("..")

    # Download the tumor mask
    if not os.path.exists(tumor_mask_path):
        logger.info(f"Missing mask slide downloading it from {mask_url}")
        os.chdir("./data/")
        os.system(f"curl -O {mask_url}")
        os.chdir("..")
    
    logger.info("Opening the slide/mask")
    slide = open_slide(slide_path)
    tumor_mask = open_slide(tumor_mask_path)
    sw = SlideWindow(
        slide,
        tumor_mask,
        windows_dim=(args.window_size,args.window_size),
        level=args.level,
        stride=args.stride,
        threshold={
            "gray" : args.gray_threshold,
            "percentage_tissues" : args.pixel_threshold
        }
        )
    num_patches_extracted = 0
    t = time.time()
    with tqdm(range(sw.num_windows),leave=False) as p_bar:
        for i in p_bar:
            if sw.move_window():
                sw.get_zoomed_imgs(
                    args.number_of_zooms,
                    "_".join([re.sub(".tif","",slide_name),str(i)])+".jpg",
                    file
                    )
                num_patches_extracted +=1
    logger.info(f"Extracted {num_patches_extracted} from the slide {slide_name} took {round(time.time()-t)} seconds")
    os.remove(slide_path)
    os.remove(tumor_mask_path)
    logger.info("Deleting from disk the downloaded slides")


if __name__ == "__main__":
    logger = custom_logger()
    parser = argparse.ArgumentParser(description="Build the number of patches from an input slide")
    parser.add_argument('--number_of_zooms',type=int,nargs='+',default=[0,1,3])
    parser.add_argument('--slides',type=str,default=None,nargs='+')
    parser.add_argument('--window_size',type=int,default=299)
    parser.add_argument('--stride',type=int,default=32)
    parser.add_argument('--level',type=int,default=3)
    parser.add_argument('--gray_threshold',type=float,default=0.8)
    parser.add_argument('--pixel_threshold',type=float,default=0.2)
    args = parser.parse_args()
    if args.slides == None:
        with open("./data/slides_name.txt","r") as file:
            slides = file.read().splitlines()
    else :
        slides = args.slides
    with open("./data/patches_tumor_label.csv","w") as file:
        with tqdm(slides) as p_bar:
            p_bar.set_description("Slide extracted ")
            for slide in p_bar:
                slide_name = "tumor_"+slide+".tif"
                tumor_mask_name = "tumor_"+slide+"_mask.tif"
                build_training_patches(logger,slide_name,tumor_mask_name,args,file)


        