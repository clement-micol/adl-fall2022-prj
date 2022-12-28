# -*- coding: utf-8 -*-
"""

Adaptation of the Evaluation code from :

https://github.com/computationalpathologygroup/Camelyon16/blob/master/Python/Evaluation_FROC.py 

Modified :
-   the compute_FP_TP_Probs to be faster via only numpy function
-   the computeFROC method to be faster via only numpy function

Evaluation code for the Camelyon16 challenge on cancer metastases detection
"""

import openslide
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from skimage import measure
import os
import pandas as pd

   
def computeEvaluationMask(maskDIR, resolution, level):
    """Computes the evaluation mask.
    
    Args:
        maskDIR:    the directory of the ground truth mask
        resolution: Pixel resolution of the image at level 0
        level:      The level at which the evaluation mask is made
        
    Returns:
        evaluation_mask
    """
    slide = openslide.open_slide(maskDIR)
    dims = slide.level_dimensions[level]
    pixelarray = np.zeros(dims[0]*dims[1], dtype='uint')
    pixelarray = np.array(slide.read_region((0,0), level, dims))
    distance = nd.distance_transform_edt((1-pixelarray[:,:,0]))
    Threshold = 75/(resolution * pow(2, level) * 2) # 75µm is the equivalent size of 5 tumor cells
    binary = distance < Threshold
    filled_image = nd.morphology.binary_fill_holes(binary)
    evaluation_mask = measure.label(filled_image, connectivity = 2) 
    return evaluation_mask
    
    
def computeITCList(evaluation_mask, resolution, level):
    """Compute the list of labels containing Isolated Tumor Cells (ITC)
    
    Description:
        A region is considered ITC if its longest diameter is below 200µm.
        As we expanded the annotations by 75µm, the major axis of the object 
        should be less than 275µm to be considered as ITC (Each pixel is 
        0.243µm*0.243µm in level 0). Therefore the major axis of the object 
        in level 5 should be less than 275/(2^5*0.243) = 35.36 pixels.
        
    Args:
        evaluation_mask:    The evaluation mask
        resolution:         Pixel resolution of the image at level 0
        level:              The level at which the evaluation mask was made
        
    Returns:
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
    """
    max_label = np.amax(evaluation_mask)    
    properties = measure.regionprops(evaluation_mask)
    Isolated_Tumor_Cells = [] 
    threshold = 275/(resolution * pow(2, level))
    for i in range(0, max_label):
        if properties[i].major_axis_length < threshold:
            Isolated_Tumor_Cells.append(i+1)
    return Isolated_Tumor_Cells
    
         
def compute_FP_TP_Probs(Ycorr, Xcorr, Probs, evaluation_mask, Isolated_Tumor_Cells, global_pooling=False):
    """Generates true positive and false positive stats for the analyzed image
    
    Args:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
        evaluation_mask:    The evaluation mask
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
         
    Returns:
        FP_probs:   A list containing the probabilities of the false positive detections
        
        TP_probs:   A list containing the probabilities of the True positive detections
        
        NumberOfTumors: Number of Tumors in the image (excluding Isolate Tumor Cells)
    """

    max_label = np.amax(evaluation_mask)
    TP_probs = np.zeros((max_label,), dtype=np.float32)
    HittedLabels = evaluation_mask[Ycorr,Xcorr]
    FP_probs = Probs[np.where(HittedLabels==0)[0]].to_numpy()
    TP_indices = np.where((HittedLabels>0) & ~(np.isin(HittedLabels,Isolated_Tumor_Cells)))[0]
    if global_pooling :
        temp_TP_label = HittedLabels[TP_indices]
        TP_probs = pd.DataFrame(dict(prob=temp_TP_probs.to_numpy(),label=temp_TP_label))
        TP_probs = TP_probs.groupby("label")["prob"].max().to_numpy()
    else :
        TP_probs = Probs[TP_indices]
    num_of_tumors = max_label - len(Isolated_Tumor_Cells);                             
    return FP_probs, TP_probs, num_of_tumors
 
 
def computeFROC(FROC_data):
    """Generates the data required for plotting the FROC curve
    
    Args:
        FROC_data:      Contains the list of TPs, FPs, number of tumors in each image
         
    Returns:
        total_FPRs:      A list containing the average number of false positives
        per image for different thresholds
        
        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds
    """
    
    unlisted_FPs = np.array([item for sublist in FROC_data[1] for item in sublist])
    unlisted_FPs = np.reshape(unlisted_FPs,(1,len(unlisted_FPs)))
    unlisted_TPs = np.array([item for sublist in FROC_data[2] for item in sublist])
    unlisted_TPs = np.reshape(unlisted_TPs,(1,len(unlisted_TPs)))
    all_probs = np.reshape(np.linspace(0,1,num=100),(100,1))  
    total_FPs = np.sum(unlisted_FPs>=all_probs,axis=1)
    total_TPs = np.sum(unlisted_TPs>=all_probs,axis=1)
    total_FPs = total_FPs/unlisted_FPs.shape[1]
    total_sensitivity = total_TPs/unlisted_TPs.shape[1] 
    return  total_FPs, total_sensitivity
   
   
def plotFROC(total_FPs, total_sensitivity):
    """Plots the FROC curve
    
    Args:
        total_FPRs:      A list containing the false positives rate
        across all image for different thresholds
        
        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds
         
    Returns:
        -
    """    
    fig = plt.figure()
    plt.xlabel('Average Number of False Positives', fontsize=12)
    plt.ylabel('Metastasis detection sensitivity', fontsize=12)  
    fig.suptitle('Free response receiver operating characteristic curve', fontsize=12)
    plt.plot(total_FPs, total_sensitivity, '-', color='#000000')    
    plt.show()       
            
        
        
        
        
        
        