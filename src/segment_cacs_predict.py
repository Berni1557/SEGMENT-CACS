# -*- coding: utf-8 -*-

"""
Created on Tue Oct 22 21:59:29 2024

@author: bernifoellmer
"""

import os, sys
import torch
from torch import nn, optim
from model import SegmentCACSModel
from glob import glob
import SimpleITK as sitk
import numpy as np
import pathlib
from scipy.ndimage import label, generate_binary_structure
import json
import argparse
import warnings
#import matplotlib.pyplot as plt

def computeAgatstonArtery(image, mask, spacing, slice_thickness):
    """ Compute agatston score from image and image label

    :param image: Image
    :type image: np.ndarray
    :param mask: Calcium scoring mask
    :type imageLabel: np.ndarray
    :param spacing: image spacing e.g. (0.39, 0.39, 3.0)
    :type spacing: Tuple
    :param slice_thickness: Slice thickness
    :type slice_thickness: float
    """
    
    # Compute overall agatston
    agatston = computeAgatston(image, mask, spacing, slice_thickness)
    
    # Compute agatston per segment
    arteries={'LM': 1, 'LAD-PROXIMAL': 2, 'LAD-MID': 3, 'LAD-DISTAL': 4, 'LAD-SIDE': 5,
              'LCX-PROXIMAL': 6, 'LCX-MID': 7, 'LCX-DISTAL': 8, 'LCX-SIDE': 9,
              'RCA-PROXIMAL': 10, 'RCA-MID': 11, 'RCA-DISTAL': 12, 'RCA-SIDE': 13,
              }
    for key in arteries:
        mask_art = np.zeros(mask.shape)
        mask_art[mask==arteries[key]]=1
        agatston_art = computeAgatston(image, mask_art, spacing, slice_thickness)
        agatston['Agatston'+key] = agatston_art['AgatstonScore']
    return agatston

def densityFactor(value):
    """ Compute density weigt factor for agatston score based on maximum HU value of a lesion

    :param value: Maximum HU value of a lesion
    :type value: int
    """
    if value<130:
        densfactor=0
    elif value>=130 and value<=199:
        densfactor=1
    elif value>199 and value<=299:
        densfactor=2
    elif value>299 and value<=399:
        densfactor=3
    else:
        densfactor=4
    return densfactor


def CACSGrading(value):
    """ Compute agatston grading from agatston score

    :param value: Agatston score
    :type value: float
    """
    if value>1 and value<=100:
        grading = (1, 'minimal')
    elif value>100 and value<=300:
        grading = (2, 'mild')
    elif value>300:
        grading = (3, 'moderate')
    else:
        grading=(0, 'zero')
    return grading

def computeAgatston(image, mask, spacing, slice_thickness):
    """ Compute agatston score from image and image label

    :param image: Image
    :type image: np.ndarray
    :param imageLabel: Image label
    :type imageLabel: np.ndarray
    :param pixelVolume: Volume of a pixel
    :type pixelVolume: float
    """
    
    # Binarize mask
    maskbin = mask.copy()
    maskbin[maskbin>0]=1
    # Extract lesions
    s = generate_binary_structure(3,3)
    mask_comp, num_comp = label(maskbin)
    # Compute parameters
    pixelArea = spacing[0]*spacing[1]
    ratio = slice_thickness/spacing[2]
    agatstonAll = 0
    agatston=dict()
    for c in range(1,num_comp+1):
        mask_les = mask_comp==c
        image_les = image * mask_les
        # Iterate over slices
        for s in range(0,image_les.shape[0]):
            image_les_slice = image_les[s,:,:]
            mask_les_slice = mask_les[s,:,:]
            # Extract maximum HU of a lesion
            attenuation = image_les_slice.max()
            area = mask_les_slice.sum() * pixelArea
            # Calculate density weigt factor
            dfactor = densityFactor(attenuation)
            # Calculate agatston score for a lesion
            agatstonLesionSlice = area * dfactor
            # Scale agatston score based on slice_thickness
            agatstonLesionSlice = agatstonLesionSlice * ratio
            agatstonAll = agatstonAll + agatstonLesionSlice
    agatston['AgatstonScore'] = agatstonAll
    agatston['Grading'] = CACSGrading(agatstonAll)
    return agatston

def readDICOM(fp):
    filepathDcm = glob(fp + '/*.dcm')[0]
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(filepathDcm)
    file_reader.ReadImageInformation()
    series_ID = file_reader.GetMetaData('0020|000e')
    sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(fp, series_ID)
    image_sitk = sitk.ReadImage(sorted_file_names)
    return image_sitk

def main(args):
    
    print('--- Start processing ---')
    
    # Define directories
    data_dir = args.data_dir
    model_dir = args.model_dir
    prediction_dir = args.prediction_dir

    # Create directories
    os.makedirs(prediction_dir, exist_ok=True)

    # Create model
    model = SegmentCACSModel()
    soft = nn.Softmax(dim=1)
    
    # Initialize model and parameter
    params={'device': args.device}
    model.create(params)
    Xmin = -2000
    Xmax = 1300
    slices = 2
    filetype = args.filetype
    
    # Load pretrained model parameter
    print('Loading model: ' + model_dir)
    model.load(model_dir)

    # Load image files from data folder    
    #files = glob(data_dir + '/*.mhd')
    # Load image files from data folder    
    if filetype=='mhd':
        files = glob(data_dir + '/*.mhd')
    elif filetype=='nii':
        files = glob(data_dir + '/*.nii')
    elif filetype=='dcm':
        files = glob(data_dir + '/*')
    else:
        raise ValueError('Processing for filetype ' + filetype + ' is not implemented.')
    
    for file in files:
        
        # Read image
        filename = os.path.splitext(os.path.basename(file))[0]
        print('Loading CT: ' + os.path.basename(file))
        if os.path.isdir(file):
            image_sitk = readDICOM(file)
        else:
            image_sitk = sitk.ReadImage(file)
        image = sitk.GetArrayFromImage(image_sitk)

        # Normalize image data
        image[image==-3024]=-2048
        image_norm = (image - Xmin) / (Xmax - Xmin)
        image_norm_ext = np.zeros((image_norm.shape[0]+2*slices, image_norm.shape[1], image_norm.shape[2]))
        image_norm_ext[slices:-slices] = image_norm
        
        # Lesion candidate mask
        lesion_candidate_mask = np.zeros(image.shape)
        lesion_candidate_mask[image>130] = 1
        
        # Init predictions
        pred_lesion_multi = np.zeros(image.shape)
        pred_region_multi = np.zeros(image.shape)
        
        print('Predicting CT: ' + os.path.basename(file))
        # Iterate over slices
        for s in range(slices, image_norm_ext.shape[0]-slices):
            # Convert to torch tensor
            s0=s-slices
            s1=s+slices+1
            Ximage = torch.FloatTensor(image_norm_ext[s0:s1]).to(args.device)
            Xmask = torch.FloatTensor(lesion_candidate_mask[s0:s0+1]).to(args.device)
            Xin = torch.cat((Ximage, Xmask), dim=0).unsqueeze(0)
            
            # Predict CT slice
            Y_region, Y_lesion, Y_zero = model.model(Xin)
            Y_lesion = soft(Y_lesion)
            Y_region = soft(Y_region)
            Y_zero = soft(Y_zero)
            
            # Delete calcifications in slice if ZERO CAC model predects no CACS
            if Y_zero[0][0]>0.5:
                Y_lesion[0,0,:,:]=1
                Y_lesion[0,1:,:,:]=0
                
            # Convert mask format
            Y_lesion_multi = torch.argmax(Y_lesion, dim=1)
            Y_region_multi = torch.argmax(Y_region, dim=1)
            
            # Filter CACS predictions based on 130 HU theshold mask
            Y_lesion_multi = Xmask*Y_lesion_multi + (1-Xmask)*torch.zeros(Y_lesion_multi.shape)
            if args.device=='cuda':
                Y_lesion_multi = Y_lesion_multi.cuda()
                        
            # Fill predictions
            pred_lesion_multi[s0,:,:] = Y_lesion_multi[0,:,:].cpu()
            pred_region_multi[s0,:,:] = Y_region_multi[0,:,:].cpu()
            
        print('Saveing predictions from: ' + os.path.basename(file))

        # Save predictions
        filepath = os.path.join(prediction_dir, filename + '_multi_label.nrrd')
        Y_region_sitk = sitk.GetImageFromArray(pred_region_multi)
        Y_region_sitk.CopyInformation(image_sitk)
        sitk.WriteImage(Y_region_sitk, filepath, True)
        
        filepath = os.path.join(prediction_dir, filename + '_multi_lesion.nrrd')
        Y_lesion_multi_sitk = sitk.GetImageFromArray(pred_lesion_multi)
        Y_lesion_multi_sitk.CopyInformation(image_sitk)
        sitk.WriteImage(Y_lesion_multi_sitk, filepath, True)
        
        # Compute and save Agatston score per segment
        filepath = os.path.join(prediction_dir, filename + '_agatston.json')
        image_org = sitk.GetArrayFromImage(image_sitk)
        slice_thickness = image_sitk.GetSpacing()[2]
        spacing = image_sitk.GetSpacing()
        agatston = computeAgatstonArtery(image_org, pred_lesion_multi, spacing=spacing, slice_thickness=slice_thickness)
        with open(filepath, 'w') as f:
            json.dump(agatston, f, indent=4)
            
    print('--- Finished processing ---')
        

if __name__ == '__main__':
    

    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(
        description='Coronary Calcium Scoring with Multi-Task model'
    )

    parser.add_argument('--model_dir', '-m', type=str,
                        action='store', dest='model_dir',
                        help='Directory of the model', 
                        default='/mnt/SSD2/cloud_data/Projects/CTP/src/modules/SegmentCACSSeg/github/model/SegmentCACS_0001619_unet.pt')
    parser.add_argument('--data_dir', '-d', type=str,
                        action='store', dest='data_dir',
                        help='Directory of data',
                        default='/mnt/SSD2/cloud_data/Projects/CTP/src/modules/SegmentCACSSeg/github/data')
    parser.add_argument('--prediction_dir', '-p', type=str,
                        action='store', dest='prediction_dir',
                        help='Directory of predictions',
                        default='/mnt/SSD2/cloud_data/Projects/CTP/src/modules/SegmentCACSSeg/github/predict')
    parser.add_argument('--filetype', '-f', type=str,
                        action='store', dest='filetype',
                        help="Filetype of the input images. Filetpye can be 'mhd', 'dcm', 'nii'", 
                        default='mhd')
    parser.add_argument('--device', '-gpu', type=str,
                        action='store', dest='device',
                        help='Device NO. of GPU',
                        default='cuda')

    args = parser.parse_args()
    main(args)
