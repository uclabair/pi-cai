from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchio as tio
import warnings
import glob
import os
import nibabel as nib
warnings.filterwarnings("ignore")
import scipy


def make_loader(patient_ids,  whole_prostate_path,  dataset, get_annotation_masks, shuffle_=False,  sampler=None,
                batch_size=1, phase='train', patch=False):
    return DataLoader(
        dataset=dataset(patient_ids, get_annotation_masks, whole_prostate_path, phase, patch),
        shuffle=shuffle_,
        sampler=sampler,
        num_workers=4,
        batch_size=batch_size
    )


def resample(image, old_spacing, order, new_spacing=[1, 1, 1]):
    spacing = np.array(list(old_spacing ))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order=order)
    return image


def squeeze_tensors(subject, anatomical_masks = False, get_annotation_masks = False):
    # resize to remove empty dim after using torchio

    squeezed_tensors = {}
    
    squeezed_tensors['t2'] = subject.t2.numpy().squeeze(axis = 0)
    squeezed_tensors['adc'] = subject.adc.numpy().squeeze(axis = 0)
    squeezed_tensors['highb'] = subject.highb.numpy().squeeze(axis = 0)
    
    if anatomical_masks:
        squeezed_tensors['t2_anatomical'] = subject.t2_anatomical.numpy().squeeze(axis = 0)
        squeezed_tensors['adc_anatomical'] = subject.adc_anatomical.numpy().squeeze(axis = 0)
        squeezed_tensors['highb_anatomical'] = subject.highb_anatomical.numpy().squeeze(axis = 0)
        
    if get_annotation_masks:
        squeezed_tensors['t2_annotation'] = subject.t2_annotation.numpy().squeeze(axis = 0)
        squeezed_tensors['adc_annotation'] = subject.adc_annotation.numpy().squeeze(axis = 0)
        squeezed_tensors['highb_annotation'] = subject.highb_annotation.numpy().squeeze(axis = 0)
        
    return squeezed_tensors
    
    
def unsqueeze_tensors(t2, adc, highb, anatomical_masks = False, get_annotation_masks = False, 
                      t2_anatom = None, adc_anatom = None, highb_anatom = None, 
                     t2_annotation = None, adc_annotation = None, highb_annotation = None):
    # in order to use torchio - need the arrays to have 4 dim not 3

    unsqueezed_tensors = {}
    
    unsqueezed_tensors['t2'] = t2[np.newaxis, :, :]
    unsqueezed_tensors['adc'] = adc[np.newaxis, :, :]
    unsqueezed_tensors['highb'] = highb[np.newaxis, :, :]
    
    if anatomical_masks:
        unsqueezed_tensors['t2_anatomical'] = t2_anatom[np.newaxis, :, :]
        unsqueezed_tensors['adc_anatomical'] = adc_anatom[np.newaxis, :, :]
        unsqueezed_tensors['highb_anatomical'] = highb_anatom[np.newaxis, :, :]
    
    if get_annotation_masks:
        unsqueezed_tensors['t2_annotation'] = t2_annotation[np.newaxis, :, :]
        unsqueezed_tensors['adc_annotation'] = adc_annotation[np.newaxis, :, :]
        unsqueezed_tensors['highb_annotation'] = highb_annotation[np.newaxis, :, :]

    return unsqueezed_tensors


def get_annot_masks(t2, adc, highb, annotation):
    # get annotation masks if requested

    t2_annotation = (t2 * annotation)
    adc_annotation = (adc * annotation)
    highb_annotation = (highb * annotation)

    return t2_annotation, adc_annotation, highb_annotation

def get_anatomical_masks(t2, adc, highb, whole_prostate):
    # get whole prostate masks if requested

    t2_anatomical = (t2 * whole_prostate)
    adc_anatomical = (adc * whole_prostate)
    highb_anatomical = (highb * whole_prostate)

    return t2_anatomical, adc_anatomical, highb_anatomical


class PicaiDataset(Dataset):
    def __init__(
        self,
        patient_ids,
        path_to_images,
        path_to_labels,
        get_annotation_masks = False,
        whole_prostate_path = None,
        phase = 'train',       
        patch = False
    ):
        self.patient_ids = patient_ids
        self.path_to_images = path_to_images
        self.path_to_labels = path_to_labels
        self.median_voxel = np.array([0.5, 0.5, 3. ], dtype=np.float32)
        self.phase = phase
        self.patch = patch
        if whole_prostate_path is None:
            self.anatomical_masks = False
        else:
            self.anatomical_masks = True
            
        self.get_annotation_masks = get_annotation_masks
        self.whole_prostate_path = whole_prostate_path


    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
    
        t2_full = nib.load(os.path.join(self.path_to_images, f'{patient_id}_0000.nii.gz'))
        t2 = t2_full.get_fdata()
        if np.array_equal(t2_full.header['pixdim'][1:4], self.median_voxel) == False:
            t2 = resample(t2, old_spacing=t2_full.header['pixdim'][1:4], new_spacing=self.median_voxel, order=3)

        adc_full = nib.load(os.path.join(self.path_to_images, f'{patient_id}_0001.nii.gz'))
        adc = adc_full.get_fdata()
        if np.array_equal(adc_full.header['pixdim'][1:4], self.median_voxel) == False:
            adc = resample(adc, old_spacing=adc_full.header['pixdim'][1:4], new_spacing=self.median_voxel, order=3)

        highb_full = nib.load(os.path.join(self.path_to_images, f'{patient_id}_0002.nii.gz'))
        highb = highb_full.get_fdata()
        if np.array_equal(highb_full.header['pixdim'][1:4], self.median_voxel) == False:
            highb = resample(highb, old_spacing=highb_full.header['pixdim'][1:4], new_spacing=self.median_voxel, order=3)

        annotation = None
        whole_prostate = None

        if self.anatomical_masks: # check if we have segmentations of prostate provided
            anatomical_path = os.path.join(self.whole_prostate_path,f'{patient_id}.nii.gz')
            whole_prostate_full = nib.load(anatomical_path)
            whole_prostate = whole_prostate_full.get_fdata()
            if np.array_equal(highb_full.header['pixdim'][1:4], self.median_voxel) == False:
                whole_prostate = resample(
                                        whole_prostate, 
                                        old_spacing = whole_prostate_full.header['pixdim'][1:4], 
                                        new_spacing = self.median_voxel, 
                                        order = 3)
                
            
            # get anatomical masks as well, just in case
            t2_anatomical, adc_anatomical, highb_anatomical = get_anatomical_masks(t2, adc, highb, whole_prostate)
            
            whole_prostate = whole_prostate[None, :, :, :]
            
            annotation_path = os.path.join(self.path_to_labels, f'{patient_id}.nii.gz')
            # annotation_path = patient_paths[-1]
            annotation_full = nib.load(annotation_path)
            annotation = annotation_full.get_fdata()
            
            if np.array_equal(highb_full.header['pixdim'][1:4], self.median_voxel) == False:
                annotation = resample(annotation, old_spacing=annotation_full.header['pixdim'][1:4], new_spacing=self.median_voxel, order=3)

            # check if we want to get annotation masks
            if self.get_annotation_masks:
                t2_annotation, adc_annotation, highb_annotation = get_annot_masks(t2, adc, highb, annotation)
            else:
                t2_annotation = None
                adc_annotation = None
                highb_annotation = None
            
            annotation = annotation[None, :, :, :]  #!!!Katya - no need for transpose here
            
            # add extra dimension to tensors before create tio subject
            unsqueezed_tensors = unsqueeze_tensors(
                                                t2, 
                                                adc, 
                                                highb, 
                                                anatomical_masks = self.anatomical_masks, 
                                                get_annotation_masks = self.get_annotation_masks, 
                                                t2_anatom = t2_anatomical, 
                                                adc_anatom = adc_anatomical, 
                                                highb_anatom = highb_anatomical,
                                                t2_annotation = t2_annotation, 
                                                adc_annotation = adc_annotation, 
                                                highb_annotation = highb_annotation
                                                )
            
            # crop to annatomical mask - in order to do this we make a torchio subject first
            if self.get_annotation_masks:
                # create tio subject so all arrays can be cropped to same region
                subject = tio.Subject(
                     t2 = tio.ScalarImage(tensor = unsqueezed_tensors['t2']),
                     adc = tio.ScalarImage(tensor = unsqueezed_tensors['adc']),
                     highb = tio.ScalarImage(tensor = unsqueezed_tensors['highb']),
                     t2_anatomical = tio.ScalarImage(tensor = unsqueezed_tensors['t2_anatomical']),
                     adc_anatomical = tio.ScalarImage(tensor = unsqueezed_tensors['adc_anatomical']),
                     highb_anatomical = tio.ScalarImage(tensor = unsqueezed_tensors['highb_anatomical']),
                     t2_annotation = tio.ScalarImage(tensor = unsqueezed_tensors['t2_annotation']),
                     adc_annotation = tio.ScalarImage(tensor = unsqueezed_tensors['adc_annotation']),
                     highb_annotation = tio.ScalarImage(tensor = unsqueezed_tensors['highb_annotation']),
                     annotation = tio.LabelMap(tensor = annotation),
                     whole_prostate = tio.LabelMap(tensor = whole_prostate)
                 )
            else:
                # if we dont need annotation masks saved
                subject = tio.Subject(
                     t2 = tio.ScalarImage(tensor = unsqueezed_tensors['t2']),
                     adc = tio.ScalarImage(tensor = unsqueezed_tensors['adc']),
                     highb = tio.ScalarImage(tensor = unsqueezed_tensors['highb']),
                     t2_anatomical = tio.ScalarImage(tensor = unsqueezed_tensors['t2_anatomical']),
                     adc_anatomical = tio.ScalarImage(tensor = unsqueezed_tensors['adc_anatomical']),
                     highb_anatomical = tio.ScalarImage(tensor = unsqueezed_tensors['highb_anatomical']),
                     annotation = tio.LabelMap(tensor = annotation),
                     whole_prostate = tio.LabelMap(tensor = whole_prostate)
                 )
            
            # crop tensors
            target_shape = 256, 256, 20
            crop_pad = tio.CropOrPad(target_shape, mask_name = 'whole_prostate')
            subject = crop_pad(subject)

            # remove empty dim
            squeezed_tensors = squeeze_tensors(
                                            subject, 
                                            anatomical_masks = self.anatomical_masks, 
                                            get_annotation_masks = self.get_annotation_masks)
            
            input_3mri = np.array([squeezed_tensors['t2'], squeezed_tensors['adc'], squeezed_tensors['highb']])#.transpose((0, 2, 3, 1))
            input_3anatomical = np.array([squeezed_tensors['t2_anatomical'], squeezed_tensors['adc_anatomical'], squeezed_tensors['highb_anatomical']])
            
            if self.get_annotation_masks:
                # if we have the annotation masks
                input_3annotation = np.array([squeezed_tensors['t2_annotation'], squeezed_tensors['adc_annotation'], squeezed_tensors['highb_annotation']])
                data = np.concatenate((
                    input_3mri, 
                    input_3anatomical, 
                    input_3annotation, 
                    subject.annotation.numpy(), 
                    subject.whole_prostate.numpy()
                ), 
                0)
            else: # otherwise just work with anatomical masks and mris
                data = np.concatenate((
                    input_3mri, 
                    input_3anatomical, 
                    subject.annotation.numpy(), 
                    subject.whole_prostate.numpy()
                ), 
                0)
        else:
            # if we do not have a path to whole prostates

            annotation_path = os.path.join(self.path_to_labels, f'{patient_id}.nii.gz')
            annotation_full = nib.load(annotation_path)
            annotation = annotation_full.get_fdata()
            if np.array_equal(highb_full.header['pixdim'][1:4], self.median_voxel) == False:
                annotation = resample(annotation, old_spacing=annotation_full.header['pixdim'][1:4], new_spacing=self.median_voxel, order=3)
            
            input_3mri = np.array([t2, adc, highb])
            
            if self.get_annotation_masks:
                # check if we want to get annotation masks in this case
                t2_annotation, adc_annotation, highb_annotation = get_annot_masks(t2, adc, highb, annotation)
                annotation = annotation[None, :, :, :]
                input_3annotation = np.array([t2_annotation, adc_annotation, highb_annotation])
                data = np.concatenate((input_3mri, input_3annotation, annotation), 0)
                
            else:        
                annotation = annotation[None, :, :, :]  #!!!Katya - no need for transpose here
                # crop to prespecified area
                data = np.concatenate((input_3mri, annotation), 0)
            
            shape1 = data.shape
            target_shape = 256, 256, 20
            crop_pad = tio.CropOrPad(target_shape)
            data = crop_pad(data)

        # get cspca label
        cspca_label = (np.all(annotation == 0) == False).astype(np.uint8)

        # data augmentation for train phase
        if self.phase == 'train':
            #anisotropy = tio.RandomAnisotropy(p = 0.3)
            flip = tio.RandomFlip(axes=[0,1], flip_probability = 0.5)
            #swap = tio.RandomSwap(patch_size = 4, num_iterations = 100, p = 0.5)
            spatial = tio.OneOf({
                tio.RandomAffine(scales = 0, 
                                translation = 0,
                                degrees = (10, 10, 0)): 0.5,
                tio.RandomElasticDeformation(max_displacement = (10, 10, 0)): 0.25,
            },
            p = 0.6)
            # data = anisotropy(data)
            data = flip(data)
            #data = swap(data)
            data = spatial(data)

        # get mri data and normalize it
        input_3mri = data[0:3]
        input_3mri = input_3mri.transpose((0, 3, 1, 2))

        normalize = tio.ZNormalization()
        input_3mri = normalize(input_3mri)
    
        #!!!Katya here should be augmented annotations
        annotation = data[5:6].transpose((0, 3, 1, 2)).astype(np.uint8)
        data_batch = {
            'input_mris':input_3mri,
            'cspca_label':cspca_label
            }
        
        annotation_3 = None
        
        # option to get annotation or anatomical masks in case needed
        if self.get_annotation_masks:
            if self.anatomical_masks:
                input_3annotation = data[6:9]
            else:
                input_3annotation = data[3:6]
            data_batch['annotation_masks'] = input_3annotation
            
        if self.anatomical_masks:
            input_3anatomical = data[3:6]
            data_batch['anatomical_masks'] = input_3anatomical
            

            

        return data_batch