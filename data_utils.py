class PicaiDataset(Dataset):
    def __init__(
        self,
        image_path,
        label_path,
        get_annotation_masks = False,
        whole_prostate_path = None,
        phase = 'train',       
        patch = False
    ):
        self.image_path = image_path
        self.label_path = label_path
        self.median_voxel = np.array([0.5, 0.5, 3. ], dtype=np.float32)
        self.phase = phase
        self.patch = patch
        self.anatomical_masks = False
        self.get_annotation_masks = get_annotation_masks
        self.whole_prostate_path = whole_prostate_path

        self.image_paths = sorted(glob.glob(os.path.join(self.image_path, '*.nii.gz')))
        self.label_paths = sorted(glob.glob(os.path.join(self.label_path, '*.nii.gz')))
        if self.whole_prostate_path is None:
            self.anatomical_masks = False
            self.reshape_method = 'default'
        else:
            self.anatomical_masks = True
            self.reshape_method = 'mask'
            self.whole_prostate_paths = sorted(glob.glob(os.path.join(self.whole_prostate_path, '*.nii.gz')))
        
        # create dictionary based on patient ids in file paths
        self.picai_set = {}
        for path in self.image_paths:
            id_1, id_2 = path.split('/')[-1].split('.')[0].split('_')[0:2]
            new_key = f'{id_1}_{id_2}'
            if new_key in self.picai_set:
                self.picai_set[new_key].append(path)
            else:
                self.picai_set[new_key] = [path]

        for label in self.label_paths:
            id_ = label.split('/')[-1].split('.')[0]
            if id_ in self.picai_set:
                self.picai_set[id_].append(label)

        if self.anatomical_masks:
            for whole_seg in self.whole_prostate_paths:
                id_ = whole_seg.split('/')[-1].split('.')[0]
                if id_ in self.picai_set:
                    self.picai_set[id_].append(whole_seg)

        self.patient_keys = list(self.picai_set.keys())

    def __len__(self):
        return len(self.patient_keys)

    def __getitem__(self, idx):

        patient_id = self.patient_keys[idx]
        patient_paths = self.picai_set[patient_id]
        
        mri_paths = patient_paths[0:3]

        t2_full = nib.load(mri_paths[0])
        t2 = t2_full.get_fdata()
        adc_full = nib.load(mri_paths[1])
        adc = adc_full.get_fdata()
        highb_full = nib.load(mri_paths[2])
        highb = highb_full.get_fdata()
        
        annotation = None
        whole_prostate = None
        
        input_3mri = np.array([t2, adc, highb]).transpose((0, 2, 3, 1))
        
        if self.anatomical_masks: # check if we have segmentations of prostate provided
            whole_prostate_path = patient_paths[-1]
            whole_prostate_full = nib.load(whole_prostate_path)
            whole_prostate = whole_prostate_full.get_fdata()
            whole_prostate = whole_prostate[None, :, :, :].transpose((0, 2, 3, 1))
            
            annotation_path = patient_paths[-2]
            annotation_full = nib.load(annotation_path)
            annotation = annotation_full.get_fdata()
            annotation = annotation[None, :, :, :].transpose((0, 2, 3, 1))
            
            # crop to annatomical mask - in order to do this we make a torchio subject first
            subject = tio.Subject(
                t2 = tio.ScalarImage(mri_paths[0]),
                adc = tio.ScalarImage(mri_paths[1]),
                highb = tio.ScalarImage(mri_paths[2]),
                annotation = tio.LabelMap(annotation_path),
                whole_prostate = tio.LabelMap(whole_prostate_path)
            )
            target_shape = 256, 256, 20
            canonical = tio.ToCanonical()
            subject = canonical(subject)
            resample = tio.Resample(self.median_voxel)
            subject = resample(subject)
            crop_pad = tio.CropOrPad(target_shape, mask_name = 'whole_prostate')
            subject = crop_pad(subject)
            
            t2 = subject.t2.numpy().squeeze(axis = 0)
            adc = subject.adc.numpy().squeeze(axis = 0)
            highb = subject.highb.numpy().squeeze(axis = 0)
            annotation = subject.annotation.numpy()
            whole_prostate = subject.whole_prostate.numpy()
            input_3mri = np.array([t2, adc, highb])#.transpose((0, 2, 3, 1))
            
            data = np.concatenate((input_3mri, annotation, whole_prostate), 0)

        else:
            whole_prostate_path = None
            annotation_path = patient_paths[-1]
            annotation_full = nib.load(annotation_path)
            annotation = annotation_full.get_fdata()
            annotation = annotation[None, :, :, :].transpose((0, 2, 3, 1))
            
            # crop to prespecified area
            data = np.concatenate((input_3mri, annotation), 0)
            target_shape = 256, 256, 20
            canonical = tio.ToCanonical()
            data = canonical(data)
            resample = tio.Resample(self.median_voxel)
            data = resample(data)
            crop_pad = tio.CropOrPad(target_shape)
            data = crop_pad(data)
            
        # get cspca label
        cspca_label = (np.all(annotation == 0) == False).astype(np.uint8)
        
        
        # data augmentation for train phase
        if self.phase == 'train':
            anisotropy = tio.RandomAnisotropy(p = 0.3)
            flip = tio.RandomFlip(axes=[0,1], flip_probability = 0.5)
            #swap = tio.RandomSwap(patch_size = 4, num_iterations = 100, p = 0.5)
            spatial = tio.OneOf({
                tio.RandomAffine(scales = 0, 
                                translation = 0,
                                degrees = (10, 10, 0)): 0.5,
                tio.RandomElasticDeformation(max_displacement = (10, 10, 0)): 0.25,
            },
            p = 0.6)
            data = anisotropy(data)
            data = flip(data)
            #data = swap(data)
            data = spatial(data)
        
        # get mri data and normalize it
        input_3mir = data[0:3]
        input_3mri = input_3mri.transpose((0, 3, 1, 2))
        normalize = tio.ZNormalization()
        input_3mri = normalize(input_3mri)
        
        annotation = annotation[0].transpose((2, 0, 1)).astype(np.uint8)
        
        data_batch = {
            'input_mris':input_3mri,
            'cspca_label':cspca_label
            }
        annotation_3 = None
        anatomical_3 = None

        # option to get annotation or anatomical masks in case needed
        if self.get_annotation_masks:
            t2_annotation = input_3mri[0] * annotation
            adc_annotation = input_3mri[1] * annotation
            highb_annotation = input_3mri[2] * annotation

            annotation_3 = np.array([t2_annotation, adc_annotation, highb_annotation])
            data_batch['annotation_masks'] = annotation_3

        if self.anatomical_masks:
            whole_prostate = whole_prostate
            whole_prostate = whole_prostate[0].transpose((2, 0, 1)).astype(np.uint8)
            t2_anatomical = input_3mri[0] * whole_prostate
            adc_anatomical = input_3mri[1] * whole_prostate
            highb_anatomical = input_3mri[2] * whole_prostate

            anatomical_3 = np.array([t2_anatomical, adc_anatomical, highb_anatomical])
            data_batch['anatomical_masks'] = anatomical_3
        
        return data_batch
            