import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
import nibabel as nib
import random
from scipy.ndimage import label, find_objects
import torch.nn.functional as F

# class UTSW_CNS(Dataset):
#     """ Liver Dataset """
#     def __init__(self, base_dir=None, split='train', num=None, transform=None, bbox_shift=None, fix_bbox=True, pos_neg=False):
#         self._base_dir = base_dir
#         self.transform = transform
#         self.bbox_shift = bbox_shift
#         self.fix_bbox = fix_bbox
#         self.pos_neg = pos_neg # make the bbox bigger or smaller
#         self.sample_list = []
#         self.num_slice = 128
#         self.split = split
#         if split=='train':
#             with open(self._base_dir+'/../dataset/cns_brain/train_cns_ctv_rel_path.txt', 'r') as f:
#                 self.curMR_list = f.readlines()
#         elif split == 'val':
#             with open(self._base_dir+'/../dataset/cns_brain/val_cns_ctv_rel_path.txt', 'r') as f:
#                 self.curMR_list = f.readlines()
#         elif split == 'test':
#             with open(self._base_dir+'/../dataset/cns_brain/test_cns_ctv_rel_path.txt', 'r') as f:
#                 self.curMR_list = f.readlines()
#         elif split == 'all':
#             with open(self._base_dir+'/../dataset/cns_brain/cns_combined_all_ctv_data_rel_path.txt', 'r') as f:
#                 self.curMR_list = f.readlines()
#         self.curMR_list = [str(item.replace(' \n', '')) for item in self.curMR_list]
#         if num is not None:
#             self.curMR_list = self.curMR_list[:num]
#         print("total {} samples".format(len(self.curMR_list)))
#         self.abs_path = os.path.join(base_dir, "../dataset/cns_brain")

#     def __len__(self):
#         return len(self.curMR_list)

#     def __getitem__(self, idx):
#         ## get bbox
#         if self.fix_bbox :
#             fun_get_box = get_all_bounding_boxes_with_ccl_box_fix
#         elif self.fix_bbox and self.pos_neg:
#             fun_get_box = get_all_bounding_boxes_with_ccl_box_shift_pos_neg
#         else:
#             fun_get_box = get_all_bounding_boxes_with_ccl_box_shift

#         curMR_name = self.curMR_list[idx] # path of the files: curMR, preMR, preGT, curGT
#         ## open files
#         curMR_paths = curMR_name.split(" ")
#         curMR = nib.load(os.path.join(self.abs_path, curMR_paths[0])).get_fdata()
#         preMR = nib.load(os.path.join(self.abs_path, curMR_paths[1])).get_fdata()
#         preGT = nib.load(os.path.join(self.abs_path, curMR_paths[2])).get_fdata()
#         curGT = nib.load(os.path.join(self.abs_path, curMR_paths[3])).get_fdata() # HWD
#         caseID = curMR_paths[0].split('/')[-2] # like: 70125695/70125695_20240828_mr.nii.gz

#         ## normalization
#         curMR = (curMR - curMR.min()) / (curMR.max() - curMR.min() + 1e-10)
#         preMR = (preMR - preMR.min()) / (preMR.max() - preMR.min() + 1e-10)
#         # curMR = (curMR - np.mean(curMR)) / np.std(curMR)
        
#         ## convert to tensor
#         curMR = curMR.reshape(1, curMR.shape[0], curMR.shape[1], curMR.shape[2]).astype(np.float32) # 1HWD
#         preMR = preMR.reshape(1, preMR.shape[0], preMR.shape[1], preMR.shape[2]).astype(np.float32)
#         preGT = preGT.reshape(1, preGT.shape[0], preGT.shape[1], preGT.shape[2]).astype(np.float32)
#         curGT = curGT.reshape(1, curGT.shape[0], curGT.shape[1], curGT.shape[2]).astype(np.float32)
        
#         if "train" in self.split:
#             curMR, curGT = preprocess_mri_and_mask(curMR, curGT, target_depth=128, target_hw=(512, 512), mode='train')
#             preMR, preGT = preprocess_mri_and_mask(preMR, preGT, target_depth=128, target_hw=(512, 512), mode='train')
#             D = curGT.shape[-1] # D is the number of slices
#             bboxes = np.zeros((D, 4))
#             for num in range(D):
#                 curGT_temp = curGT.numpy()[0,...]
#                 get2D = curGT_temp[:,:,num]
#                 bboxes[num, :] = fun_get_box(get2D, self.bbox_shift)
#         else:
#             D = curGT.shape[-1] # D is the number of slices
#             bboxes = np.zeros((D, 4))
#             for num in range(D):
#                 get2D = curGT[:,:,num] 
#                 bboxes[num, :] = fun_get_box(get2D, self.bbox_shift)       


#         bboxes = torch.tensor(bboxes).float() # Dx4            
#         sample = {'curMR': curMR, 'preMR': preMR, 'preGT': preGT, "curGT": curGT, "bboxes": bboxes,"caseID": caseID}

#         # if self.transform:
#         #     sample = self.transform(sample)

#         return sample



class UTSW_CNS(Dataset):
    """ Liver Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None, bbox_shift=None, fix_bbox=True, pos_neg=False):
        self._base_dir = base_dir
        self.transform = transform
        self.bbox_shift = bbox_shift
        self.fix_bbox = fix_bbox
        self.pos_neg = pos_neg # make the bbox bigger or smaller
        self.sample_list = []
        self.split = split
        if split=='train':
            with open(self._base_dir+'/../dataset/cns_brain/train_cns_ctv_rel_path.txt', 'r') as f:
                self.curMR_list = f.readlines()
        elif split == 'val':
            with open(self._base_dir+'/../dataset/cns_brain/val_cns_ctv_rel_path.txt', 'r') as f:
                self.curMR_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir+'/../dataset/cns_brain/test_cns_ctv_rel_path.txt', 'r') as f:
                self.curMR_list = f.readlines()
        elif split == 'all':
            with open(self._base_dir+'/../dataset/cns_brain/cns_combined_all_ctv_data_rel_path.txt', 'r') as f:
                self.curMR_list = f.readlines()
        self.curMR_list = [str(item.replace(' \n', '')) for item in self.curMR_list]
        if num is not None:
            self.curMR_list = self.curMR_list[:num]
        print("total {} samples".format(len(self.curMR_list)))
        self.abs_path = os.path.join(base_dir, "../dataset/cns_brain")

    def __len__(self):
        return len(self.curMR_list)

    def __getitem__(self, idx):
        curMR_name = self.curMR_list[idx] # path of the files: curMR, preMR, preGT, curGT
        ## open files
        curMR_paths = curMR_name.split(" ")
        curMR = nib.load(os.path.join(self.abs_path, curMR_paths[0])).get_fdata()
        preMR = nib.load(os.path.join(self.abs_path, curMR_paths[1])).get_fdata()
        preGT = nib.load(os.path.join(self.abs_path, curMR_paths[2])).get_fdata()
        curGT = nib.load(os.path.join(self.abs_path, curMR_paths[3])).get_fdata() # HWD
        caseID = curMR_paths[0].split('/')[1].split('_mr.nii.gz')[0] # like: 70125695/70125695_20240828_mr.nii.gz
        H,W,D = curGT.shape

        ## normalization
        curMR = (curMR - curMR.min()) / (curMR.max() - curMR.min() + 1e-10)
        preMR = (preMR - preMR.min()) / (preMR.max() - preMR.min() + 1e-10)
        # curMR = (curMR - np.mean(curMR)) / np.std(curMR)

        ## get bbox
        if self.fix_bbox :
            fun_get_box = get_all_bounding_boxes_with_ccl_box_fix
        elif self.fix_bbox and self.pos_neg:
            fun_get_box = get_all_bounding_boxes_with_ccl_box_shift_pos_neg
        else:
            fun_get_box = get_all_bounding_boxes_with_ccl_box_shift

        bboxes = np.zeros((D, 4))
        for num in range(D):
            get2D = curGT[:,:,num] 
            bboxes[num, :] = fun_get_box(get2D, self.bbox_shift)
        
        ## convert to tensor
        curMR = curMR.reshape(1, curMR.shape[0], curMR.shape[1], curMR.shape[2]).astype(np.float32) # 1HWD
        curMR = torch.from_numpy(curMR)
        preMR = preMR.reshape(1, preMR.shape[0], preMR.shape[1], preMR.shape[2]).astype(np.float32)
        preMR = torch.from_numpy(preMR)
        preGT = preGT.reshape(1, preGT.shape[0], preGT.shape[1], preGT.shape[2]).astype(np.float32)
        preGT = torch.from_numpy(preGT)
        curGT = curGT.reshape(1, curGT.shape[0], curGT.shape[1], curGT.shape[2]).astype(np.float32)
        curGT = torch.from_numpy(curGT) #HWD
        bboxes = torch.tensor(bboxes).float() # Dx4

        ## because the spatial and temporal resolution of the MRI is different, we need to preprocess the data
        if "train" in self.split:
            curMR, curGT = preprocess_mri_and_mask(curMR, curGT, target_depth=128, target_hw=(512, 512), mode='train')
            preMR, preGT = preprocess_mri_and_mask(preMR, preGT, target_depth=128, target_hw=(512, 512), mode='train')
            ## update bboxes after preprocessing
            D = curGT.shape[-1] # D is the number of slices
            bboxes = np.zeros((D, 4))
            curGT_temp = curGT.numpy()[0,...]
            for num in range(D):
  
                get2D = curGT_temp[:,:,num]
                bboxes[num, :] = fun_get_box(get2D, self.bbox_shift)
            bboxes = torch.tensor(bboxes).float() 



        sample = {'curMR': curMR, 'preMR': preMR, 'preGT': preGT, "curGT": curGT, "bboxes": bboxes,"caseID": caseID}
        # if self.transform:
        #     sample = self.transform(sample)

        return sample
      

###########################
def get_all_bounding_boxes_with_ccl(mask, bbox_shift=0):
    """
    Extract bounding boxes for all connected components of each labeled object in a segmentation mask.

    Args:
        mask (torch.Tensor): A (H, W) tensor where each unique value represents a cell label.

    Returns:
        dict: {label: [(x_min, y_min, x_max, y_max), ...]}
    """
    if torch.torch.is_tensor(mask):
        mask_np = mask.cpu().numpy()  # Convert to NumPy for processing
    else:
        mask_np = mask
    unique_labels = np.unique(mask_np)
    unique_labels = unique_labels[unique_labels != 0]  # Remove background (label 0)


    sel_bbox = np.ones((4))* (-100)#.to(mask_256.device) * (-100), save the mask
    if len(unique_labels) < 1: # empty, there is no box in the masl
        sel_bbox = np.ones((4))* (-100)
        return sel_bbox

    bboxes = {}
    for label_value in unique_labels:
        binary_mask = (mask_np == label_value).astype(np.int32)  # Create a binary mask for the current label
        
        # Apply connected component labeling
        labeled_mask, num_components = label(binary_mask)

        # Extract bounding boxes for each connected component
        slices = find_objects(labeled_mask)
        boxes = []
        
        for s in slices:
            if s is not None:
                y_min, x_min = s[0].start, s[1].start
                y_max, x_max = s[0].stop - 1, s[1].stop - 1
                boxes.append((x_min, y_min, x_max, y_max))

        bboxes[label_value] = boxes
    ## randomly select one bboxes and save as tensor
    if len(bboxes[unique_labels[0]]) > 1:
        ## get and rand int
        NUM = torch.randint(0, len(bboxes[unique_labels[0]]), (1,)).numpy()[0]
        sel_bbox = np.array(bboxes[unique_labels[0]][NUM])
    else:
        sel_bbox = np.array(bboxes[unique_labels[0]][0])
    

    return sel_bbox

def get_all_bounding_boxes_with_ccl_box_fix(mask, bbox_shift=0):
    """
    Extract bounding boxes for all connected components of each labeled object in a segmentation mask.

    Args:
        mask (torch.Tensor): A (H, W) tensor where each unique value represents a cell label.
        fix bbox_shift

    Returns:
        dict: {label: [(x_min, y_min, x_max, y_max), ...]}
    """
    if torch.torch.is_tensor(mask):
        mask_np = mask.cpu().numpy()  # Convert to NumPy for processing
    else:
        mask_np = mask
    unique_labels = np.unique(mask_np)
    unique_labels = unique_labels[unique_labels != 0]  # Remove background (label 0)


    sel_bbox = np.ones((4))* (-100)#.to(mask_256.device) * (-100), save the mask
    if len(unique_labels) < 1: # empty, there is no box in the masl
        sel_bbox = np.ones((4))* (-100)
        return sel_bbox

    bboxes = {}
    for label_value in unique_labels:
        binary_mask = (mask_np == label_value).astype(np.int32)  # Create a binary mask for the current label
        
        # Apply connected component labeling
        labeled_mask, num_components = label(binary_mask)

        # Extract bounding boxes for each connected component
        slices = find_objects(labeled_mask)
        boxes = []
        
        for s in slices:
            if s is not None:
                y_min, x_min = max(0, s[0].start - bbox_shift), max(0, s[1].start - bbox_shift)
                y_max, x_max = max(0, s[0].stop - 1 + bbox_shift), max(0, s[1].stop - 1 + bbox_shift)
                boxes.append((x_min, y_min, x_max, y_max))

        bboxes[label_value] = boxes
    ## randomly select one bboxes and save as tensor
    ## !!!notice: you should save all bbox in the testing stage
    if len(bboxes[unique_labels[0]]) > 1:
        ## get and rand int
        NUM = torch.randint(0, len(bboxes[unique_labels[0]]), (1,)).numpy()[0]
        sel_bbox = np.array(bboxes[unique_labels[0]][NUM])
    else:
        sel_bbox = np.array(bboxes[unique_labels[0]][0])
    

    return sel_bbox

def get_all_bounding_boxes_with_ccl_box_shift(mask, bbox_shift=5):
    """
    Extract bounding boxes for all connected components of each labeled object in a segmentation mask.

    Args:
        mask (torch.Tensor): A (H, W) tensor where each unique value represents a cell label.
        bbox_shift = enlarge the bbox

    Returns:
        dict: {label: [(x_min, y_min, x_max, y_max), ...]}
    """
    if torch.torch.is_tensor(mask):
        mask_np = mask.cpu().numpy()  # Convert to NumPy for processing
    else:
        mask_np = mask
    unique_labels = np.unique(mask_np)
    unique_labels = unique_labels[unique_labels != 0]  # Remove background (label 0)


    sel_bbox = np.ones((4))* (-100)#.to(mask_256.device) * (-100), save the mask
    if len(unique_labels) < 1: # empty, there is no box in the masl
        sel_bbox = np.ones((4))* (-100)
        return sel_bbox

    bboxes = {}
    for label_value in unique_labels:
        binary_mask = (mask_np == label_value).astype(np.int32)  # Create a binary mask for the current label
        
        # Apply connected component labeling
        labeled_mask, num_components = label(binary_mask)

        # Extract bounding boxes for each connected component
        slices = find_objects(labeled_mask)
        boxes = []
        
        for s in slices:
            if s is not None:
                y_min = max(0,  s[0].start - random.randint(0, bbox_shift))
                x_min = max(0,  s[1].start - random.randint(0, bbox_shift))
                y_max = max(0,  s[0].stop - 1 + random.randint(0, bbox_shift))
                x_max = max(0,  s[1].stop - 1 + random.randint(0, bbox_shift))

                boxes.append((x_min, y_min, x_max, y_max))

        bboxes[label_value] = boxes
    ## randomly select one bboxes and save as tensor
    if len(bboxes[unique_labels[0]]) > 1:
        ## get and rand int
        NUM = torch.randint(0, len(bboxes[unique_labels[0]]), (1,)).numpy()[0]
        sel_bbox = np.array(bboxes[unique_labels[0]][NUM])
    else:
        sel_bbox = np.array(bboxes[unique_labels[0]][0])
    

    return sel_bbox

def get_all_bounding_boxes_with_ccl_box_shift_pos_neg(mask, bbox_shift=5):
    """
    Extract bounding boxes for all connected components of each labeled object in a segmentation mask.

    Args:
        mask (torch.Tensor): A (H, W) tensor where each unique value represents a cell label.
        bbox_shift = enlarge or en-small the bbox

    Returns:
        dict: {label: [(x_min, y_min, x_max, y_max), ...]}
    """
    if torch.torch.is_tensor(mask):
        mask_np = mask.cpu().numpy()  # Convert to NumPy for processing
    else:
        mask_np = mask
    unique_labels = np.unique(mask_np)
    unique_labels = unique_labels[unique_labels != 0]  # Remove background (label 0)


    sel_bbox = np.ones((4))* (-100)#.to(mask_256.device) * (-100), save the mask
    if len(unique_labels) < 1: # empty, there is no box in the masl
        sel_bbox = np.ones((4))* (-100)
        return sel_bbox

    bboxes = {}
    for label_value in unique_labels:
        binary_mask = (mask_np == label_value).astype(np.int32)  # Create a binary mask for the current label
        
        # Apply connected component labeling
        labeled_mask, num_components = label(binary_mask)

        # Extract bounding boxes for each connected component
        slices = find_objects(labeled_mask)
        boxes = []
        
        for s in slices:
            if s is not None:
                y_min = max(0, s[0].start + random.choice([1, -1]) * random.randint(0, bbox_shift))
                x_min = max(0, s[1].start + random.choice([1, -1]) * random.randint(0, bbox_shift))
                y_max = max(0, s[0].stop - 1 + random.choice([1, -1]) * random.randint(0, bbox_shift))
                x_max = max(0, s[1].stop - 1 + random.choice([1, -1]) * random.randint(0, bbox_shift))

                boxes.append((x_min, y_min, x_max, y_max))

        bboxes[label_value] = boxes
    ## randomly select one bboxes and save as tensor
    if len(bboxes[unique_labels[0]]) > 1:
        ## get and rand int
        NUM = torch.randint(0, len(bboxes[unique_labels[0]]), (1,)).numpy()[0]
        sel_bbox = np.array(bboxes[unique_labels[0]][NUM])
    else:
        sel_bbox = np.array(bboxes[unique_labels[0]][0])
    

    return sel_bbox

#################################: transform
class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        curMR, preMR, preGT, curGT, caseID = sample['curMR'], sample['preMR'], sample['preGT'], sample['curGT'], sample['caseID']

        # pad the sample if necessary
        if curGT.shape[0] <= self.output_size[0] or curGT.shape[1] <= self.output_size[1] or curGT.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - curGT.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - curGT.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - curGT.shape[2]) // 2 + 3, 0)
            curMR = np.pad(curMR, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            curGT = np.pad(curGT, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            preMR = np.pad(preMR, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            preGT = np.pad(preGT, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = curMR.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        curGT = curGT[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        curMR = curMR[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        preGT = preGT[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        preMR = preMR[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'curMR': curMR, 'preMR': preMR, 'preGT': preGT, "curGT": curGT, "caseID": caseID}


class RandomCrop(object):
    """
    Crop randomly the curMR in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        curMR, preMR, preGT, curGT, caseID = sample['curMR'], sample['preMR'], sample['preGT'], sample['curGT'], sample['caseID']

        # pad the sample if necessary
        if curGT.shape[0] <= self.output_size[0] or curGT.shape[1] <= self.output_size[1] or curGT.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - curGT.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - curGT.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - curGT.shape[2]) // 2 + 3, 0)
            curMR = np.pad(curMR, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            curGT = np.pad(curGT, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            preMR = np.pad(preMR, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            preGT = np.pad(preGT, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = curMR.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        curGT = curGT[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        curMR = curMR[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        preGT = preGT[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        preMR = preMR[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'curMR': curMR, 'preMR': preMR, 'preGT': preGT, "curGT": curGT, "caseID": caseID}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        curMR, preMR, preGT, curGT, caseID = sample['curMR'], sample['preMR'], sample['preGT'], sample['curGT'], sample['caseID']
        k = np.random.randint(0, 4)
        curMR = np.rot90(curMR, k)
        curGT = np.rot90(curGT, k)
        axis = np.random.randint(0, 2)
        curMR = np.flip(curMR, axis=axis).copy()
        curGT = np.flip(curGT, axis=axis).copy()

        preMR = np.rot90(preMR, k)
        preGT = np.rot90(preGT, k)
        axis = np.random.randint(0, 2)
        preMR = np.flip(preMR, axis=axis).copy()
        preGT = np.flip(preGT, axis=axis).copy()


        return {'curMR': curMR, 'preMR': preMR, 'preGT': preGT, "curGT": curGT, "caseID": caseID}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        curMR, preMR, preGT, curGT, caseID = sample['curMR'], sample['preMR'], sample['preGT'], sample['curGT'], sample['caseID']
        noise = np.clip(self.sigma * np.random.randn(curMR.shape[0], curMR.shape[1], curMR.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        curMR = curMR + noise
        preMR = preMR + noise
        return {'curMR': curMR, 'preMR': preMR, 'preGT': preGT, "curGT": curGT, "caseID": caseID}


class CreateOnehotcurGT(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        curMR, preMR, preGT, curGT, caseID = sample['curMR'], sample['preMR'], sample['preGT'], sample['curGT'], sample['caseID']
        onehot_curGT = np.zeros((self.num_classes, curGT.shape[0], curGT.shape[1], curGT.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_curGT[i, :, :, :] = (curGT == i).astype(np.float32)
        return {'curMR': curMR, 'curGT': curGT,'onehot_curGT':onehot_curGT}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        curMR = sample['curMR']
        curMR = curMR.reshape(1, curMR.shape[0], curMR.shape[1], curMR.shape[2]).astype(np.float32)
        preMR = sample['preMR']
        preMR = preMR.reshape(1, preMR.shape[0], preMR.shape[1], preMR.shape[2]).astype(np.float32)
        preGT = sample['preGT']
        preGT = preGT.reshape(1, preGT.shape[0], preGT.shape[1], preGT.shape[2]).astype(np.float32)

        if 'onehot_curGT' in sample:
            return {'curMR': torch.from_numpy(curMR), 'curGT': torch.from_numpy(sample['curGT']).long(),
                    'onehot_curGT': torch.from_numpy(sample['onehot_curGT']).long(),
                    'caseID': sample['caseID']}
        else:
            return {'curMR': torch.from_numpy(curMR), 'preMR': torch.from_numpy(preMR), 
                     'preGT': torch.from_numpy(preGT).long(), 'curGT': torch.from_numpy(sample['curGT']).long(),
                     'caseID': sample['caseID']}

# def preprocess_mri_and_mask(mri_tensor, mask_tensor, target_depth=128, mode='train'):
#     """
#     Preprocess MRI and mask tensors together.

#     Parameters:
#     - mri_tensor (torch.Tensor): MRI tensor of shape [1, H, W, D]
#     - mask_tensor (torch.Tensor): Mask tensor of shape [1, H, W, D]
#     - target_depth (int): Desired depth for training mode
#     - mode (str): One of 'train', 'val', or 'test'

#     Returns:
#     - Tuple[torch.Tensor, torch.Tensor]: Processed (mri_tensor, mask_tensor), both [1, H, W, D]
#     """

#     def _reshape_for_interp(t):  # [1, H, W, D] → [1, 1, D, H, W]
#         return t.permute(0, 3, 1, 2).unsqueeze(0)

#     def _reshape_back(t):  # [1, 1, D, H, W] → [1, H, W, D]
#         return t.squeeze(0).permute(0, 2, 3, 1)

#     # Reshape for interpolation
#     mri_tensor = torch.from_numpy(mri_tensor).float()  # Ensure tensor is float
#     mask_tensor = torch.from_numpy(mask_tensor).float()  # Ensure tensor is float
#     mri = _reshape_for_interp(mri_tensor)
#     mask = _reshape_for_interp(mask_tensor)

#     if mode == 'train':
#         mri = F.interpolate(
#             mri, size=(target_depth, mri.shape[3], mri.shape[4]),
#             mode='trilinear', align_corners=False
#         )
#         mask = F.interpolate(
#             mask, size=(target_depth, mask.shape[3], mask.shape[4]),
#             mode='nearest'
#         )

#     # Reshape back to original format
#     mri = _reshape_back(mri)
#     mask = _reshape_back(mask)

#     return mri, mask


def preprocess_mri_and_mask(mri_tensor, mask_tensor, target_depth=128, target_hw=(512, 512), mode='train'):
    """
    Preprocess MRI and mask tensors:
    - Resize H and W to 512 if needed.
    - Crop or pad D to 128 (same for both MRI and mask).
    
    Args:
        mri_tensor: Tensor [1, H, W, D]
        mask_tensor: Tensor [1, H, W, D]
        target_depth: Target D size (default: 128)
        target_hw: Target (H, W) size (default: 512x512)
        mode: 'train' (apply crop) or 'val'/'test' (only pad D if needed)
        
    Returns:
        mri_tensor, mask_tensor: Processed tensors [1, 512, 512, 128]
    """
    # # Ensure contiguous
    # mri_tensor = mri_tensor.contiguous()
    # mask_tensor = mask_tensor.contiguous()
    
    # # Reshape for interpolation
    # mri_tensor = torch.from_numpy(mri_tensor).float()  # Ensure tensor is float : 1HWD
    # mask_tensor = torch.from_numpy(mask_tensor).float()  # Ensure tensor is float : 1HWD

    _, H, W, D = mri_tensor.shape
    target_H, target_W = target_hw

    # Step 1: Resize H and W if needed
    if H != target_H or W != target_W:
        mri_tensor = mri_tensor.permute(0, 3, 1, 2).unsqueeze(0)  # [1, 1, D, H, W]
        mask_tensor = mask_tensor.permute(0, 3, 1, 2).unsqueeze(0)

        mri_tensor = F.interpolate(mri_tensor, size=(D, target_H, target_W), mode='trilinear', align_corners=False)
        mask_tensor = F.interpolate(mask_tensor, size=(D, target_H, target_W), mode='nearest')

        mri_tensor = mri_tensor.squeeze(0).permute(0, 2, 3, 1)  # [1, H, W, D]
        mask_tensor = mask_tensor.squeeze(0).permute(0, 2, 3, 1)

    # Step 2: Adjust depth D
    if D > target_depth and mode == 'train':
        # Random crop along D
        start_idx = random.randint(0, D - target_depth)
        end_idx = start_idx + target_depth
        mri_tensor = mri_tensor[:, :, :, start_idx:end_idx]
        mask_tensor = mask_tensor[:, :, :, start_idx:end_idx]

    elif D < target_depth:
        # Pad zeros at the end of D axis
        pad_D = target_depth - D
        mri_tensor = F.pad(mri_tensor, (0, pad_D), mode='constant', value=0)
        mask_tensor = F.pad(mask_tensor, (0, pad_D), mode='constant', value=0)

    # Final shape should be [1, 512, 512, 128]
    return mri_tensor, mask_tensor


if __name__=="__main__":
    mask = torch.zeros((100, 100), dtype=torch.uint8)
    mask[30:60, 40:80] = 1  # Example object 1
    mask[10:20, 10:30] = 1  # Example object 2

    # bboxes = get_all_bounding_boxes_with_ccl(mask)
    bboxes = get_all_bounding_boxes_with_ccl_box_shift(mask, bbox_shift=5)
    print(bboxes)  # [(40, 30, 79, 59), (10, 10, 29, 19)]