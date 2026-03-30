import os
from torch.utils.data import DataLoader
from data_load.isic2018 import ISIC2018



## load train, validation or test dataset
def load_data_isic2018(root_path, transform_policy=None, batch_size=1,  split="test", bbox_shift=None, fix_bbox=True, 
                       pos_neg=True, modality_type="Dermoscopy", use_v2=False):
    """
    root_path: the project root path
    transform_policy: transform method
    batch_size: batch size for training
    split: train, val or test
    """
    ## set data path, data transform method
    if modality_type == "Dermoscopy":
        sample_dataset = ISIC2018(root=root_path, split=split, transform_img=transform_policy,)



    if split == "train":
        print(f"total samples: {len(sample_dataset)}")
        sample_loader = DataLoader(sample_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,) # training dataset
        print(f"{len(sample_loader)} iterations per epoch")
    else:
        sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,) # validation or testing dataset
        print(f"total {split} samples: {len(sample_dataset)}")
    return sample_loader