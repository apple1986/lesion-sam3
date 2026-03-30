import os
from torch.utils.data import DataLoader
from data_load.msd_kidney import KITS2019



## load train, validation or test dataset
def load_data_msd_kidney(root_path, transform_policy=None, batch_size=1,  split="test", bbox_shift=None, fix_bbox=True, 
                       pos_neg=True, modality_type="CT", use_v2=True):
    """
    root_path: the project root path
    transform_policy: transform method
    batch_size: batch size for training
    split: train, val or test
    """
    ## set data path, data transform method
    if modality_type == "CT":
        sample_dataset = KITS2019(base_dir=root_path, split=split, transform=transform_policy,
                             bbox_shift=bbox_shift, fix_bbox=fix_bbox, 
                             pos_neg=pos_neg, modality_type=modality_type,
                             use_v2=use_v2)
    if split == "train":
        print(f"total samples: {len(sample_dataset)}")
        sample_loader = DataLoader(sample_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,) # training dataset
        print(f"{len(sample_loader)} iterations per epoch")
    else:
        sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True,) # validation or testing dataset
        print(f"total {split} samples: {len(sample_dataset)}")
    return sample_loader