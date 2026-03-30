import os
from torch.utils.data import DataLoader
from data_load.busi import BUSI, BUSI_AutoPmpt



## load train, validation or test dataset
def load_data_busi(root_path, transform_policy=None, batch_size=1,  split="test", bbox_shift=None, fix_bbox=True, 
                       pos_neg=True, modality_type="Ultrasound", use_v2=False):
    """
    root_path: the project root path
    transform_policy: transform method
    batch_size: batch size for training
    split: train, val or test
    """
    ## set data path, data transform method
    if modality_type == "Ultrasound":
        sample_dataset = BUSI(root=root_path, split=split, transform_img=transform_policy,)

    if split == "train":
        print(f"total samples: {len(sample_dataset)}")
        sample_loader = DataLoader(sample_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,) # training dataset
        print(f"{len(sample_loader)} iterations per epoch")
    else:
        sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,) # validation or testing dataset
        print(f"total {split} samples: {len(sample_dataset)}")
    return sample_loader


def load_data_busi_auto_pmpt(root_path, transform_policy=None, batch_size=1,  split="test", modality_type="Ultrasound"):
    """
    root_path: the project root path
    transform_policy: transform method
    batch_size: batch size for training
    split: train, val or test
    """
    ## set data path, data transform method
    if modality_type == "Ultrasound":
        sample_dataset = BUSI_AutoPmpt(root=root_path, split=split, transform_img=transform_policy,)

    if split == "train":
        print(f"total samples: {len(sample_dataset)}")
        sample_loader = DataLoader(sample_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,) # training dataset
        print(f"{len(sample_loader)} iterations per epoch")
    else:
        sample_loader = DataLoader(sample_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,) # validation or testing dataset
        print(f"total {split} samples: {len(sample_dataset)}")
    return sample_loader