import os
from torch.utils.data import DataLoader
from data_load.ustwlits import UTSWData
from data_load.ustwlits_sam import UTSWLiver_SAM
from data_load.ustw_cns import UTSW_CNS



## load train, validation or test dataset
def load_data(root_path, transform_policy=None, batch_size=1,  split="train", tv_type="itv", use_reg=False, register_method="Syn"):
    """
    root_path: the project root path
    transform_policy: transform method
    batch_size: batch size for training
    split: train, val or test
    split: brain_all, dataset: brain_cns: ctv
    split: abd_all, dataset: labdominal_itv
    split: liver_all, dataset: liver_itv

    """
    ## set data path, data transform method
    sample_dataset = UTSWData(base_dir=root_path, split=split, tv_type=tv_type,
                            transform=transform_policy)
    if split == "train":
        print(f"total samples: {len(sample_dataset)}")
        sample_loader = DataLoader(sample_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,) # training dataset
        print(f"{len(sample_loader)} iterations per epoch")
    else:
        sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True,) # validation or testing dataset
        print(f"total {split} samples: {len(sample_dataset)}")
    return sample_loader

def load_data_sam(root_path, transform_policy, batch_size, split="train", bbox_shift=None, fix_bbox=True, pos_neg=True):
    """
    root_path: the project root path
    transform_policy: transform method
    batch_size: batch size for training
    split: train, val or test
    """
    ## set data path, data transform method
    sample_dataset = UTSWLiver_SAM(base_dir=root_path, split=split, 
                            transform=transform_policy, bbox_shift=bbox_shift, fix_bbox=fix_bbox, pos_neg=pos_neg)
    if split == "train":
        print(f"total samples: {len(sample_dataset)}")
        sample_loader = DataLoader(sample_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,) # training dataset
        print(f"{len(sample_loader)} iterations per epoch")
    else:
        sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True,) # validation or testing dataset
        print(f"total {split} samples: {len(sample_dataset)}")
    return sample_loader

def load_data_cns(root_path, transform_policy, batch_size, split="train", bbox_shift=None, fix_bbox=True, pos_neg=True):
    """
    root_path: the project root path
    transform_policy: transform method
    batch_size: batch size for training
    split: train, val or test
    """
    ## set data path, data transform method
    sample_dataset = UTSW_CNS(base_dir=root_path, split=split, 
                            transform=transform_policy, bbox_shift=bbox_shift, fix_bbox=fix_bbox, pos_neg=pos_neg)
    if split == "train":
        print(f"total samples: {len(sample_dataset)}")
        sample_loader = DataLoader(sample_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,) # training dataset
        print(f"{len(sample_loader)} iterations per epoch")
    else:
        sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True,) # validation or testing dataset
        print(f"total {split} samples: {len(sample_dataset)}")
    return sample_loader