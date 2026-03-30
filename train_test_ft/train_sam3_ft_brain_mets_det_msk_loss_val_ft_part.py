# ap: freeze the visual and text encoders, ignore the semantic mask loss from pixel decoder
import sys
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2")
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2")

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from data_load.brats_builder import load_data_brats_mets
from data_load.sam3_data import MRISliceDataset, convert_to_batched_datapoint
import shutil
import matplotlib.pyplot as plt
from ap_sam3.eval_sam3 import evaluate_sam3

# ===== SAM3 =====
import sam3
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
from sam3 import build_sam3_image_model
import torch.distributed as dist

if not dist.is_initialized():
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method="tcp://127.0.0.1:29503",
        world_size=1,
        rank=0,
    )
from hydra.utils import instantiate

# =========================
# Build SAM3
# =========================
def load_sam3(device):
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        eval_mode=False,
        enable_inst_interactivity=False,
        device=device,
    )
    return model

# ------------------------------------------------------------
# Total Loss: detection and mask loss
# ------------------------------------------------------------
## semantic seg loss
loss_fn_semantic_seg = {
    "_target_": "sam3.train.loss.loss_fns.SemanticSegCriterion",
    "presence_head": True,
    "presence_loss": False,   # Change
    "focal": True,
    "focal_alpha": 0.6,
    "focal_gamma": 2.0,
    "downsample": False,
    "weight_dict": {
        "loss_semantic_seg": 20.0, #ori:20.0, 40,
        "loss_semantic_dice": 10.0, #ori: 30.0, 60
        "loss_semantic_presence": 1.0,        
    },
}

matcher_conf = {
    "_target_": "sam3.train.matcher.BinaryHungarianMatcherV2",
    "focal": True, # use focal loss to do classification
    "cost_class": 4.0, # weight of classification loss # ↑ prioritize correct object matching
    "cost_bbox": 1.0, # L1 bbox loss weight ↓ bbox should not dominate
    "cost_giou": 2.0, # IoU loss weight
    "alpha": 0.25, # focal α
    "gamma": 2.0, # focal γ
    "stable": False,
}
loss_conf = {
    "_target_": "sam3.train.loss.sam3_loss.Sam3LossWrapper",

    "matcher": matcher_conf,
    # ---------- One-to-Many Matching ----------
    "o2m_weight": 0.1, # One-to-Many Matching, weight of the total loss
    "o2m_matcher": {
        "_target_": "sam3.train.matcher.BinaryOneToManyMatcher",
        "alpha": 0.5, # postive and negative rate, Balance between classification & localization in O2M matching
        "threshold": 0.4, # IoU for threshold, Minimum confidence to participate in O2M
        "topk": 2, # each GT for 4 queries: Max number of queries per GT
    },
    "use_o2m_matcher_on_o2m_aux": False,
    
    # ---------- Object Finding + Segmentation ----------
    "loss_fns_find": [ # Object-Finding Losses, Increase only if localization is poor, used when GT boxes exist
         # --- Bounding box regression ---
        {
            "_target_": "sam3.train.loss.loss_fns.Boxes",
            "weight_dict": {
                "loss_bbox": 0.005, #,5.0,
                "loss_giou": 0.05, # 0.1
            },
        },
        # --- Classification + presence ---
        {
            "_target_": "sam3.train.loss.loss_fns.IABCEMdetr", # Classification + Presence Loss: core loss for concept-based segmentation
            "weak_loss": False,
            "weight_dict": {
                "loss_ce": 2.0, # Query-level object classification
                "presence_loss": 5.0, # Image-level presence supervision: classification of presence token, 0.5
            },
            "pos_weight": 5.0, # postive sample weight: Handles foreground/background imbalance
            "alpha": 0.25, # Focal loss parameters
            "gamma": 2.0, # Focal loss parameters
            "use_presence": True, # True for weak labels or text prompts
            "pos_focal": False,
            "pad_n_queries": 200,
            "pad_scale_pos": 1.0,
        },
        # --- Mask loss (THIS IS THE KEY ADDITION) ---: detection detector used
        {
            "_target_": "sam3.train.loss.loss_fns.Masks",
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
            "weight_dict": {
                "loss_mask": 20.0,   # pixel-wise focal/BCE
                "loss_dice": 5.0,    # overlap quality
            },
            "compute_aux": False,
        },        
    ],

     # ---------- Semantic Segmentation ----------
    # "loss_fn_semantic_seg": None,
    "loss_fn_semantic_seg": None, #loss_fn_semantic_seg,

    "scale_by_find_batch_size": True,
}
# ------------------------------------------------------------
# Plot loss
# ------------------------------------------------------------
def plot_loss(steps, losses, ckpt_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, linewidth=1)
    plt.xlabel("Optimizer Step")
    plt.ylabel("Total Loss")
    plt.title("SAM3 Training Loss")
    plt.grid(True)

    plt.savefig(os.path.join(ckpt_dir, "loss_curve.png"))
    plt.close()   # SAFE here, does NOT affect saved file


# ------------------------------------------------------------
# Training config
# ------------------------------------------------------------
# set path
root_path = "/home/gxu/proj1/lesionSeg/sam3_brain"
ckpt_dir = os.path.join(root_path, "checkpoint_exp/brats_mets_ft_part_det_loss_seg_det_conf")
os.makedirs(ckpt_dir, exist_ok=True)  
# copy the current script
script_path = os.path.join(ckpt_dir, os.path.basename(__file__))
shutil.copyfile(__file__, script_path)
loss_fn = instantiate(loss_conf, _convert_="all")
device = "cuda:1"
batch_size = 1
num_epochs = 10
learning_rate = 1e-6
image_size = 1008
ACCUM_STEPS = 10   # update after 10 images
scaler = torch.amp.GradScaler(device)
loss_history = []
step_history = []

# =========================
# Training
# =========================
def main():
    # load training data
    base_loader = load_data_brats_mets(root_path, transform_policy=None, batch_size=1, split="train")
    train_dataset = MRISliceDataset(base_loader)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    )
    print(f"✅ Total training slices: {len(train_dataset)}")

    # load validation data
    base_loader = load_data_brats_mets(root_path, transform_policy=None, batch_size=1, split="val")
    val_dataset = MRISliceDataset(base_loader)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )
    print(f"✅ Total validation slices: {len(val_dataset)}")

    # load model and setting parameters
    model = load_sam3(device)
    for p in model.backbone.vision_backbone.parameters():
        p.requires_grad = False
    for p in model.backbone.language_backbone.parameters():
        p.requires_grad = False        

    optimizer = torch.optim.AdamW(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=learning_rate,
                        weight_decay=1e-4,)

    # model training
    model.train()
    optimizer.zero_grad()
    global_step = 0
    epoch_loss = 0.0
    accum_count = 0
    accum_loss = 0
    dice_val_pre = -1.0 # previous dice in validation dataset
    for epoch in range(num_epochs):
        epoch_loss = 0
        # # evaluation on val dataset
        # metrics = evaluate_sam3(model, val_loader, device)
        # # save it as a txt file
        # with open(os.path.join(ckpt_dir, "eval_record_epoch.txt"), "a") as f:
        #     write_content = f'Epoch: [{epoch}/{num_epochs}],  Dice: {metrics['dice']:.4f},  IoU: {metrics['iou']:.4f}\n'
        #     f.write(write_content)  

        # training
        model.train()
        for imgs, masks in tqdm(train_loader):
            imgs = imgs.to(device)        # B,3,1008,1008
            masks = masks.to(device)      # B,1008,1008
          
            # Forward each image independently
            for i in range(imgs.shape[0]):
                img = imgs[i].unsqueeze(0)          # 1,3,1008,1008
                gt  = masks[i].unsqueeze(0).unsqueeze(0)  # 1,1,1008,1008

                # Directly forward through model (training mode)
                img_bd = convert_to_batched_datapoint(img, gt)
                with torch.amp.autocast(device, dtype=torch.float16):
                    out = model(img_bd) # torch.Size([1, 3, 1008, 1008]) 
                    # Convert GT (CRITICAL)
                    find_targets = [model.back_convert(t) for t in img_bd.find_targets]               
                    loss = loss_fn(out, find_targets)["core_loss"] / ACCUM_STEPS
                # scale loss
                scaler.scale(loss).backward()

                iter_loss = loss.item()
                accum_loss = accum_loss + iter_loss
                epoch_loss = epoch_loss + iter_loss
                accum_count = accum_count + 1 # update account

                if accum_count == ACCUM_STEPS:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step = global_step + 1
        
                    print(f"[Step {global_step}] Loss: {accum_loss:.4f}")
                    with open(os.path.join(ckpt_dir, "train_loss_record_iter.txt"), "a") as f:
                            write_content = f'Epoch: [{epoch+1}/{num_epochs}]\tStep: [{global_step}]\tLoss: {accum_loss:.4f}\n'
                            f.write(write_content)

                    loss_history.append(accum_loss)
                    step_history.append(global_step)
                    # update count
                    accum_loss = 0.0
                    accum_count = 0

                    # save every 500 optimizer steps
                    if global_step % 500 == 0:
                        # Save checkpoint
                        torch.save(
                            model.state_dict(),
                            os.path.join(ckpt_dir, f"sam3_mri_global_step.pth")
                        )

        print(f"✅ Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss:.4f}")
        # save it as a txt file
        with open(os.path.join(ckpt_dir, "train_loss_record_epoch.txt"), "a") as f:
            write_content = f'Epoch: [{epoch+1}/{num_epochs}]\tLoss: {epoch_loss:.4f}\n'
            f.write(write_content)        

        # evaluation on val dataset
        metrics = evaluate_sam3(model, val_loader, device)
        dice_val_cur = metrics['dice']
        print(
            f"[Epoch {epoch + 1}] "
            f"Dice: {metrics['dice']:.4f}, "
            f"IoU: {metrics['iou']:.4f}"
        )
        # save it as a txt file
        with open(os.path.join(ckpt_dir, "eval_record_epoch.txt"), "a") as f:
            write_content = f'Epoch: [{epoch+1}/{num_epochs}],  Dice: {metrics['dice']:.4f},  IoU: {metrics['iou']:.4f}\n'
            f.write(write_content)              

        if dice_val_cur > dice_val_pre:
            dice_val_pre = dice_val_cur
            # save model
            torch.save(
                        model.state_dict(),
                        os.path.join(ckpt_dir, f"sam3_mri_best_val.pth")
                    )
        model.train()
       
        # Save checkpoint in each epoch
        torch.save(
            model.state_dict(),
            os.path.join(ckpt_dir, f"sam3_mri_epoch_{num_epochs}.pth")
        )
    # plot final loss curve
    plot_loss(step_history, loss_history, ckpt_dir)

if __name__ == "__main__":
    main()
