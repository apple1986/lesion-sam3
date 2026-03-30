# ap: freeze visual & text encoders, use Hydra config for optimizer, scheduler, training, different lrs for different parts
import sys, os, shutil
import numpy as np
from random import shuffle
from sympy import use
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torch.nn import functional as F

sys.path.extend([
    "/home/gxu/proj1/lesionSeg/utswlits3d_v2",
    "/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2"
])

from data_load.brats_builder import load_data_brats_mets
from data_load.all_data_loader import all_datasets
from data_load.sam3_data import MRISliceDataset_v2, convert_to_batched_datapoint
from ap_sam3.eval_sam3 import evaluate_sam3

import sam3
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
from sam3 import build_sam3_image_model
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sam3.train.utils.train_utils import register_omegaconf_resolvers
# from sam3.train.optim.optimizer import construct_optimizer

# ------------------------
# Load Hydra config
# ------------------------
cfg = OmegaConf.load("train_configs/mets_ft_adp.yaml")

if not dist.is_initialized():
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method="tcp://127.0.0.1:29507",
        world_size=1,
        rank=0,
    )

# ------------------------
# Utility: plot loss curve
# ------------------------
def plot_loss(steps, losses, ckpt_dir):
    plt.figure(figsize=(8,5))
    plt.plot(steps, losses, linewidth=1)
    plt.xlabel("Optimizer Step")
    plt.ylabel("Total Loss")
    plt.title("SAM3 Training Loss")
    plt.grid(True)
    plt.savefig(os.path.join(ckpt_dir, "loss_curve.png"))
    plt.close()

# ------------------------
# Build SAM3 model
# ------------------------
def load_sam3(device, use_adapter=False):
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        eval_mode=False,
        enable_inst_interactivity=False,
        device=device,
    )
    if use_adapter:
        from ap_knife.adapters import insert_mlp_adapters_sam3
        insert_mlp_adapters_sam3(model, bottleneck=64, start_block=20)
        # freeze entire model
        for p in model.parameters():
            p.requires_grad = False

        # unfreeze adapters only
        for blk in model.backbone.vision_backbone.trunk.blocks:
            if hasattr(blk, "adapter"):
                for p in blk.adapter.parameters():
                    p.requires_grad = True
    model = model.to(device)
    return model

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable




# class UniversalSliceDataset(Dataset):
#     def __init__(
#         self,
#         dataloader,
#         image_keys=None,     # e.g. ("curMR",), ("dwi",), ("curCT","curPT")
#         mask_key="curGT",
#         min_fg_pixels=10,
#     ):
#         # self.dataloader = list(dataloader)  # keep iterable as list
#         self.dataloader = dataloader
#         self.image_keys = image_keys
#         self.mask_key = mask_key
#         self.min_fg_pixels = min_fg_pixels
#         self.slice_index = []  # store (batch_idx, slice_idx) for fast access

#         # build index lazily
#         print(f"⏳ Indexing slices (lazy, no data copy yet)...")
#         for b_idx, batch in enumerate(self.dataloader):
#             if isinstance(batch, dict):
#                 # find first available image key
#                 img_key = None
#                 for k in image_keys:
#                     if k in batch:
#                         img_key = k
#                         break
#                 if img_key is None:
#                     raise KeyError(f"No image key found in batch {b_idx}")
                
#                 vol = torch.cat([batch[k] for k in image_keys], dim=1)  # (B,C,H,W,D)
#                 gt  = batch[mask_key]
#                 for b in range(vol.shape[0]):
#                     D = vol[b].shape[-1]
#                     for z in range(D):
#                         mask_sum = gt[b, 0, :, :, z].sum()
#                         if mask_sum >= min_fg_pixels:
#                             self.slice_index.append((b_idx, b, z))
#             else:
#                 # tuple-based 2D
#                 img, mask = batch[:2]
#                 if mask.sum() >= min_fg_pixels:
#                     self.slice_index.append((b_idx, 0, 0))  # dummy slice idx

#         print(f"# Total slices indexed: {len(self.slice_index)}")

#         self.img_transform = v2.Compose([
#             v2.ToDtype(torch.uint8, scale=True),
#             v2.Resize((1008, 1008)),
#             v2.ToDtype(torch.float32, scale=True),
#             v2.Normalize(mean=[0.5]*3, std=[0.5]*3),
#         ])

#     def __len__(self):
#         return len(self.slice_index)

#     def __getitem__(self, idx):
#         b_idx, b, z = self.slice_index[idx]
#         batch = self.dataloader[b_idx]

#         # ---------- load image ----------
#         if isinstance(batch, dict):
#             imgs = [batch[k][b, :, :, :, z] for k in self.image_keys]
#             img = torch.cat(imgs, dim=0)
#             mask = batch[self.mask_key][b, 0, :, :, z]

#             # PET/CT -> add zeros as 3rd channel
#             if len(self.image_keys) == 2:
#                 zero = torch.zeros_like(img[:1])
#                 img = torch.cat([img[0:1], img[1:2], zero], dim=0)

#         else:
#             img, mask = batch[:2]

#         # ---- Normalize image to (C, H, W) ----
#         if img.ndim == 2:
#             # (H,W) -> (1,H,W)
#             img = img.unsqueeze(0)

#         elif img.ndim == 3:
#             # Could be (H,W,C) or (C,H,W)
#             if img.shape[-1] in [1, 2, 3]:
#                 # (H,W,C) -> (C,H,W)
#                 img = img.permute(2, 0, 1)

#         elif img.ndim == 4:
#             # (1,H,W,C) or (1,C,H,W) -> remove batch dim
#             img = img.squeeze(0)
#             if img.shape[-1] in [1, 2, 3]:
#                 img = img.permute(2, 0, 1)

#         # ---- Ensure exactly 3 channels ----
#         C = img.shape[0]

#         if C == 1:
#             img = img.repeat(3, 1, 1)          # (3,H,W)
#         elif C == 2:
#             zero = torch.zeros_like(img[:1])   # (1,H,W)
#             img = torch.cat([img, zero], dim=0)  # (3,H,W)
#         elif C > 3:
#             img = img[:3]

#         img = self.img_transform(img)


#         # ----------------------------
#         # Ensure mask is 2D (H,W)
#         # ----------------------------
#         if mask.ndim == 3:
#             mask = mask[0]  # remove channel dim if present

#         elif mask.ndim > 2:
#             # collapse extra dims (for volumes)
#             mask = mask.squeeze()

#         # ----------------------------
#         # Resize mask using nearest
#         # ----------------------------
#         mask = mask.unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
#         mask = F.interpolate(mask, size=(1008, 1008), mode="nearest")[0, 0]


#         return img, mask

# ------------------------------------------------------------
# Universal Slice Dataset (FIXED)
# ------------------------------------------------------------
class UniversalSliceDataset(Dataset):
    """
    SAFE slice dataset:
    - raw_loader is materialized ONCE
    - __getitem__ never touches DataLoader
    - multiprocessing-safe
    """

    def __init__(self, raw_loader, image_keys=None, mask_key="curGT", min_fg_pixels=10):
        self.image_keys = image_keys
        self.mask_key = mask_key
        self.min_fg_pixels = min_fg_pixels

        print("⏳ Indexing slices (materializing raw loader once)...")

        # ✅ CRITICAL FIX: materialize loader ONCE
        self.batches = list(raw_loader)

        self.slice_index = []  # (batch_idx, b, z)

        for batch_idx, batch in enumerate(self.batches):
            if isinstance(batch, dict):
                gt = batch[mask_key]  # (B,1,H,W,D)
                B, _, _, _, D = gt.shape
                for b in range(B):
                    for z in range(D):
                        if gt[b, 0, :, :, z].sum() >= min_fg_pixels:
                            self.slice_index.append((batch_idx, b, z))
            else:
                img, mask = batch[:2]
                if mask.sum() >= min_fg_pixels:
                    self.slice_index.append((batch_idx, 0, 0))

        print(f"✅ Total indexed slices: {len(self.slice_index)}")

        self.img_transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize((1008, 1008)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

    def __len__(self):
        return len(self.slice_index)

    def __getitem__(self, idx):
        batch_idx, b, z = self.slice_index[idx]
        batch = self.batches[batch_idx]

        # ---------- IMAGE ----------
        if isinstance(batch, dict):
            imgs = [batch[k][b, :, :, :, z] for k in self.image_keys]
            img = torch.cat(imgs, dim=0)

            # PET/CT → 3 channels
            if len(self.image_keys) == 2:
                zero = torch.zeros_like(img[:1])
                img = torch.cat([img[0:1], img[1:2], zero], dim=0)

            mask = batch[self.mask_key][b, 0, :, :, z]
        else:
            img, mask = batch[:2]

        # ---- Normalize image to (C, H, W) ----
        if img.ndim == 2:
            # (H,W) -> (1,H,W)
            img = img.unsqueeze(0)

        elif img.ndim == 3:
            # Could be (H,W,C) or (C,H,W)
            if img.shape[-1] in [1, 2, 3]:
                # (H,W,C) -> (C,H,W)
                img = img.permute(2, 0, 1)

        elif img.ndim == 4:
            # (1,H,W,C) or (1,C,H,W) -> remove batch dim
            img = img.squeeze(0)
            if img.shape[-1] in [1, 2, 3]:
                img = img.permute(2, 0, 1)

        # ---- Ensure exactly 3 channels ----
        C = img.shape[0]

        if C == 1:
            img = img.repeat(3, 1, 1)          # (3,H,W)
        elif C == 2:
            zero = torch.zeros_like(img[:1])   # (1,H,W)
            img = torch.cat([img, zero], dim=0)  # (3,H,W)
        elif C > 3:
            img = img[:3]

        img = self.img_transform(img)


        # ----------------------------
        # Ensure mask is 2D (H,W)
        # ----------------------------
        if mask.ndim == 3:
            mask = mask[0]  # remove channel dim if present

        elif mask.ndim > 2:
            # collapse extra dims (for volumes)
            mask = mask.squeeze()

        # ----------------------------
        # Resize mask using nearest
        # ----------------------------
        mask = mask.unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
        mask = F.interpolate(mask, size=(1008, 1008), mode="nearest")[0, 0]

        return img, mask


# ------------------------------------------------------------
# Build all train loaders
# ------------------------------------------------------------
def build_train_dataloaders(root_path, batch_size):
    loaders = []
    for name, fn, keys, modality in all_datasets():
        raw = fn(root_path, split="train", modality_type=modality, use_v2=True, batch_size=1)

        if keys is not None:
            image_keys = keys["image"]
            if isinstance(image_keys, str):
                image_keys = (image_keys,)
            ds = UniversalSliceDataset(raw, image_keys=image_keys, mask_key=keys["mask"])
        else:
            ds = UniversalSliceDataset(raw)

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

        loaders.append((name, loader))
        print(f"✅ Loaded {name}: {len(ds)} slices")

    return loaders

# def build_train_dataloaders(root_path, batch_size):
#     DATASET_REGISTRY = all_datasets()
#     loaders = []

#     for name, fn, keys, modality in DATASET_REGISTRY:
#         raw_loader = fn(root_path, split="train", modality_type=modality, use_v2=True, batch_size=1)
#         if keys is not None:
#             image_keys = keys["image"]
#             if isinstance(image_keys, str):
#                 image_keys = (image_keys,)
#             dataset = UniversalSliceDataset(raw_loader, image_keys=image_keys, mask_key=keys["mask"])
#         else:
#             dataset = UniversalSliceDataset(raw_loader)

#         loader = DataLoader(
#             dataset,
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=2,
#             pin_memory=False,
#             drop_last=True
#         )
#         loaders.append((name, loader))
#         print(f"✅ Loaded {name}: {len(dataset)} slices")
#     return loaders

# ------------------------
# Main training
# ------------------------
def main():
    # seting hyper-parameters and save path
    USE_ADAPTER = True

    root_path = "/home/gxu/proj1/lesionSeg/sam3_brain"
    save_folder = cfg.training.save_folder
    ckpt_dir = os.path.join(root_path, "checkpoint_exp", save_folder)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Copy script for reproducibility
    shutil.copyfile(__file__, os.path.join(ckpt_dir, os.path.basename(__file__)))
    shutil.copyfile(os.path.join(root_path, "train_configs/mets_ft_adp.yaml"), os.path.join(ckpt_dir, "mets_ft_adp_used.yaml"))

    # Load loss, optimizer, scheduler from config
    loss_fn = instantiate(cfg.loss, _convert_="all")

    device = cfg.training.device
    batch_size = cfg.training.batch_size
    num_epochs = cfg.training.epochs
    ACCUM_STEPS = cfg.training.accum_steps
    scaler = torch.amp.GradScaler(enabled=cfg.training.mixed_precision == "fp16")

    # ------------------------
    # Load ALL training datasets
    # ------------------------
    train_loaders = build_train_dataloaders(
        root_path=root_path,
        batch_size=batch_size
    )

    # # Validation stays single-dataset (Brats)
    # val_dataset = MRISliceDataset_v2(
    #     load_data_brats_mets(root_path, split="val", modality_type="v2")
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True
    # )

    # print(f"✅ Validation slices: {len(val_dataset)}")

    # ------------------------
    # Build model
    # ------------------------
    model = load_sam3(device, use_adapter=USE_ADAPTER)
    for p in model.backbone.vision_backbone.parameters():
        p.requires_grad = not cfg.model.freeze_vision_encoder
    for p in model.backbone.language_backbone.parameters():
        p.requires_grad = not cfg.model.freeze_text_encoder
    model.train()

    if USE_ADAPTER: # only for the adapter finetuning case
        for p in model.parameters():
            p.requires_grad = False

        for blk in model.backbone.vision_backbone.trunk.blocks:
            if hasattr(blk, "adapter"):
                for p in blk.adapter.parameters():
                    p.requires_grad = True

    total, trainable = count_params(model)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    
    # ------------------------
    # Optimizer & Scheduler
    # ------------------------
    adapter_lr = cfg.scratch.adapter_lr

    adapter_params = []
    geometry_params = []
    transformer_params = []
    segmentation_params = []
    scoring_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "adapter" in name:
            adapter_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            # ======================
            # Adapter (HIGH LR)
            # ======================
            {
                "params": adapter_params,
                "lr": adapter_lr,
                # "weight_decay": 0.0,   # recommended for adapters
            },
        ],
        weight_decay=1e-4,
    )

    # checking learning rates
    for i, pg in enumerate(optimizer.param_groups):
        if any(p is adapter_params[0] for p in pg["params"]):
            print(f"Adapter group {i} LR = {pg['lr']}")

    # CRITICAL: remove frozen params
    for pg in optimizer.param_groups:
        pg["params"] = [p for p in pg["params"] if p.requires_grad]

    num_optim_params = sum(
        p.numel() for pg in optimizer.param_groups for p in pg["params"]
    )
    print(f"Params in optimizer: {num_optim_params:,}")

    optimizer.zero_grad(set_to_none=True)
    gradient_clipper = (
            instantiate(cfg.optim.gradient_clip) if cfg.optim else None
        )


    # ------------------------
    # Training loop
    # ------------------------
    loss_history = []
    step_history = []
    global_step = 0
    dice_val_best = -1.0
    accum_loss = 0.0
    accum_count = 0

    for epoch in range(num_epochs):

        # 🔀 Randomize dataset order every epoch
        shuffle(train_loaders)

        print(f"\n🔁 Epoch {epoch+1} dataset order:")
        print([name for name, _ in train_loaders])

        epoch_loss = 0.0

        for ds_name, train_loader in train_loaders:
            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1} | {ds_name}",
                leave=False
            )

            for imgs, masks in pbar:
                imgs, masks = imgs.to(device), masks.to(device)

                for i in range(imgs.shape[0]):
                    img = imgs[i].unsqueeze(0)
                    gt  = masks[i].unsqueeze(0).unsqueeze(0)

                    img_bd = convert_to_batched_datapoint(img, gt)

                    with torch.amp.autocast(device, dtype=torch.float16):
                        out = model(img_bd)
                        targets = [model.back_convert(t) for t in img_bd.find_targets]
                        loss = loss_fn(out, targets)["core_loss"] / ACCUM_STEPS

                    scaler.scale(loss).backward()

                    epoch_loss += loss.item()
                    accum_loss += loss.item()
                    accum_count += 1

                    if accum_count == ACCUM_STEPS:
                        if gradient_clipper is not None:
                            scaler.unscale_(optimizer)
                            gradient_clipper(model=model)

                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                        global_step += 1

                        with open(os.path.join(ckpt_dir, "train_loss_record_iter.txt"), "a") as f:
                            f.write(
                                f"Epoch:{epoch+1} Step:{global_step} "
                                f"Dataset:{ds_name} Loss:{accum_loss:.4f}\n"
                            )

                        loss_history.append(accum_loss)
                        step_history.append(global_step)
                        accum_loss = 0.0
                        accum_count = 0

        # Epoch-level logging
        print(f"✅ Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss:.4f}")
        with open(os.path.join(ckpt_dir, "train_loss_record_epoch.txt"), "a") as f:
            f.write(f"Epoch: [{epoch+1}/{num_epochs}]\tLoss: {epoch_loss:.4f}\n")

        # # Save each trained model
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"sam3_mri_epoch_{epoch+1}.pth"))

        # # ------------------------
        # # Validation
        # # ------------------------
        # metrics = evaluate_sam3(model, val_loader, device)
        # dice_val_cur = metrics['dice']
        # print(f"[Epoch {epoch+1}] Dice: {dice_val_cur:.4f}, IoU: {metrics['iou']:.4f}")

        # with open(os.path.join(ckpt_dir, "eval_record_epoch.txt"), "a") as f:
        #     f.write(f"Epoch: [{epoch+1}/{num_epochs}], Dice: {dice_val_cur:.4f}, IoU: {metrics['iou']:.4f}\n")

        # if dice_val_cur > dice_val_best:
        #     dice_val_best = dice_val_cur
        #     torch.save(model.state_dict(), os.path.join(ckpt_dir, "sam3_mri_best_val.pth"))


    # # Save the final model
    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"sam3_mri_epoch_{epoch+1}.pth"))

    # Plot final loss curve
    plot_loss(step_history, loss_history, ckpt_dir)


if __name__ == "__main__":
    register_omegaconf_resolvers()
    main()
    if dist.is_initialized():
        dist.destroy_process_group()
