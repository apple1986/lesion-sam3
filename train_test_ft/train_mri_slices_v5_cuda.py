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

from dataset.ustwlits import ToTensor
from data_load.brats_builder import load_data_brats_mets
from ap_knife.get_prompts import get_all_bounding_boxes_with_ccl_box_rand as gen_box_rand
from torchvision.transforms import v2
from sam2.utils.transforms import SAM2Transforms

# ===== SAM3 =====
import sam3
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
from sam3 import build_sam3_image_model
from sam3.model.data_misc import BatchedDatapoint, FindStage, BatchedFindTarget, BatchedInferenceMetadata
from sam3.model.utils.misc import copy_data_to_device


class MRISliceDataset(Dataset):
    def __init__(self, dataloader):
        self.data = []

        # ----------------------------
        # Preload all slices into list
        # ----------------------------
        for batch in dataloader:
            curMR = batch["curMR"]  # (B,C,H,W,D)
            curGT = batch["curGT"]  # (B,C,H,W,D)

            for b in range(curMR.shape[0]):
                C, H, W, D = curMR[b].shape
                for z in range(D):
                    img = curMR[b, :, :, :, z]   # (C,H,W)
                    mask = curGT[b, :, :, :, z]  # (C,H,W)
                    if mask.sum() < 10:
                        continue
                    mask = mask[0] if mask.shape[0] > 1 else mask.squeeze(0)
                    self.data.append((img, mask))

        # ----------------------------
        # Image transform (torchvision v2)
        # Mask transform is handled manually
        # ----------------------------
        self.sam2_transforms = SAM2Transforms(resolution=512, mask_threshold=0)

        self.img_transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),        # convert from float → uint8
                v2.Resize(size=(1008, 1008)),               # resize image
                v2.ToDtype(torch.float32, scale=True),      # back to float
                v2.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, mask = self.data[idx]   # img = (C,H,W), mask=(H,W)

        # ----------------------------
        # Ensure 3 channels for SAM3
        # ----------------------------
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] > 3:
            img = img[:3]

        # ----------------------------
        # Apply image transform
        # ----------------------------
        img = self.sam2_transforms(img.permute((1,2,0)).numpy())
        img = self.img_transform(img)  # (H,W,C)

        # ----------------------------
        # Resize mask (using nearest)
        # ----------------------------
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(1008, 1008),
            mode="nearest"
        )[0, 0]  # (1008,1008)

        return img, mask


def convert_to_batched_datapoint(imgs: torch.Tensor, masks: torch.Tensor) -> BatchedDatapoint:
    import torch.nn.functional as F
    B, C, H, W = imgs.shape
    device = imgs.device

    # ==========================================================
    # 1. Resize to SAM3-required 1008×1008
    # ==========================================================
    if (H, W) != (1008, 1008):
        imgs = F.interpolate(imgs, (1008, 1008), mode="bilinear", align_corners=False)
        masks = F.interpolate(masks.float(), (1008, 1008), mode="nearest").long()

    # ==========================================================
    # 2. Text prompt
    # ==========================================================
    find_text_batch = ["lesion"] * B

    # ==========================================================
    # 3. Points + Boxes
    # ==========================================================
    boxes_list = []
    for i in range(B):
        m = masks[i, 0]  # (1008,1008)              

        # ---------- boxes ----------
        boxes = gen_box_rand(m, bbox_shift=0)

        if boxes is None or len(boxes) == 0:
            boxes_list.append(torch.zeros((0, 4), device=device))
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32, device=device)
            if boxes.ndim == 1:
                boxes = boxes.unsqueeze(0)  # (1,4)
            boxes_list.append(boxes)


    # ==========================================================
    # 4. Build Find Inputs (PROMPTS)
    # ==========================================================
    find_inputs = [
        FindStage(
            img_ids=torch.tensor([i], device=device),
            text_ids=torch.tensor([i], device=device),

            # enable box prompt!!!
            input_boxes=boxes_list[i].unsqueeze(1), 
            # input_boxes=boxes_list[i] if boxes_list[i].shape[0] > 0 else None,
            input_boxes_mask=None, #boxes_mask_list[i] if boxes_list[i].shape[0] > 0 else None,
            input_boxes_label=torch.ones((boxes_list[i].shape[0], 1), dtype=torch.long, device=device),

            input_points=None,
            input_points_mask=None,
        )
        for i in range(B)
    ]

    # ==========================================================
    # 5. Find Targets (GT: boxes + mask)
    # ==========================================================
    find_targets = []
    for i in range(B):

        num_box = boxes_list[i].shape[0]   # 1 or 0
        semantic_seg = masks[i].long().to(device)  # (1, H, W)

        find_targets.append(
            BatchedFindTarget(
                num_boxes=torch.tensor([num_box], dtype=torch.long, device=device),

                boxes=boxes_list[i].to(device),              # (1,4) or (0,4)
                boxes_padded=boxes_list[i].unsqueeze(0).to(device),       # safe for 1 object
                repeated_boxes=boxes_list[i].to(device),     # SAM3 expects repeat, but 1 is fine

                segments=masks[i].reshape(1, -1).to(device), # shape = (1, HW)

                semantic_segments=semantic_seg,
                is_valid_segment=torch.tensor([True], device=device),
                is_exhaustive=torch.tensor([True], device=device),
                object_ids=torch.tensor([0], device=device),
                object_ids_padded=torch.tensor([0], device=device)
            )
        )

    # ==========================================================
    # 6. Metadata
    # ==========================================================
    find_metadatas = [
        BatchedInferenceMetadata(
            coco_image_id=torch.tensor([i]),
            original_image_id=torch.tensor([i]),
            original_category_id=torch.tensor([0]),
            original_size=torch.tensor([H, W]),
            object_id=torch.tensor([0]),
            frame_index=torch.tensor([0]),
            is_conditioning_only=torch.tensor([False])
        )
        for i in range(B)
    ]

    # ==========================================================
    # 7. Final Output
    # ==========================================================
    return BatchedDatapoint(
        img_batch=imgs,
        find_text_batch=find_text_batch,
        find_inputs=find_inputs,
        find_targets=find_targets,
        find_metadatas=find_metadatas
    )




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
# =========================
# Dice Loss
# =========================
def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2 * inter + eps) / (union + eps)


## loss
import torch.distributed as dist

if not dist.is_initialized():
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method="tcp://127.0.0.1:29500",
        world_size=1,
        rank=0,
    )
from hydra.utils import instantiate

matcher_conf = {
    "_target_": "sam3.train.matcher.BinaryHungarianMatcherV2",
    "focal": True, # use focal loss to do classification
    "cost_class": 2.0, # weight of classification loss
    "cost_bbox": 5.0, # L1 bbox loss weight
    "cost_giou": 2.0, # IoU loss weight
    "alpha": 0.25, # focal α
    "gamma": 2.0, # focal γ
    "stable": False,
}
loss_conf = {
    "_target_": "sam3.train.loss.sam3_loss.Sam3LossWrapper",

    "matcher": matcher_conf,

    "o2m_weight": 2.0, # One-to-Many Matching, weight of the total loss
    "o2m_matcher": {
        "_target_": "sam3.train.matcher.BinaryOneToManyMatcher",
        "alpha": 0.3, # postive and negative rate
        "threshold": 0.4, # IoU for threshold
        "topk": 4, # each GT for 4 queries
    },
    "use_o2m_matcher_on_o2m_aux": False,

    "loss_fns_find": [
        {
            "_target_": "sam3.train.loss.loss_fns.Boxes",
            "weight_dict": {
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
            },
        },
        {
            "_target_": "sam3.train.loss.loss_fns.IABCEMdetr",
            "weak_loss": False,
            "weight_dict": {
                "loss_ce": 5.0, # query classification
                "presence_loss": 2.0, # classification of presence token
            },
            "pos_weight": 10.0, # postive sample weight
            "alpha": 0.25, # focal loss
            "gamma": 2.0, # focal loss
            "use_presence": True,
            "pos_focal": False,
            "pad_n_queries": 200,
            "pad_scale_pos": 1.0,
        },
    ],

    "loss_fn_semantic_seg": None,
    "scale_by_find_batch_size": True,
}

loss_fn = instantiate(loss_conf, _convert_="all")
device = "cuda:1"
batch_size = 1
num_epochs = 10
lr = 5e-6
image_size = 512
# =========================
# Training
# =========================
def main():
    root_path = "/home/gxu/proj1/lesionSeg/sam3_brain"

    all_transform = transforms.Compose([ToTensor()])

    base_loader = load_data_brats_mets(root_path, all_transform, batch_size=1, split="train")

    train_dataset = MRISliceDataset(base_loader)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    print(f"✅ Total training slices: {len(train_dataset)}")

    model = load_sam3(device)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for p in model.backbone.vision_backbone.parameters():
        p.requires_grad = False
    for p in model.backbone.language_backbone.parameters():
        p.requires_grad = False        

    optimizer = torch.optim.AdamW(
        [   
        # {"params": model.backbone.language_backbone.parameters(), "lr": lr*0.1},
            # {"params": model.backbone.vision_backbone.parameters(), "lr": lr*0.1},
            {"params": model.geometry_encoder.parameters(), "lr": lr},
            {
                "params": (
                    list(model.transformer.parameters()) +
                    list(model.segmentation_head.parameters()) +
                    list(model.dot_prod_scoring.parameters())
                ),
                "lr": lr,
            },
        ],
        weight_decay=1e-4,
    )

    model.train()

    cnt = 0
    total_four_slice_loss = 0
    iter_num = 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        for imgs, masks in tqdm(train_loader):
            imgs = imgs.to(device)        # B,3,512,512
            masks = masks.to(device)      # B,512,512

            optimizer.zero_grad()
            
            # Forward each image independently
            for i in range(imgs.shape[0]):
                img = imgs[i].unsqueeze(0)          # 1,3,512,512
                gt  = masks[i].unsqueeze(0).unsqueeze(0)  # 1,1,512,512

                # Directly forward through model (training mode)
                # pred_logits shape: (1,1,H,W)
                if gt.sum() < 10:
                    # Skip slices with no lesion
                    continue
                img_bd = convert_to_batched_datapoint(img, gt)
                out = model(img_bd) # torch.Size([1, 3, 1008, 1008]) 
                cnt = cnt + 1

                # Convert GT (CRITICAL)
                find_targets = [
                    model.back_convert(x) for x in img_bd.find_targets
                ]               
                loss_dict = loss_fn(out, find_targets)
                loss = loss_dict["core_loss"]
                total_four_slice_loss += loss
                if cnt == 10:
                    # update total_four_slice_loss
                    total_loss = total_four_slice_loss / cnt
                    total_loss.backward()
                    optimizer.step()
                    print(f"Mean Slice: Loss: {total_loss.item():.4f}")
                    epoch_loss += total_loss.item()
                    # update
                    iter_num += 1
                    cnt = 0
                    total_four_slice_loss = 0
                
                # if iter_num == 10:
                #     iter_num = 0
                #     print(f"saving model")
                #     # save it as a txt file
                #     ckpt_dir = os.path.join(root_path, "checkpoint_exp/brats_mets_ft_detector")
                #     os.makedirs(ckpt_dir, exist_ok=True)
                #     with open(os.path.join(ckpt_dir, "train_loss_record.txt"), "a") as f:
                #         write_content = f'Epoch: [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}\n'
                #         f.write(write_content)        

                #     # Save checkpoint
                #     torch.save(
                #         model.state_dict(),
                #         os.path.join(ckpt_dir, f"sam3_mri_epoch_{num_epochs}.pth")
                #     )



        print(f"✅ Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss:.4f}")
        # save it as a txt file
        ckpt_dir = os.path.join(root_path, "checkpoint_exp/brats_mets_ft_detector")
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(os.path.join(ckpt_dir, "train_loss_record.txt"), "a") as f:
            write_content = f'Epoch: [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}\n'
            f.write(write_content)        

        # Save checkpoint
        torch.save(
            model.state_dict(),
            os.path.join(ckpt_dir, f"sam3_mri_epoch_{num_epochs}.pth")
        )

if __name__ == "__main__":
    main()
