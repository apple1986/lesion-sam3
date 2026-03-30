# epoch=50, seg_loss, only one bbox as input
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
from sam3.train.trainer import unwrap_ddp_if_wrapped

# ===== SAM3 =====
import sam3
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
from sam3 import build_sam3_image_model
from sam3.model.data_misc import BatchedDatapoint, FindStage, BatchedFindTarget, BatchedInferenceMetadata
from sam3.model.utils.misc import copy_data_to_device


device = "cuda"
batch_size = 1
num_epochs = 50
lr = 5e-6
image_size = 512


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


# def convert_to_batched_datapoint(imgs: torch.Tensor, masks: torch.Tensor) -> BatchedDatapoint:
#     import torch.nn.functional as F
#     B, C, H, W = imgs.shape
#     device = imgs.device

#     # ==========================================================
#     # 1. Resize to SAM3-required 1008×1008
#     # ==========================================================
#     if (H, W) != (1008, 1008):
#         imgs = F.interpolate(imgs, (1008, 1008), mode="bilinear", align_corners=False)
#         masks = F.interpolate(masks.float(), (1008, 1008), mode="nearest").long()

#     # ==========================================================
#     # 2. Text prompt
#     # ==========================================================
#     find_text_batch = ["lesion"] * B

#     # ==========================================================
#     # 3. Points + Boxes
#     # ==========================================================
#     input_points_list = []
#     input_points_mask_list = []
#     boxes_list = []
#     boxes_mask_list = []

#     def mask_to_box(mask2d):
#         ys, xs = mask2d.nonzero(as_tuple=True)
#         if len(xs) == 0:
#             return None
#         return torch.tensor(
#             [xs.min(), ys.min(), xs.max(), ys.max()],
#             dtype=torch.float32,
#             device=mask2d.device
#         )

#     for i in range(B):
#         m = masks[i, 0]  # (1008,1008)
#         fg = (m > 0).nonzero()

#         # ----- point -----
#         if len(fg) == 0:
#             pt = torch.rand((1, 2), device=device) * 1008.0
#             pmask = torch.tensor([False], device=device)
#         else:
#             yx = fg.float().mean(0)
#             pt = yx.flip(0).unsqueeze(0)
#             pmask = torch.tensor([True], device=device)

#         input_points_list.append(pt)
#         input_points_mask_list.append(pmask)

#         # ----- bounding box -----
#         box = mask_to_box(m)

#         if box is None:
#             boxes_list.append(torch.zeros((0, 4), device=device))
#             boxes_mask_list.append(torch.zeros((0,), dtype=torch.bool, device=device))
#         else:
#             boxes_list.append(box.unsqueeze(0))  # (1,4)
#             boxes_mask_list.append(torch.tensor([True], device=device))

#     # ==========================================================
#     # 4. Build Find Inputs (PROMPTS)
#     # ==========================================================
#     find_inputs = [
#         FindStage(
#             img_ids=torch.tensor([i], device=device),
#             text_ids=torch.tensor([i], device=device),

#             # enable box prompt!!!
#             input_boxes=boxes_list[i] if boxes_list[i].shape[0] > 0 else None,
#             input_boxes_mask=None, #boxes_mask_list[i] if boxes_list[i].shape[0] > 0 else None,
#             input_boxes_label=None,

#             input_points=input_points_list[i],
#             input_points_mask=input_points_mask_list[i]
#         )
#         for i in range(B)
#     ]

#     # ==========================================================
#     # 5. Find Targets (GT: boxes + mask)
#     # ==========================================================
#     find_targets = []
#     for i in range(B):

#         num_box = boxes_list[i].shape[0]   # 1 or 0
#         semantic_seg = masks[i].long()  # (1, H, W)

#         find_targets.append(
#             BatchedFindTarget(
#                 num_boxes=torch.tensor([num_box], dtype=torch.long, device=device),

#                 boxes=boxes_list[i],              # (1,4) or (0,4)
#                 boxes_padded=boxes_list[i],       # safe for 1 object
#                 repeated_boxes=boxes_list[i],     # SAM3 expects repeat, but 1 is fine

#                 segments=masks[i].reshape(1, -1), # shape = (1, HW)

#                 semantic_segments=semantic_seg,
#                 is_valid_segment=torch.tensor([True], device=device),
#                 is_exhaustive=torch.tensor([True], device=device),
#                 object_ids=torch.tensor([0], device=device),
#                 object_ids_padded=torch.tensor([0], device=device)
#             )
#         )

#     # ==========================================================
#     # 6. Metadata
#     # ==========================================================
#     find_metadatas = [
#         BatchedInferenceMetadata(
#             coco_image_id=torch.tensor([i]),
#             original_image_id=torch.tensor([i]),
#             original_category_id=torch.tensor([0]),
#             original_size=torch.tensor([H, W]),
#             object_id=torch.tensor([0]),
#             frame_index=torch.tensor([0]),
#             is_conditioning_only=torch.tensor([False])
#         )
#         for i in range(B)
#     ]

#     # ==========================================================
#     # 7. Final Output
#     # ==========================================================
#     return BatchedDatapoint(
#         img_batch=imgs,
#         find_text_batch=find_text_batch,
#         find_inputs=find_inputs,
#         find_targets=find_targets,
#         find_metadatas=find_metadatas
#     )

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
    input_points_list = []
    input_points_mask_list = []
    boxes_list = []
    boxes_mask_list = []

    # def mask_to_box(mask2d):
    #     ys, xs = mask2d.nonzero(as_tuple=True)
    #     if len(xs) == 0:
    #         return None
    #     return torch.tensor(
    #         [xs.min(), ys.min(), xs.max(), ys.max()],
    #         dtype=torch.float32,
    #         device=mask2d.device
    #     )

    for i in range(B):
        m = masks[i, 0]  # (1008,1008)
        fg = (m > 0).nonzero()

        # ----- point -----
        if len(fg) == 0:
            pt = torch.rand((1, 2), device=device) * 1008.0
            pmask = torch.tensor([False], device=device)
        else:
            yx = fg.float().mean(0)
            pt = yx.flip(0).unsqueeze(0)
            pmask = torch.tensor([True], device=device)

        input_points_list.append(pt)
        input_points_mask_list.append(pmask)

        # ----- bounding box -----
        boxes = gen_box_rand(m, bbox_shift=2)
        if boxes.ndim > 1:
            # randomly select one boxes
            rand_idx = torch.randint(low=0, high=boxes.shape[0], size=(1,)).numpy()[0]
            box = torch.tensor(boxes[rand_idx], dtype=torch.float32, device=device)
        else:
            box = torch.tensor(boxes, dtype=torch.float32, device=device)

        if box is None:
            boxes_list.append(torch.zeros((0, 4), device=device))
            boxes_mask_list.append(torch.zeros((0,), dtype=torch.bool, device=device))
        else:
            boxes_list.append(box.unsqueeze(0))  # (1,4)
            boxes_mask_list.append(torch.tensor([True], device=device))

    # ==========================================================
    # 4. Build Find Inputs (PROMPTS)
    # ==========================================================
    find_inputs = [
        FindStage(
            img_ids=torch.tensor([i], device=device),
            text_ids=torch.tensor([i], device=device),

            # enable box prompt!!!
            input_boxes=boxes_list[i].unsqueeze(0),
            # input_boxes=boxes_list[i] if boxes_list[i].shape[0] > 0 else None,
            input_boxes_mask=None, #boxes_mask_list[i] if boxes_list[i].shape[0] > 0 else None,
            input_boxes_label=torch.ones((1, 1), dtype=torch.long, device=device),

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
        semantic_seg = masks[i].long()  # (1, H, W)

        find_targets.append(
            BatchedFindTarget(
                num_boxes=torch.tensor([num_box], dtype=torch.long, device=device),

                boxes=boxes_list[i],              # (1,4) or (0,4)
                boxes_padded=boxes_list[i],       # safe for 1 object
                repeated_boxes=boxes_list[i],     # SAM3 expects repeat, but 1 is fine

                segments=masks[i].reshape(1, -1), # shape = (1, HW)

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
def load_sam3():
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        eval_mode=False,
        enable_inst_interactivity=False,
    )
    return model.to(device)

# =========================
# Dice Loss
# =========================
def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2 * inter + eps) / (union + eps)


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

    model = load_sam3()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for imgs, masks in tqdm(train_loader):
            imgs = imgs.to(device)        # B,3,512,512
            masks = masks.to(device)      # B,512,512

            optimizer.zero_grad()
            loss_total = 0

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
                pred_logits = out[0]["semantic_seg"] # torch.Size([1, 1, 288, 288])
                # 1. Resize prediction to GT size
                pred_logits = F.interpolate(pred_logits, size=gt.shape[-2:], mode='bilinear', align_corners=False)

                # Compute loss
                loss = dice_loss(pred_logits, gt) + F.binary_cross_entropy_with_logits(pred_logits, gt)
                print(f"    Slice {i+1}/{imgs.shape[0]} | Loss: {loss.item():.4f}")
                loss.backward()
                loss_total += loss.item()

                optimizer.step()
                epoch_loss += loss_total


        print(f"✅ Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss:.4f}")
        # save it as a txt file
        ckpt_dir = os.path.join(root_path, "checkpoint_exp/brats_mets_box_text")
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
