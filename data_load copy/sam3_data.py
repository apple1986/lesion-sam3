import torch
import torch.nn as nn
from torchvision.transforms import Normalize, Resize, ToTensor
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch.nn.functional as F
from ap_knife.get_prompts import (
    get_all_bounding_boxes_with_ccl_box_rand as gen_box_rand
)
from sam3.model.data_misc import (
    BatchedDatapoint,
    FindStage,
    BatchedFindTarget,
    BatchedInferenceMetadata,
)
# ------------------------------------------------------------
# Dataset Format
# ------------------------------------------------------------
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
        # self.img_transform = v2.Compose(
        #     [
        #         v2.Resize(size=(1008, 1008)),               # resize image
        #         v2.ToDtype(torch.float32, scale=True),      # back to float
        #         v2.Normalize(mean=[0.5, 0.5, 0.5], # [0.485, 0.456, 0.406]
        #                      std=[0.5, 0.5, 0.5]), #  [0.229, 0.224, 0.225]
        #     ]
        # )
        self.transform_v1 = torch.jit.script(
            nn.Sequential(
                Resize((1008, 1008)),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #  all previous model used
            )
        )
        self.img_transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                # v2.Resize(size=(1008, 1008)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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
        img = self.transform_v1(img)
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


class MRISliceDataset_v2(Dataset):
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
        self.img_transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(1008, 1008)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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
        # semantic_seg = masks[i].long().to(device)  # (1, H, W)
        # semantic_seg = semantic_seg.repeat(num_box, 1, 1) 
        ## convert into instance masks
        gt_mask = masks[i, 0].to(device)  # (H, W), union mask
        H, W = gt_mask.shape

        instance_masks = []

        for b in boxes_list[i]:
            x1, y1, x2, y2 = b.long()

            inst_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
            inst_mask[y1:y2, x1:x2] = gt_mask[y1:y2, x1:x2] > 0

            instance_masks.append(inst_mask)

        if len(instance_masks) > 0:
            segments = torch.stack(instance_masks, dim=0)  # (num_box, H, W)
        else:
            segments = torch.zeros((0, H, W), dtype=torch.bool, device=device)

        # semantic_seg = segments.clone()

        find_targets.append(
            BatchedFindTarget(
                num_boxes=torch.tensor([num_box], dtype=torch.long, device=device),

                boxes=boxes_list[i].to(device),              # (1,4) or (0,4)
                boxes_padded=boxes_list[i].unsqueeze(0).to(device),       # safe for 1 object
                repeated_boxes=boxes_list[i].to(device),     # SAM3 expects repeat, but 1 is fine

                # segments=masks[i].reshape(1, -1).to(device), # shape = (1, HW)
                # semantic_segments=semantic_seg,

                segments = segments, #masks[i].to(torch.bool).to(device),  # instance masks
                semantic_segments=masks[i].long().to(device), #semantic_seg, # semantic_mask: 1xHxW

                is_valid_segment=torch.ones(num_box, dtype=torch.bool,device=device), #torch.tensor([True], device=device),
                is_exhaustive=torch.tensor([True], device=device),

                object_ids=torch.arange(num_box, device=device),
                object_ids_padded=torch.arange(num_box, device=device),                
                # object_ids=torch.tensor([0], device=device),
                # object_ids_padded=torch.tensor([0], device=device)
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