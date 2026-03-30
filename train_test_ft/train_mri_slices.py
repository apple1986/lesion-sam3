import sys
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2")
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2")
# sys.path.append("/home/gxu/proj1/lesionSeg/sam3_brain")

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset.ustwlits import ToTensor
from data_load.brats_builder import load_data_brats_mets
from gp_knife.get_prompts import get_all_bounding_boxes_with_ccl_box_rand

# ========= SAM3 ==========
import sam3
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox

# =========================
device = "cuda"
batch_size = 4
num_epochs = 50
lr = 1e-5
image_size = 512


# =========================
# ✅ Slice Dataset (从 3D MRI 随机采样 2D slice)
# =========================
class MRISliceDataset(Dataset):
    def __init__(self, dataloader):
        self.data = []
        for batch in dataloader:

            curMR = batch["curMR"]   # ✅ (B, C, H, W, D)
            curGT = batch["curGT"]   # ✅ (B, C, H, W, D)

            # 取 batch 中第 0 个样本
            curMR = curMR[0]   # (C, H, W, D)
            curGT = curGT[0]   # (C, H, W, D)

            C, H, W, D = curMR.shape

            for z in range(D):
                img  = curMR[:, :, :, z]   # ✅ (C, H, W)
                mask = curGT[:, :, :, z]   # ✅ (C, H, W)

                # ✅ 如果 mask 是单通道，取第 0 通道
                if mask.shape[0] > 1:
                    mask = mask[0]
                else:
                    mask = mask.squeeze(0)   # -> (H, W)

                self.data.append((img, mask))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, mask = self.data[idx]

        # ✅ 图像转 3 通道
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)   # (3,H,W)
        elif img.shape[0] >= 3:
            img = img[:3]               # 防止多模态直接截前 3 个

        # ✅ Resize image
        img = F.interpolate(
            img.unsqueeze(0),
            size=(512, 512),
            mode="bilinear",
            align_corners=False
        )[0]

        # ✅ Resize mask（严格 2D）
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),  # (1,1,H,W)
            size=(512, 512),
            mode="nearest"
        )[0, 0]   # → (512,512)

        return img, mask



# =========================
# ✅ Build SAM3
# =========================
def load_sam3():
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        enable_inst_interactivity=True,
    )
    return model.to(device)


# =========================
# ✅ Dice Loss
# =========================
def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2 * inter + eps) / (union + eps)


# =========================
# ✅ Training Script
# =========================
def main():

    root_path = "/home/gxu/proj1/lesionSeg/sam3_brain"

    all_transform = transforms.Compose([
        ToTensor(),
    ])

    base_loader = load_data_brats_mets(
        root_path,
        all_transform,
        batch_size=1,
        split="train"
    )

    print("✅ Loading MRI slices ...")
    train_dataset = MRISliceDataset(base_loader)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    print(f"✅ Total training slices: {len(train_dataset)}")

    # ---------- Model ----------
    model = load_sam3()
    processor = Sam3Processor(model, confidence_threshold=0.5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()

    for epoch in range(num_epochs):

        epoch_loss = 0

        for imgs, masks in tqdm(train_loader):

            imgs = imgs.to(device)       # B,3,512,512
            masks = masks.to(device)     # B,512,512

            optimizer.zero_grad()
            loss_total = 0

            for i in range(imgs.shape[0]):

                img = imgs[i]
                gt  = masks[i]

                inference_state = processor.set_image(img)
                processor.reset_all_prompts(inference_state)

                # ✅ 使用 GT 生成 box
                gt_np = gt.cpu().numpy().astype(np.uint8)
                if np.sum(gt_np) < 10:
                    continue
                boxes = get_all_bounding_boxes_with_ccl_box_rand(gt_np)
                if boxes.ndim > 1:
                    box = boxes[0]
                else:
                    box = boxes

                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w  = x2 - x1
                h  = y2 - y1

                # box_tensor = torch.tensor([[cx, cy, w, h]]).to(device)
                # norm_box = normalize_bbox(box_tensor, 512, 512).flatten().tolist()

                # convert the xyxy to cxcywh
                box_input_cxcywh = np.array([cx, cy, w, h])
                box_input_cxcywh = torch.from_numpy(box_input_cxcywh).to(device).view(-1, 4)
                norm_box = normalize_bbox(box_input_cxcywh, 512, 512).flatten().tolist()                

                inference_state = processor.add_geometric_prompt(
                    state=inference_state,
                    box=norm_box,
                    label=True
                )

                # pred = inference_state["masks"][0]
                pred_logits = inference_state["masks_logits"]
                # pred = F.interpolate(pred.unsqueeze(0), size=(512, 512), mode="bilinear")[0, 0]

                loss = dice_loss(pred_logits, gt) + F.binary_cross_entropy_with_logits(pred_logits, gt.unsqueeze(0).unsqueeze(0))
                loss.backward()
                loss_total += loss.item()

            optimizer.step()
            epoch_loss += loss_total

        print(f"✅ Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss:.4f}")

        torch.save(
            model.state_dict(),
            f"/home/gxu/proj1/lesionSeg/sam3_brain/checkpoint_exp/brats_mets/sam3_mri_epoch_{epoch+1}.pth"
        )


if __name__ == "__main__":
    main()
