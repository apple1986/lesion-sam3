# ============================================================
# Exemplar-based SAM3 inference with bbox propagation
# ============================================================

import sys, os, shutil, argparse
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2")
sys.path.append("/home/gxu/proj1/lesionSeg/sam3_brain")

import numpy as np
import torch
import cv2
from tqdm import tqdm
from os.path import join
from os import makedirs

from ap_knife.sam_data_transforms import SAM3Transforms
from ap_knife.save_data import save_rgb_image, save_mask
from gp_knife.get_prompts import get_all_bounding_boxes_with_ccl_box_rand
from gp_knife.eval_model import get_dice_hd_nsd_asd
from ap_knife.eval_metrics import detection_accuracy_2d
from data_load.busi_builder import load_data_busi

import sam3
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox

# ============================================================
# Utility functions
# ============================================================

def crop_by_bbox(img, mask, box):
    x1, y1, x2, y2 = map(int, box)
    return img[y1:y2, x1:x2], mask[y1:y2, x1:x2]

def resize_to_32(img, mask):
    img32 = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)
    mask32 = cv2.resize(mask, (32, 32), interpolation=cv2.INTER_NEAREST)
    return img32, mask32

def norm_box(box, W, H, device):
    box_512 = box / np.array([W, H, W, H]) * 512
    x1, y1, x2, y2 = box_512
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    box = torch.tensor([[cx, cy, w, h]], device=device)
    return normalize_bbox(box, 512, 512).flatten().tolist()

@torch.no_grad()
def load_sam3(ckpt, device):
    return build_sam3_image_model(
        bpe_path=f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz",
        device=device,
        checkpoint_path=ckpt,
        enable_inst_interactivity=True,
        eval_mode=True
    )

# ============================================================
# Main
# ============================================================

def main():

    device = "cuda:1"
    root_path = "/home/gxu/proj1/lesionSeg/sam3_brain"
    ckpt = join(root_path, "checkpoint/sam3.pt")
    save_root = join(root_path, "busi/sam3_exemplar")

    makedirs(save_root, exist_ok=True)
    makedirs(join(save_root, "seg_result"), exist_ok=True)
    makedirs(join(save_root, "eval_result"), exist_ok=True)

    # Model
    model = load_sam3(ckpt, device)
    processor = Sam3Processor(model, confidence_threshold=0.5, device=device)
    transforms = SAM3Transforms(resolution=512, mask_threshold=0)

    # Dataset
    test_loader = load_data_busi(
        root_path,
        transform=None,
        batch_size=1,
        split="test",
        modality_type="Ultrasound",
        use_v2=True
    )

    dice_fun, hd95_fun, nsd_fun, asd_fun = get_dice_hd_nsd_asd()
    box_func = get_all_bounding_boxes_with_ccl_box_rand

    exemplar_box = None
    exemplar_img32 = None
    exemplar_mask32 = None
    is_first_case = True

    dice_all = []

    for img, gt, caseID in tqdm(test_loader):
        img = img.to(device)
        gt = gt.to(device)

        img_np = img[0].permute(1, 2, 0).cpu().numpy()
        gt_np = gt[0].cpu().numpy()

        if gt_np.sum() < 2:
            continue

        label_id = np.unique(gt_np)[1]
        mask_bin = (gt_np == label_id).astype(np.uint8)

        bbox_all = box_func(mask_bin, bbox_shift=0)
        box_ori = bbox_all[0] if bbox_all.ndim > 1 else bbox_all

        # -------------------- EXAMPLAR --------------------
        if is_first_case:
            crop_img, crop_mask = crop_by_bbox(img_np, gt_np, box_ori)
            exemplar_img32, exemplar_mask32 = resize_to_32(crop_img, crop_mask)
            exemplar_box = box_ori.copy()
            is_first_case = False

            save_rgb_image(exemplar_img32, join(save_root, "exemplar_img32.png"))
            save_mask(exemplar_mask32, join(save_root, "exemplar_mask32.png"))

        # -------------------- INFERENCE --------------------
        img_tensor = transforms(img_np).to(device)
        state = processor.set_image(img_tensor)
        processor.reset_all_prompts(state)

        norm_bbox = norm_box(exemplar_box, img_np.shape[1], img_np.shape[0], device)
        state = processor.add_geometric_prompt(state, box=norm_bbox, label=True)

        pred = np.zeros_like(gt_np, dtype=np.uint8)
        for n in range(len(state["scores"])):
            m = state["masks"][n][0].cpu().numpy().astype(np.uint8)
            pred[m > 0] = 1

        # -------------------- MASKED EVAL --------------------
        eval_mask = np.ones_like(gt_np, dtype=np.uint8)
        x1, y1, x2, y2 = map(int, exemplar_box)
        eval_mask[y1:y2, x1:x2] = 0

        pd = torch.from_numpy(pred * eval_mask)[None, None]
        gt_t = torch.from_numpy(gt_np * eval_mask)[None, None]

        dice = dice_fun(pd, gt_t).item()
        dice_all.append(dice)

        save_mask(pred, join(save_root, "seg_result", f"{caseID[0]}_pd.png"))
        save_mask(gt_np, join(save_root, "seg_result", f"{caseID[0]}_gt.png"))

        print(f"{caseID[0]} | Dice (masked): {dice:.4f}")

    print(f"\nMean Dice: {np.mean(dice_all):.4f}")

# ============================================================
if __name__ == "__main__":
    main()
