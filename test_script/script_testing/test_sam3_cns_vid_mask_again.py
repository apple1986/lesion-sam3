#!/usr/bin/env python3
"""
run_sam3_video_inference.py

BBox + TEXT prompt only (no centroid refinement)
"""

import sys
import os
from os.path import join
from pathlib import Path
import argparse
import tempfile
import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms
import SimpleITK as sitk
import cv2

sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2")
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2")

from sam2.utils.transforms import SAM2Transforms
from dataset.ustwlits import ToTensor
from builders.utswdataset_builder import load_data
from gp_knife.get_prompts import (
    get_all_bounding_boxes_with_ccl_box_fix,
    get_all_bounding_boxes_with_ccl_box_rand,
)
from gp_knife.eval_model import get_dice_hd_nsd_asd

import sam3
from sam3.model_builder import build_sam3_video_model


# ---------------------------
# Helpers
# ---------------------------
def ensure_rgb_uint8(frame):
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] == 1:
        arr = np.concatenate([arr, arr, arr], axis=-1)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def volume_to_frames_list(volume):
    return [ensure_rgb_uint8(volume[z]) for z in range(volume.shape[0])]


def write_frames_to_folder(frames, folder):
    os.makedirs(folder, exist_ok=True)
    for i, f in enumerate(frames):
        fname = os.path.join(folder, f"{i:06d}.jpg")
        cv2.imwrite(fname, cv2.cvtColor(f, cv2.COLOR_RGB2BGR))


def normalize_box_xyxy(box_xyxy, W, H):
    x0, y0, x1, y1 = box_xyxy
    return [x0 / W, y0 / H, x1 / W, y1 / H]


def add_mask_prompt(predictor, inference_state, frame_idx, obj_id, gt_mask_slice, device='cuda'):
    """
    Add a GT mask + text prompt to SAM3 video predictor for a given frame.

    gt_mask_slice: np.array of shape (H, W), binary or uint8
    """
    # convert to tensor, add batch dim (1, H, W), and ensure dtype is uint8
    mask = torch.from_numpy(gt_mask_slice.astype(np.uint8))[None, :, :].squeeze(0).to(device)

    return predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        mask=mask,
        add_mask_to_memory=True,
    )

def run_sam3_inference(curMR_cat, curGT, predictor, caseID):
    """
    curMR_cat: torch tensor [B, 3, H, W, D]
    curGT:     torch tensor [B, 1, H, W, D]
    return:    segs_3d_case [D, H, W] uint8
    """

    B, C, H, W, D = curMR_cat.shape

    curMR_np = torch.permute(curMR_cat, (0, 4, 2, 3, 1)).cpu().numpy()[0]
    curGT_np = torch.permute(curGT, (0, 3, 1, 2)).cpu().numpy()[0]

    frames = volume_to_frames_list(curMR_np)

    with tempfile.TemporaryDirectory(prefix=f"sam3_vframes_{caseID}_") as tmpdir:
        write_frames_to_folder(frames, tmpdir)

        inference_state = predictor.init_state(video_path=tmpdir)
        predictor.clear_all_points_in_video(inference_state)

        segs_3d_case = np.zeros_like(curGT_np, dtype=np.uint8)

        label_ids = np.unique(curGT_np)
        label_ids = label_ids[label_ids != 0]

        for label_id in label_ids:
            mask3d = (curGT_np == label_id).astype(np.uint8)
            z_inds = np.where(mask3d > 0)[0]
            if z_inds.size == 0:
                continue

            prompt_slice = int(z_inds[0])
            obj_id = int(label_id)

            add_mask_prompt(
                predictor,
                inference_state,
                prompt_slice,
                obj_id,
                mask3d[prompt_slice],
            )

            video_segments = {}

            for frame_idx, obj_ids, _, video_res, _ in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=0,
                max_frame_num_to_track=D,
                reverse=False,
                propagate_preflight=True,
            ):
                video_segments[frame_idx] = {}

                for i, rid in enumerate(obj_ids):
                    mask_arr = video_res[i]
                    if torch.is_tensor(mask_arr):
                        mask_arr = mask_arr.cpu().numpy()
                    if mask_arr.ndim == 3:
                        mask_arr = mask_arr[0]
                    if mask_arr.shape != (H, W):
                        mask_arr = cv2.resize(mask_arr.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

                    video_segments[frame_idx][int(rid)] = (mask_arr > 0).astype(np.uint8)

            for z in range(D):
                if z in video_segments and obj_id in video_segments[z]:
                    segs_3d_case[z] |= video_segments[z][obj_id]

    return segs_3d_case.astype(np.uint8)


# ---------------------------
# Main
# ---------------------------
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    box_func = get_all_bounding_boxes_with_ccl_box_fix if args.fix_bbox else get_all_bounding_boxes_with_ccl_box_rand
    sam2_transforms = SAM2Transforms(resolution=args.image_size, mask_threshold=0)

    to_tensor = transforms.Compose([ToTensor()])
    test_loader = load_data(args.data_root, to_tensor, batch_size=1, split=args.split, tv_type=args.tv_type)

    dice_fun, hd95_fun, nsd_fun, asd_fun = get_dice_hd_nsd_asd()

    sam3_model = build_sam3_video_model()
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone

    save_eval_path = Path(args.pred_save_dir)
    save_eval_path.mkdir(parents=True, exist_ok=True)

    dice_all, hd95_all, nsd_all, asd_all = [], [], [], []

    for batch in tqdm(test_loader, desc="cases"):
        caseID = batch["caseID"][0]
        curMR, preMR, preGT, curGT = batch["curMR"], batch["preMR"], batch["preGT"], batch["curGT"]

        curMR = curMR.to(device)
        preMR = preMR.to(device)
        curGT = curGT.to(device)
        preGT = preGT.to(device)
        # B, C, H, W, D = curMR.shape
        curGT_np = torch.permute(curGT, (0, 3, 1, 2)).cpu().numpy()[0] # CDHW

        # ===============================
        # PASS 1: curMR + preMR + preGT
        # ===============================
        curMR_cat_1 = torch.cat((curMR, preMR, preGT), dim=1)

        segs_3d_case = run_sam3_inference(
            curMR_cat_1,
            curGT,
            predictor,
            caseID,
        )

        # =========================================
        # PASS 2: curMR + preMR + segs_3d_case ✅
        # =========================================
        pd_tensor = torch.from_numpy(segs_3d_case)[None, None].permute(0, 1, 3, 4, 2).float().to(device)

        # # Convert both to bool
        # preGT_bool = preGT.bool()
        # pd_bool = pd_tensor.bool()

        # # Element-wise OR
        # fused_mask = (preGT_bool & pd_bool).float()  # back to float for network input      

        curMR_cat_2 = torch.cat((curMR, pd_tensor, preGT), dim=1) # curMR, pd_tensor, preGT
        segs_3d_case = run_sam3_inference(
            curMR_cat_2,
            curGT,
            predictor,
            caseID,
        )

        pd_tensor = torch.from_numpy(segs_3d_case)[None, None] # BCDHW
        gt_tensor = torch.from_numpy(curGT_np)[None, None] # BCDHW

        dice_case = dice_fun(pd_tensor, gt_tensor).numpy()[0, 0]
        hd95_case = hd95_fun(pd_tensor, gt_tensor).numpy()[0, 0]
        nsd_case = nsd_fun(pd_tensor, gt_tensor).numpy()[0, 0]
        asd_case = asd_fun(pd_tensor, gt_tensor).numpy()[0, 0]
        print(f"case {caseID} dice: {dice_case:.4f}")

        # write per-case
        with open(save_eval_path / f"test_each_rand{args.bbox_shift}_prdGT_pdMsk.txt", "a") as f:
            f.write(f"{caseID}\tdice:{dice_case:.4f}\thd95:{hd95_case:.4f}\tnsd:{nsd_case:.4f}\tasd:{asd_case:.4f}\n")            

        dice_all.append(dice_case); hd95_all.append(hd95_case); nsd_all.append(nsd_case); asd_all.append(asd_case)

        if args.visualize:
            seg_sitk = sitk.GetImageFromArray(segs_3d_case.astype(np.uint8))
            sitk.WriteImage(seg_sitk, join(args.pred_save_dir, f"{caseID}_pd_sam3_vid.nii.gz"))

    ## save format results
    with open(os.path.join(save_eval_path, "format_test_mean_dice_hd_nsd_asd_prdGT_pdMsk"+".txt"), "a") as f:
        write_content = f"{'#'*60}\nmean dice\tmean nsd\tmean hd95\tmean asd\t\n"
        write_content = write_content + f"{np.array(dice_all).mean():.4f}±{np.array(dice_all).std():.4f}\t" \
                                        f"{np.array(nsd_all).mean():.4f}±{np.array(nsd_all).std():.4f}\t" \
                                        f"{np.array(hd95_all).mean():.4f}±{np.array(hd95_all).std():.4f}\t" \
                                        f"{np.array(asd_all).mean():.4f}±{np.array(asd_all).std():.4f}\t\n"
        write_content = write_content + f"{np.array(dice_all).mean():.2f}±{np.array(dice_all).std():.2f}\t" \
                                        f"{np.array(nsd_all).mean():.2f}±{np.array(nsd_all).std():.2f}\t" \
                                        f"{np.array(hd95_all).mean():.2f}±{np.array(hd95_all).std():.2f}\t" \
                                        f"{np.array(asd_all).mean():.2f}±{np.array(asd_all).std():.2f}\t\n"
        write_content = write_content + f"{np.array(dice_all).mean()*100:.2f}\t" \
                                        f"{np.array(nsd_all).mean():.2f}\t" \
                                        f"{np.array(hd95_all).mean()*100:.2f}\t" \
                                        f"{np.array(asd_all).mean():.2f}\n"
        f.write(write_content)

# ---------------------------
# Entry
# ---------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="/home/gxu/proj1/lesionSeg/sam3_brain")
    p.add_argument("--pred_save_dir", type=str, default="/home/gxu/proj1/lesionSeg/sam3_brain/results/brain_cns/sam3_3ch_video_mask_again")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--bbox_shift", type=int, default=0)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--visualize", default=False)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--fix_bbox", default=False)
    p.add_argument("--split", type=str, default="brain_ctv")
    p.add_argument("--tv_type", type=str, default="brain_ctv")
    args = p.parse_args()
    main(args)
