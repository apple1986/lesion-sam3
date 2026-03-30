#!/usr/bin/env python3
"""
run_sam3_video_inference.py

Convert each 3D medical volume (D,H,W[,C]) into a frame folder, init SAM3 tracker state,
add box prompts on annotated slices, propagate masks across slices (video mode), then
evaluate and save results.

Notes:
 - This writes temporary JPEG frames for each case into a temp folder and calls predictor.init_state(video_path=frames_dir),
   matching the SAM3 example behavior that accepts a folder of frames.
 - Confirm your SAM3 version expects normalized boxes ([0..1]) — code uses normalized boxes.
 - If your SAM3 exposes an API to init state directly from frames (list of numpy arrays), you can replace the folder-writing
   part with that call (search for init_state_from_frames / init_state(frames=...) in your local package).
"""
import sys
import os
from os.path import join
from pathlib import Path
import argparse
import shutil
import tempfile
import numpy as np
from tqdm import tqdm
import torch
from torch.nn import functional as F
from torchvision import transforms
import SimpleITK as sitk
import cv2

# local paths (adjust if needed)
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

# SAM3 video model
import sam3
from sam3.model_builder import build_sam3_video_model
from sam3.model.sam3_video_inference import Sam3VideoInference

# ---------------------------
# Helpers
# ---------------------------
def ensure_rgb_uint8(frame):
    """
    Convert a single 2D/3-channel frame to uint8 RGB.
    frame: HxW (grayscale) or HxWxC (C>=1)
    returns: HxWx3 uint8
    """
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] == 1:
        arr = np.concatenate([arr, arr, arr], axis=-1)
    # # Normalize to 0-255 if not already uint8
    # if arr.dtype != np.uint8:
    #     arr = arr.astype(np.float32)
    #     mn, mx = arr.min(), arr.max()
    #     if mx - mn > 1e-6:
    #         arr = (arr - mn) / (mx - mn)
    #     else:
    #         arr = arr - mn
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr

def volume_to_frames_list(volume):
    """
    volume: np.ndarray of shape (D, H, W) or (D, H, W, C)
    returns list of HxWx3 uint8 frames
    """
    frames = []
    for z in range(volume.shape[0]):
        if z == 139:
            pass
        frame = volume[z]
        frames.append(ensure_rgb_uint8(frame))
    return frames

def write_frames_to_folder(frames, folder):
    """
    Write frames (list of numpy arrays) to folder as zero-padded JPEGs:
       000000.jpg, 000001.jpg, ...
    Returns list of file paths.
    """
    os.makedirs(folder, exist_ok=True)
    paths = []
    for i, f in enumerate(frames):
        fname = os.path.join(folder, f"{i:06d}.jpg")
        cv2.imwrite(fname, cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        paths.append(fname)
    return paths

def normalize_box_xyxy(box_xyxy, W, H):
    """
    Convert [x0,y0,x1,y1] in pixel coordinates to normalized [x0/W, y0/H, x1/W, y1/H]
    """
    x0, y0, x1, y1 = box_xyxy
    return [x0 / float(W), y0 / float(H), x1 / float(W), y1 / float(H)]

def add_sam3_prompt(predictor, inference_state, frame_idx, obj_id, box_xyxy, text_prompt, H, W, device='cpu'):
    """
    Add a SAM3 prompt using text_str and boxes_xywh.
    box_xyxy: [x0,y0,x1,y1] in pixels
    """
    # Convert to normalized x,y,w,h
    x0, y0, x1, y1 = box_xyxy
    x = x0 / W
    y = y0 / H
    w = (x1 - x0) / W
    h = (y1 - y0) / H
    boxes_xywh = torch.tensor([[x, y, w, h]], dtype=torch.float32).to(device)

    # SAM3 API: add_prompt(frame_idx, text_str, boxes_xywh, obj_ids)
    predictor.add_prompt(
        inference_state=inference_state,
        frame_idx=frame_idx,
        text_str=text_prompt,
        boxes_xywh=boxes_xywh,
        obj_ids=[obj_id],
        clear_old_points=True,
    )

def add_sam3_xywh_prompt(
    sam_infer,
    inference_state,
    frame_idx,
    obj_id,
    box_xyxy,
    text_prompt,
    prompt_type='text',
    device="cuda",
):
    """
    box_xyxy: [x0, y0, x1, y1] in pixel coords
    converts to normalized XYWH for Sam3VideoInference.add_prompt
    """
    x0, y0, x1, y1 = box_xyxy
    x = x0 #/ W
    y = y0 #/ H
    w = (x1 - x0) #/ W
    h = (y1 - y0) #/ H

    boxes_xywh = torch.tensor([[x, y, w, h]], dtype=torch.float32, device=device)

    if prompt_type == 'text':
        # only text prompt
        sam_infer.add_prompt(
            inference_state=inference_state,
            frame_idx=frame_idx,
            text_str=text_prompt,
            boxes_xywh=None,
            box_labels=None,
        )

    elif prompt_type == 'bbox':
        # only box prompt
        sam_infer.add_prompt(
            inference_state=inference_state,
            frame_idx=frame_idx,
            text_str=None,
            boxes_xywh=boxes_xywh,
            box_labels=[obj_id],
        )
    else:
        # both box and text prompt
        sam_infer.add_prompt(
            inference_state=inference_state,
            frame_idx=frame_idx,
            text_str=text_prompt,
            boxes_xywh=boxes_xywh,
            box_labels=[obj_id],
        )




# ---------------------------
# Main
# ---------------------------
def main(args):
    device = args.device #torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # if device.type == "cuda":
    #     torch.cuda.manual_seed_all(args.seed)

    # choose bbox function
    box_func = get_all_bounding_boxes_with_ccl_box_fix if args.fix_bbox else get_all_bounding_boxes_with_ccl_box_rand

    # transforms for image encoder if you need, but SAM3 tracker will read frames directly
    sam2_transforms = SAM2Transforms(resolution=args.image_size, mask_threshold=0)

    # load dataset
    to_tensor = transforms.Compose([ToTensor()])
    test_loader = load_data(args.data_root, to_tensor, batch_size=1, split=args.split, tv_type=args.tv_type)

    # metrics
    dice_fun, hd95_fun, nsd_fun, asd_fun = get_dice_hd_nsd_asd()

    # build SAM3 video model and predictor (tracker)
    # sam3_model = build_sam3_video_model()
    # predictor = sam3_model.tracker
    # # sometimes examples copy backbone
    # predictor.backbone = sam3_model.detector.backbone
    sam3_model = build_sam3_video_model().to(device)
    # move detector & tracker explicitly to device
    sam3_model.detector = sam3_model.detector.to(device)
    sam3_model.tracker = sam3_model.tracker.to(device)
    predictor = Sam3VideoInference(
    detector=sam3_model.detector,
    tracker=sam3_model.tracker,)
    predictor.to(device)

    save_eval_path = Path(args.pred_save_dir)
    save_eval_path.mkdir(parents=True, exist_ok=True)

    dice_all, hd95_all, nsd_all, asd_all = [], [], [], []
    case_cnt = 0

    for batch in tqdm(test_loader, desc="cases"):
        case_cnt += 1
        caseID = batch["caseID"][0]
        # your dataset shapes — original script used curMR: B C H W D
        curMR, preMR, preGT, curGT = batch["curMR"], batch["preMR"], batch["preGT"], batch["curGT"]
        curMR = curMR.to(device)
        preMR = preMR.to(device)
        curGT = curGT.to(device)
        preGT = preGT.to(device)

        B, C, H, W, D = curMR.shape
        print(f"[{case_cnt}] case {caseID}, slices D={D}")

        # convert to 3-channel numpy volume (D,H,W,3)
        curMR_cat = torch.cat((curMR, preMR, preGT), dim=1)  # Bx3xHxWxD?
        # adjust dims carefully — original used permute to (0, D, H, W, 3)
        curMR_np = torch.permute(curMR_cat, (0, 4, 2, 3, 1)).cpu().numpy()[0]  # (D, H, W, 3)
        curGT_np = torch.permute(curGT, (0, 3, 1, 2)).cpu().numpy()[0]  # (D, H, W)

        # create frames from volume
        frames = volume_to_frames_list(curMR_np)  # list of HxWx3 uint8

        # write frames to temporary folder then call predictor.init_state(video_path=frames_dir)
        with tempfile.TemporaryDirectory(prefix=f"sam3_vframes_{caseID}_") as tmpdir:
            # SAM3 example accepts a folder of jpg frames
            write_frames_to_folder(frames, tmpdir)

            # init inference state with frames folder
            # NOTE: the SAM3 example uses predictor.init_state(video_path=frames_dir)
            # If your SAM3 expects a different arg name, adapt here.
            inference_state = predictor.init_state(resource_path=tmpdir)

            # clear previous points etc.
            # predictor.clear_all_points_in_video(inference_state)

            segs_3d_case = np.zeros_like(curGT_np, dtype=np.uint8)

            # find labels to infer
            label_ids = np.unique(curGT_np)
            label_ids = label_ids[label_ids != 0]  # skip background

            # for each label build per-slice boxes using your box_func
            for label_id in label_ids:
                mask3d = (curGT_np == label_id).astype(np.uint8)
                z_inds = np.unique(np.where(mask3d > 0)[0])
                if z_inds.size == 0:
                    continue

                # create bbox dict: z -> boxes (Nx4) or single 4
                bbox_dict = {}
                for z in z_inds:
                    bbox_xyxy = box_func(mask3d[z, :, :], bbox_shift=args.bbox_shift)
                    bbox_dict[z] = bbox_xyxy

                # We'll add a prompt on one slice for this object then propagate
                # Choose the first slice containing the object as the prompt slice
                prompt_slice = int(z_inds[0])
                # if multiple boxes on prompt slice, we'll add them sequentially as separate obj_ids
                obj_id = int(label_id)  # you may choose other mapping

                # Build box(es) on prompt slice and add them
                box_ori = bbox_dict[prompt_slice]
                # box_ori may be shape (N,4) or (4,)
                if getattr(box_ori, "ndim", 0) > 1:
                    # multiple boxes: choose the largest area one as the prompt by default
                    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in box_ori]
                    largest_idx = int(np.argmax(areas))
                    chosen_box = box_ori[largest_idx]
                else:
                    chosen_box = box_ori

                # # Normalize box to [0..1] coords expected by SAM3
                norm_box = normalize_box_xyxy(chosen_box, W=W, H=H)
                norm_box = np.array(norm_box, dtype=np.float32) #[None, :]  # shape (1,4)

                # Add the box prompt on prompt_slice using SAM3 add_prompt
                add_sam3_xywh_prompt(
                    predictor,
                    inference_state,
                    frame_idx=prompt_slice,
                    obj_id=obj_id,
                    box_xyxy=norm_box,
                    text_prompt= "Tumor",  #"Lesion",
                    prompt_type='both',
                    device=device,
                )
                # now propagate across the video (slices)
                # collect propagated masks in dictionary
                video_segments = {}

                for frame_idx, out in predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=0,
                    max_frame_num_to_track=D,
                    reverse=False,
                ):
                    if out is None:
                        continue

                    obj_ids = out["out_obj_ids"]                 # (N,)
                    binary_masks = out["out_binary_masks"]      # (N, H', W')
                    probs = out["out_probs"]                    # (N,)

                    video_segments[frame_idx] = {}

                    for i, obj_id in enumerate(obj_ids):
                        mask_arr = binary_masks[i].astype(np.uint8)  # already boolean → uint8

                        # resize if needed
                        if mask_arr.shape != (H, W):
                            mask_arr = cv2.resize(
                                mask_arr,
                                (W, H),
                                interpolation=cv2.INTER_NEAREST,
                            )

                        video_segments[frame_idx][int(obj_id)] = mask_arr


                # After propagation, accumulate into segs_3d_case using our obj_id (or out_obj_ids)
                for z in range(D):
                    if z in video_segments and obj_id in video_segments[z]:
                        mask2d = video_segments[z][obj_id]

                        # ✅ force to 2D safely
                        if mask2d.ndim == 3:
                            mask2d = mask2d[0]        # if (1, H, W)
                            # OR: mask2d = mask2d[..., 0]  # if (H, W, 1)

                        segs_3d_case[z][mask2d > 0] = 1
                        # segs_3d_case[z][video_segments[z][obj_id] > 0] = 1

            # evaluation for this case
            pd_tensor = torch.from_numpy(segs_3d_case.astype(np.uint8))[None, None, ...]  # B C D H W style depends on your metrics
            gt_tensor = torch.from_numpy(curGT_np.astype(np.uint8))[None, None, ...]
            # your metric functions might expect other shapes; adapt if necessary
            dice_case = dice_fun(pd_tensor, gt_tensor).numpy()[0, 0]
            hd95_case = hd95_fun(pd_tensor, gt_tensor).numpy()[0, 0]
            nsd_case = nsd_fun(pd_tensor, gt_tensor).numpy()[0, 0]
            asd_case = asd_fun(pd_tensor, gt_tensor).numpy()[0, 0]
            print(f"case {caseID} dice: {dice_case:.4f}")

            # write per-case
            with open(save_eval_path / f"test_each_rand{args.bbox_shift}.txt", "a") as f:
                f.write(f"{caseID}\tdice:{dice_case:.4f}\thd95:{hd95_case:.4f}\tnsd:{nsd_case:.4f}\tasd:{asd_case:.4f}\n")

            dice_all.append(dice_case); hd95_all.append(hd95_case); nsd_all.append(nsd_case); asd_all.append(asd_case)

            # save nifti if requested
            if args.visualize:
                seg_sitk = sitk.GetImageFromArray(segs_3d_case.astype(np.uint8))
                seg_sitk.SetSpacing((0.89, 0.89, 3.0))
                sitk.WriteImage(seg_sitk, join(args.pred_save_dir, f"{caseID}_pd_sam3_vid.nii.gz"))

    # # summary stats
    # print("Mean dice:", np.mean(dice_all))
    # with open(save_eval_path / f"summary_rand{args.bbox_shift}.txt", "a") as f:
    #     f.write(f"mean_dice\t{np.mean(dice_all):.4f}\tstd\t{np.std(dice_all):.4f}\n")

    ## save format results
    with open(os.path.join(save_eval_path, "format_test_mean_dice_hd_nsd_asd"+str(args.bbox_shift)+".txt"), "a") as f:
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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="/home/gxu/proj1/lesionSeg/sam3_brain")
    p.add_argument("--pred_save_dir", type=str, default="/home/gxu/proj1/lesionSeg/sam3_brain/results/brain_cns/sam3_3ch_video_both_only")
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
