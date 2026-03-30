# ============================================================
#  Generalized SAM3 Prior-Enhanced Evaluation Script (FIXED)
#  - Dataset-specific save paths
#  - Dataset-specific loaders
# ============================================================

import sys, os, argparse, shutil
import numpy as np
import torch
import cv2
import SimpleITK as sitk
from os.path import join
from os import makedirs
from tqdm import tqdm

# ---------------- PATHS ----------------
sys.path.extend([
    "/home/gxu/proj1/lesionSeg/utswlits3d_v2",
    "/home/gxu/proj1/lesionSeg/sam3_brain",
    "/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2",
])

# ---------------- SAM3 ----------------
import sam3
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox

# ---------------- DATA / METRICS ----------------
from ap_knife.sam_data_transforms import SAM3Transforms
from ap_knife.eval_metrics import (
    iou_score,
    detection_accuracy_2d,
    detection_accuracy_3d,
    get_dice_hd_nsd_asd,
)
from ap_knife.get_prompts import get_all_bounding_boxes_with_ccl_box_rand

# ---------------- LOADERS ----------------
from data_load.all_data_loader import all_utsw_set

# ============================================================
# DATASET REGISTRY
# ============================================================

DATASET_REGISTRY = all_utsw_set()

# ============================================================
# MODEL
# ============================================================

@torch.no_grad()
def load_sam3(checkpoint, device):
    model = build_sam3_image_model(
        bpe_path=f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz",
        device=device,
        checkpoint_path=checkpoint,
        enable_inst_interactivity=True,
        eval_mode=True,
    )
    return model


def norm_box(box, W, H, device):
    box_512 = box / np.array([W, H, W, H]) * 512
    x1, y1, x2, y2 = box_512
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    box = torch.tensor([[cx, cy, w, h]], device=device)
    return normalize_bbox(box, 512, 512).flatten().tolist()


def make_dataset_paths(save_root, dataset_name, tag="sam3_text_exemplar"):
    base = join(save_root, dataset_name, tag)
    paths = {
        "base": base,
        "eval": join(base, "eval_result"),
        "pred": join(base, "seg_result"),
    }
    for p in paths.values():
        makedirs(p, exist_ok=True)
    return paths


# ============================================================
# MAIN
# ============================================================

def main():
    device = "cuda:1"
    USE_TEXT = True
    USE_EXEMP = True
    VISUALIZE = False
    BOX_NUM = 1
    np.random.seed(0)  # ensure deterministic bbox selection

    sam3_ckpt = "/home/gxu/proj1/lesionSeg/sam3_brain/checkpoint/sam3.pt"
    save_root = "/home/gxu/proj1/lesionSeg/sam3_brain/checkpoint/use_premask"

    # ---------------- Load model ----------------
    model = load_sam3(sam3_ckpt, device)
    model.load_state_dict(torch.load(sam3_ckpt, map_location=device), strict=False)
    model.eval()

    processor = Sam3Processor(model, confidence_threshold=0.5, device=device)
    sam3_tf = SAM3Transforms(resolution=512, mask_threshold=0)
    dice_fun, hd95_fun, nsd_fun, asd_fun = get_dice_hd_nsd_asd()

    # ---------------- Loop datasets ----------------
    for name, loader_fn, io_map, split, tv_type in DATASET_REGISTRY:
        if name in ["busi", "isic", "kvasir"]:
            continue

        print(f"\n===== Testing {name.upper()} =====")

        paths = make_dataset_paths(save_root, name)
        save_eval_path = paths["eval"]
        pred_save_dir = paths["pred"]

        loader = loader_fn(
            "/home/gxu/proj1/lesionSeg/sam3_brain",
            batch_size=1,
            split=split,
            transform_policy=None,
            tv_type=tv_type
        )

        dice_all, hd95_all, nsd_all = [], [], []
        asd_all, iou_all = [], []
        acc_2d_all, acc_3d_all = [], []

        # ---------------- Loop cases ----------------
        for batch in tqdm(loader):
            previous_pred = None  # reset per case

            img_tensors = [batch[k].to(device) for k in io_map["image"]]
            gt = batch[io_map["mask"]].to(device)
            caseID = str(batch["caseID"][0])

            img_np = [torch.permute(t, (0, 3, 1, 2)).cpu().numpy()[0] for t in img_tensors]
            gt_np = torch.permute(gt, (0, 3, 1, 2)).cpu().numpy()[0]

            segs_3d_temp = np.zeros_like(gt_np, dtype=np.uint8)
            D, H, W = gt_np.shape
            z_ids = np.sort(np.unique(np.where(gt_np > 0)[0]))

            # ---------------- Slice inference ----------------
            for z in z_ids:
                base = img_np[0][z, ...]
                prior = np.zeros_like(base) if previous_pred is None else previous_pred

                img_3c = np.stack([base, base, prior], axis=-1)
                img_tensor = sam3_tf(img_3c).to(device)

                # inference
                inference_state = processor.set_image(img_tensor)
                processor.reset_all_prompts(inference_state)
                if USE_TEXT:
                    inference_state = processor.set_text_prompt(state=inference_state, prompt="lesion")


                box_ori = get_all_bounding_boxes_with_ccl_box_rand(gt_np[z], bbox_shift=0)

                if box_ori.ndim > 1:
                    rand_idx = np.random.randint(0, box_ori.shape[0], BOX_NUM)
                    # rand_idx = [0]  # always use the first box (baseline behavior)
                    for idx in rand_idx:
                        if USE_EXEMP:
                            inference_state = processor.add_geometric_prompt(
                                state=inference_state,
                                box=norm_box(box_ori[idx], W, H, device),
                                label=True,
                            )
                else:
                    if USE_EXEMP:
                        inference_state = processor.add_geometric_prompt(
                            state=inference_state,
                            box=norm_box(box_ori, W, H, device),
                            label=True,
                        )

                nb_objects = len(inference_state["scores"])
                for n in range(nb_objects):
                    seg = inference_state["masks"][n][0].cpu().numpy().astype(np.uint8)
                    if seg.shape != (H, W):
                        seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)
                    segs_3d_temp[z, seg > 0] = 1

                previous_pred = (segs_3d_temp[z] > 0).astype(np.float32)

            # ---------------- Metrics ----------------
            pd = torch.from_numpy(segs_3d_temp)[None, None]
            gt_t = torch.from_numpy(gt_np)[None, None]

            dice_case = dice_fun(pd, gt_t).numpy()[0, 0]
            hd95_case = hd95_fun(pd, gt_t).numpy()[0, 0]
            nsd_case = nsd_fun(pd, gt_t).numpy()[0, 0]
            asd_case = asd_fun(pd, gt_t).numpy()[0, 0]
            iou_case = iou_score(segs_3d_temp > 0, gt_np > 0)

            det2d = [
                detection_accuracy_2d(gt_np[i], segs_3d_temp[i])[0]
                for i in range(D) if gt_np[i].sum() > 5
            ]
            det_acc_2d = np.mean(det2d)
            det_acc_3d, _, _ = detection_accuracy_3d(gt_np, segs_3d_temp)
            print(f"case {caseID} dice: {dice_case}")

            # ---------------- Save per-case ----------------
            with open(join(save_eval_path, "test_each_dice_hd95_nsd_asd_rand0.txt"), "a") as f:
                f.write(
                    f"{caseID} dice:\t{dice_case:.4f}\thd95:\t{hd95_case:.4f}"
                    f"\tnsd:\t{nsd_case:.4f}\tasd:\t{asd_case:.4f}"
                    f"\tiou:\t{iou_case:.4f}\tacc_2d:\t{det_acc_2d:.4f}"
                    f"\tacc_3d:\t{det_acc_3d:.4f}\n"
                )

            dice_all.append(dice_case)
            hd95_all.append(hd95_case)
            nsd_all.append(nsd_case)
            asd_all.append(asd_case)
            iou_all.append(iou_case)
            acc_2d_all.append(det_acc_2d)
            acc_3d_all.append(det_acc_3d)

            if VISUALIZE:
                sitk.WriteImage(
                    sitk.GetImageFromArray(segs_3d_temp),
                    join(pred_save_dir, f"{caseID}_pd_sam3.nii.gz"),
                )
                sitk.WriteImage(
                    sitk.GetImageFromArray(gt_np),
                    join(pred_save_dir, f"{caseID}_gt_sam3.nii.gz"),
                )

        # ---------------- Save mean metrics ----------------
        with open(join(save_eval_path, "format_test_mean_all_metrics.txt"), "a") as f:
            f.write(
                f"{'#'*60}\n"
                f"{np.mean(dice_all):.4f}±{np.std(dice_all):.4f}\t"
                f"{np.mean(nsd_all):.4f}±{np.std(nsd_all):.4f}\t"
                f"{np.mean(hd95_all):.4f}±{np.std(hd95_all):.4f}\t"
                f"{np.mean(asd_all):.4f}±{np.std(asd_all):.4f}\t"
                f"{np.mean(iou_all):.4f}±{np.std(iou_all):.4f}\t"
                f"{np.mean(acc_2d_all):.4f}±{np.std(acc_2d_all):.4f}\t"
                f"{np.mean(acc_3d_all):.4f}±{np.std(acc_3d_all):.4f}\n"
            )
        # save formatted results for all metrics
        with open(os.path.join(save_eval_path, "format_test_mean_all_metrics.txt"), "a") as f:
            write_content = f"{'#'*60}\nmean dice\tmean nsd\tmean hd95\tmean asd\tmean iou\tmean acc_2d\tmean acc_3d\n"
            write_content += f"{np.array(dice_all).mean():.4f}±{np.array(dice_all).std():.4f}\t" \
                            f"{np.array(nsd_all).mean():.4f}±{np.array(nsd_all).std():.4f}\t" \
                            f"{np.array(hd95_all).mean():.4f}±{np.array(hd95_all).std():.4f}\t" \
                            f"{np.array(asd_all).mean():.4f}±{np.array(asd_all).std():.4f}\t" \
                            f"{np.array(iou_all).mean():.4f}±{np.array(iou_all).std():.4f}\t" \
                            f"{np.array(acc_2d_all).mean():.4f}±{np.array(acc_2d_all).std():.4f}\t" \
                            f"{np.array(acc_3d_all).mean():.4f}±{np.array(acc_3d_all).std():.4f}\t\n"
            write_content += f"{np.array(dice_all).mean()*100:.2f}\t" \
                            f"{np.array(nsd_all).mean():.2f}\t" \
                            f"{np.array(hd95_all).mean():.2f}\t" \
                            f"{np.array(asd_all).mean():.2f}\t" \
                            f"{np.array(iou_all).mean()*100:.2f}\t" \
                            f"{np.array(acc_2d_all).mean()*100:.2f}\t" \
                            f"{np.array(acc_3d_all).mean()*100:.2f}\n"
            f.write(write_content)

        print(f"Results saved to {save_eval_path}")


if __name__ == "__main__":
    main()
