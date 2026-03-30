# ap: use SAM3Transforms without mean/std normalization
# input image is in [0,1]; processor handles normalization internally

import os
import sys
import argparse
import shutil
from os.path import join
from os import makedirs

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm

# ---------------- Path setup ----------------
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2")
sys.path.append("/home/gxu/proj1/lesionSeg/sam3_brain")
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2")

# ---------------- Project modules ----------------
from ap_knife.sam_data_transforms import SAM3Transforms
from ap_knife.eval_metrics import (
    dice_score,
    iou_score,
    detection_accuracy_2d,
    detection_accuracy_3d,
    get_dice_hd_nsd_asd,
)
from ap_knife.get_prompts import (
    get_all_bounding_boxes_with_ccl_box_fix,
    get_all_bounding_boxes_with_ccl_box_rand,
)

from data_load.all_data_loader import all_testsets

# ---------------- SAM3 ----------------
import sam3
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import  normalize_bbox

# =================================================
#  Environment
# =================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

torch.set_float32_matmul_precision("high")
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

# =================================================
#  Args
# =================================================
parser = argparse.ArgumentParser(description="Run inference with SAM3")

parser.add_argument("-data_root", type=str, default=None)
parser.add_argument(
    "-pred_save_dir",
    type=str,
    default="/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/check_results",
)
parser.add_argument(
    "-sam2_checkpoint",
    type=str,
    default="/home/gxu/proj1/lesionSeg/utswlits3d_v2/checkpoint/SAM/sam2_hiera_tiny.pt",
)
parser.add_argument("-model_cfg", type=str, default="sam2_hiera_t.yaml")
parser.add_argument("-device", type=str, default="cuda:1")
parser.add_argument("-bbox_shift", type=int, default=5)
parser.add_argument("-num_workers", type=int, default=16)
parser.add_argument("--visualize", default=True)

args = parser.parse_args()

# =================================================
#  Utils
# =================================================
@torch.no_grad()
def load_sam3(checkpoint, device):
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device=device,
        enable_inst_interactivity=True,
        checkpoint_path=checkpoint,
        eval_mode=True,
    )
    return model


def to_three_channel(x: torch.Tensor):
    return torch.cat([x, x, x], dim=1)


def to_numpy_dhw(x: torch.Tensor):
    return torch.permute(x, (0, 4, 2, 3, 1)).cpu().numpy()[0]


def get_loader_for_testset(testset, root):
    name, builder = testset[0], testset[1]

    kwargs = dict(
        root_path=root,
        transform_policy=None,
        batch_size=1,
        split="test",
    )

    if name in ["brats"]:
        kwargs["modality_type"] = "v2"
    elif name in ["busi", "kvasir", "isic"]:
        kwargs["modality_type"] = testset[3]
    else:
        kwargs["modality_type"] = testset[3]
        kwargs["use_v2"] = True

    loader = builder(**kwargs)
    print(f"Loaded test set: {name}")
    return loader


# =================================================
#  Main config
# =================================================
device = args.device
root_path = "/home/gxu/proj1/lesionSeg/sam3_brain"
model_path = join(root_path, "checkpoint")
save_root_path = join(model_path, "use_all_bbox")
sam3_checkpoint = join(model_path, "sam3.pt")

sam3_transforms = SAM3Transforms(resolution=512, mask_threshold=0)

flag_fix = False
box_func = (
    get_all_bounding_boxes_with_ccl_box_fix
    if flag_fix
    else get_all_bounding_boxes_with_ccl_box_rand
)

# ---------------- Load model ----------------
model = load_sam3(sam3_checkpoint, device)
state_dict = torch.load(sam3_checkpoint, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

processor = Sam3Processor(model, confidence_threshold=0.5, device=device)

dice_fun, hd95_fun, nsd_fun, asd_fun = get_dice_hd_nsd_asd()

# =================================================
#  Inference
# =================================================
for testset in all_testsets():

    testLoader = get_loader_for_testset(testset, root_path)
    set_name = testset[0]
    if set_name in ["brats", "isles", "colon", "kidney", "liver", ]: # "lung", "pancreas"
        continue

    for NUM in range(0, 11, 1):
        args.bbox_shift = NUM
        print(f"\nTesting with bbox shift: {NUM}")

        save_res_path = join(save_root_path, f"shift_{NUM}")
        save_eval_path = join(save_res_path, f"{set_name}/eval_result")
        pred_save_dir = join(save_res_path, "seg_result")

        makedirs(save_eval_path, exist_ok=True)
        makedirs(pred_save_dir, exist_ok=True)

        # backup script
        script_path = join(save_eval_path, os.path.basename(__file__))
        if not os.path.exists(script_path):
            shutil.copyfile(__file__, script_path)

        # metrics
        dice_all, hd95_all, nsd_all, asd_all = [], [], [], []
        iou_all, acc2d_all, acc3d_all = [], [], []

        cnt_case = 0

        for data in testLoader:
            cnt_case += 1

            # -------- Dataset-specific loading --------
            if set_name == "brats":
                curInput, curGT = data["curMR"], data["curGT"]

            elif set_name == "isles":
                curInput, curGT = data["dwi"], data["curGT"]

            elif set_name in ["colon", "kidney", "liver", "lung", "pancreas"]:
                curInput, curGT = data["curCT"], data["curGT"]

            elif set_name == "sts":
                curInput, curPT, curGT = data["curCT"], data["curPT"], data["curGT"]
                curInput = torch.cat([curInput, curPT, torch.zeros_like(curPT)], dim=1)

            else:
                raise ValueError(f"Unknown dataset {set_name}")

            caseID = data["caseID"][0]

            curInput = curInput.to(device)
            curGT = curGT.to(device)
            # exclude no lesion case or outlier case
            if curGT.sum() < 2:
                continue

            B, C, H, W, D = curInput.shape
            print(f"Case {cnt_case} slices: {D}")

            curGT_np = torch.permute(curGT, (0, 1, 4, 2, 3)).cpu().numpy()[0, 0]
            if C == 1:
                curInput = to_three_channel(curInput)
            curInput_np = to_numpy_dhw(curInput)

            # -------- Bounding boxes --------
            label_ids = np.unique(curGT_np)[1:]

            segs_3d = np.zeros_like(curGT_np, dtype=np.uint8)

            for label_id in label_ids:
                marker = (curGT_np == label_id).astype(np.uint8)
                zs = np.sort(np.unique(np.where(marker > 0)[0]))

                bbox_dict = {
                    z: box_func(marker[z], bbox_shift=args.bbox_shift)
                    for z in zs
                }

                # -------- Slice inference --------
                for z in zs:
                    img = curInput_np[z]

                    img_tensor = sam3_transforms(img).to(device)

                    state = processor.set_image(img_tensor)

                    boxes = bbox_dict[z]

                    if boxes.ndim == 1:
                        boxes = boxes[None, :]

                    for box in boxes:
                        box_512 = box / np.array([W, H, W, H]) * 512

                        masks, scores, _ = model.predict_inst(
                            state,
                            box=box_512[None, :],
                            multimask_output=True,
                        )

                        best = masks[np.argsort(scores)[::-1][0]]

                        if best.shape != (H, W):
                            best = cv2.resize(
                                best, (H, W), interpolation=cv2.INTER_NEAREST
                            )

                        segs_3d[z, best > 0] = 1

            # -------- Metrics --------
            pd = torch.from_numpy(segs_3d)[None, None]
            gt = torch.from_numpy(curGT_np)[None, None]

            dice = dice_fun(pd, gt).numpy()[0, 0]
            hd95 = hd95_fun(pd, gt).numpy()[0, 0]
            nsd = nsd_fun(pd, gt).numpy()[0, 0]
            asd = asd_fun(pd, gt).numpy()[0, 0]
            iou = iou_score(segs_3d > 0, curGT_np > 0)

            # detection acc
            acc2d = []
            for i in range(segs_3d.shape[0]):
                if curGT_np[i].sum() > 5:
                    a, _, _ = detection_accuracy_2d(
                        curGT_np[i], segs_3d[i]
                    )
                    acc2d.append(a)

            acc2d = np.mean(acc2d)
            acc3d, _, _ = detection_accuracy_3d(curGT_np, segs_3d)

            # log
            with open(
                join(save_eval_path, f"test_each_metrics_rand{NUM}.txt"), "a"
            ) as f:
                f.write(
                    f"{caseID}\t{dice:.4f}\t{hd95:.4f}\t{nsd:.4f}\t{asd:.4f}"
                    f"\t{iou:.4f}\t{acc2d:.4f}\t{acc3d:.4f}\n"
                )

            dice_all.append(dice)
            hd95_all.append(hd95)
            nsd_all.append(nsd)
            asd_all.append(asd)
            iou_all.append(iou)
            acc2d_all.append(acc2d)
            acc3d_all.append(acc3d)

        # -------- Summary --------
        def stat(x):
            return f"{np.mean(x):.4f}±{np.std(x):.4f}"

        with open(join(save_eval_path, "test_mean_all_metrics.txt"), "a") as f:
            f.write(
                f"Dice {stat(dice_all)}\n"
                f"NSD  {stat(nsd_all)}\n"
                f"HD95 {stat(hd95_all)}\n"
                f"ASD  {stat(asd_all)}\n"
                f"IoU  {stat(iou_all)}\n"
                f"Acc2D {stat(acc2d_all)}\n"
                f"Acc3D {stat(acc3d_all)}\n"
            )

        shutil.copyfile(
            join(save_eval_path, "test_mean_all_metrics.txt"),
            join(save_res_path, f"{set_name}_format_test_mean_all_metrics.txt"),
        )
# ap: use SAM3Transforms without mean/std normalization
# input image is in [0,1]; processor handles normalization internally

import os
import sys
import argparse
import shutil
from os.path import join
from os import makedirs

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm

# ---------------- Path setup ----------------
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2")
sys.path.append("/home/gxu/proj1/lesionSeg/sam3_brain")
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2")

# ---------------- Project modules ----------------
from ap_knife.sam_data_transforms import SAM3Transforms
from ap_knife.eval_metrics import (
    dice_score,
    iou_score,
    detection_accuracy_2d,
    detection_accuracy_3d,
    get_dice_hd_nsd_asd,
)
from ap_knife.get_prompts import (
    get_all_bounding_boxes_with_ccl_box_fix,
    get_all_bounding_boxes_with_ccl_box_rand,
)

from data_load.all_data_loader import all_testsets

# ---------------- SAM3 ----------------
import sam3
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import normalize_bbox

# =================================================
#  Environment
# =================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

torch.set_float32_matmul_precision("high")
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

# =================================================
#  Args
# =================================================
parser = argparse.ArgumentParser(description="Run inference with SAM3")

parser.add_argument("-data_root", type=str, default=None)
parser.add_argument(
    "-pred_save_dir",
    type=str,
    default="/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/check_results",
)
parser.add_argument(
    "-sam2_checkpoint",
    type=str,
    default="/home/gxu/proj1/lesionSeg/utswlits3d_v2/checkpoint/SAM/sam2_hiera_tiny.pt",
)
parser.add_argument("-model_cfg", type=str, default="sam2_hiera_t.yaml")
parser.add_argument("-device", type=str, default="cuda:1")
parser.add_argument("-bbox_shift", type=int, default=5)
parser.add_argument("-num_workers", type=int, default=16)
parser.add_argument("--visualize", default=True)

args = parser.parse_args()

# =================================================
#  Utils
# =================================================
@torch.no_grad()
def load_sam3(checkpoint, device):
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device=device,
        enable_inst_interactivity=True,
        checkpoint_path=checkpoint,
        eval_mode=True,
    )
    return model


def to_three_channel(x: torch.Tensor):
    return torch.cat([x, x, x], dim=1)


def to_numpy_dhw(x: torch.Tensor):
    return torch.permute(x, (0, 4, 2, 3, 1)).cpu().numpy()[0]


def get_loader_for_testset(testset, root):
    name, builder = testset[0], testset[1]

    kwargs = dict(
        root_path=root,
        transform_policy=None,
        batch_size=1,
        split="test",
    )

    if name in ["brats"]:
        kwargs["modality_type"] = "v2"
    elif name in ["busi", "kvasir", "isic"]:
        kwargs["modality_type"] = testset[3]
    else:
        kwargs["modality_type"] = testset[3]
        kwargs["use_v2"] = True

    loader = builder(**kwargs)
    print(f"Loaded test set: {name}")
    return loader


# =================================================
#  Main config
# =================================================
device = args.device
root_path = "/home/gxu/proj1/lesionSeg/sam3_brain"
model_path = join(root_path, "checkpoint")
save_root_path = join(model_path, "use_all_bbox")
sam3_checkpoint = join(model_path, "sam3.pt")

sam3_transforms = SAM3Transforms(resolution=512, mask_threshold=0)

flag_fix = False
box_func = (
    get_all_bounding_boxes_with_ccl_box_fix
    if flag_fix
    else get_all_bounding_boxes_with_ccl_box_rand
)

# ---------------- Load model ----------------
model = load_sam3(sam3_checkpoint, device)
state_dict = torch.load(sam3_checkpoint, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

processor = Sam3Processor(model, confidence_threshold=0.5, device=device)

dice_fun, hd95_fun, nsd_fun, asd_fun = get_dice_hd_nsd_asd()

# =================================================
#  Inference
# =================================================
for testset in all_testsets():

    testLoader = get_loader_for_testset(testset, root_path)
    set_name = testset[0]

    for NUM in range(1, 12, 1):
        print(NUM)
        args.bbox_shift = NUM
        print(f"\nTesting with bbox shift: {NUM}")

        save_res_path = join(save_root_path, f"shift_{NUM}")
        save_eval_path = join(save_res_path, f"{set_name}/eval_result")
        pred_save_dir = join(save_res_path, "seg_result")

        makedirs(save_eval_path, exist_ok=True)
        makedirs(pred_save_dir, exist_ok=True)

        # backup script
        script_path = join(save_eval_path, os.path.basename(__file__))
        if not os.path.exists(script_path):
            shutil.copyfile(__file__, script_path)

        # metrics
        dice_all, hd95_all, nsd_all, asd_all = [], [], [], []
        iou_all, acc2d_all, acc3d_all = [], [], []

        cnt_case = 0

        for data in testLoader:
            cnt_case += 1

            # -------- Dataset-specific loading --------
            if set_name == "brats":
                curInput, curGT = data["curMR"], data["curGT"]

            elif set_name == "isles":
                curInput, curGT = data["dwi"], data["curGT"]

            elif set_name in ["colon", "kidney", "liver", "lung", "pancreas"]:
                curInput, curGT = data["curCT"], data["curGT"]

            elif set_name == "sts":
                curInput, curGT = data["curCT"], data["curGT"]

            else:
                raise ValueError(f"Unknown dataset {set_name}")

            caseID = data["caseID"][0]

            curInput = curInput.to(device)
            curGT = curGT.to(device)
            # exclude no lesion case or outlier case
            if curGT.sum() < 2:
                continue

            B, C, H, W, D = curInput.shape
            print(f"Case {cnt_case} slices: {D}")

            curGT_np = torch.permute(curGT, (0, 1, 4, 2, 3)).cpu().numpy()[0, 0]
            curInput = to_three_channel(curInput)
            curInput_np = to_numpy_dhw(curInput)

            # -------- Bounding boxes --------
            label_ids = np.unique(curGT_np)[1:]

            segs_3d = np.zeros_like(curGT_np, dtype=np.uint8)

            for label_id in label_ids:
                marker = (curGT_np == label_id).astype(np.uint8)
                zs = np.sort(np.unique(np.where(marker > 0)[0]))

                bbox_dict = {
                    z: box_func(marker[z], bbox_shift=args.bbox_shift)
                    for z in zs
                }

                # -------- Slice inference --------
                for z in zs:
                    img = curInput_np[z]

                    img_tensor = sam3_transforms(img).to(device)

                    state = processor.set_image(img_tensor)

                    boxes = bbox_dict[z]

                    if boxes.ndim == 1:
                        boxes = boxes[None, :]

                    for box in boxes:
                        box_512 = box / np.array([W, H, W, H]) * 512

                        masks, scores, _ = model.predict_inst(
                            state,
                            box=box_512[None, :],
                            multimask_output=True,
                        )

                        best = masks[np.argsort(scores)[::-1][0]]

                        if best.shape != (H, W):
                            best = cv2.resize(
                                best, (H, W), interpolation=cv2.INTER_NEAREST
                            )

                        segs_3d[z, best > 0] = 1

            # -------- Metrics --------
            pd = torch.from_numpy(segs_3d)[None, None]
            gt = torch.from_numpy(curGT_np)[None, None]

            dice = dice_fun(pd, gt).numpy()[0, 0]
            hd95 = hd95_fun(pd, gt).numpy()[0, 0]
            nsd = nsd_fun(pd, gt).numpy()[0, 0]
            asd = asd_fun(pd, gt).numpy()[0, 0]
            iou = iou_score(segs_3d > 0, curGT_np > 0)

            # detection acc
            acc2d = []
            for i in range(segs_3d.shape[0]):
                if curGT_np[i].sum() > 5:
                    a, _, _ = detection_accuracy_2d(
                        curGT_np[i], segs_3d[i]
                    )
                    acc2d.append(a)

            acc2d = np.mean(acc2d)
            acc3d, _, _ = detection_accuracy_3d(curGT_np, segs_3d)

            # log
            with open(
                join(save_eval_path, f"test_each_metrics_rand{NUM}.txt"), "a"
            ) as f:
                f.write(
                    f"{caseID}\t{dice:.4f}\t{hd95:.4f}\t{nsd:.4f}\t{asd:.4f}"
                    f"\t{iou:.4f}\t{acc2d:.4f}\t{acc3d:.4f}\n"
                )

            dice_all.append(dice)
            hd95_all.append(hd95)
            nsd_all.append(nsd)
            asd_all.append(asd)
            iou_all.append(iou)
            acc2d_all.append(acc2d)
            acc3d_all.append(acc3d)

        # -------- Summary --------
        def stat(x):
            return f"{np.mean(x):.4f}±{np.std(x):.4f}"

        with open(join(save_eval_path, "test_mean_all_metrics.txt"), "a") as f:
            f.write(
                f"Dice {stat(dice_all)}\n"
                f"NSD  {stat(nsd_all)}\n"
                f"HD95 {stat(hd95_all)}\n"
                f"ASD  {stat(asd_all)}\n"
                f"IoU  {stat(iou_all)}\n"
                f"Acc2D {stat(acc2d_all)}\n"
                f"Acc3D {stat(acc3d_all)}\n"
            )

        shutil.copyfile(
            join(save_eval_path, "test_mean_all_metrics.txt"),
            join(save_res_path, f"{set_name}_format_test_bbox_shift_{NUM}.txt"),
        )
