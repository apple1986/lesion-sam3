#p: test in t1c, t1c+t2f, t1c+t2f+t1n
import sys
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2")
sys.path.append("/home/gxu/proj1/lesionSeg/sam3_brain")
# sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2")
from medpy.metric.binary import true_negative_rate
import numpy as np
from os.path import join
from os import makedirs, listdir
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
import SimpleITK as sitk
import os
import argparse
import shutil

from ap_knife.sam_data_transforms import SAM3Transforms
from ap_knife.save_data import save_rgb_image, save_mask
from ap_knife.eval_metrics import dice_score, iou_score, detection_accuracy_2d, detection_accuracy_3d

from torch import multiprocessing as mp
from torchvision import transforms
from data_load.busi_builder import load_data_busi
from gp_knife.get_prompts import get_all_bounding_boxes_with_ccl_box_fix, get_all_bounding_boxes_with_ccl_box_rand, get_all_bounding_boxes_with_ccl
from gp_knife.eval_model import get_dice_hd_nsd_asd
from medpy.metric import dc
import cv2
import numpy as np
from skimage.exposure import match_histograms
## sam3
import sam3
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results
from sam3.model.box_ops import box_xywh_to_cxcywh


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# use bfloat16 for the enitre model to reduce memory consumption
torch.autocast(device_type="cuda:1", dtype=torch.bfloat16).__enter__()

## set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)
###########################
## set parameters
parser = argparse.ArgumentParser(
    description="Run inference on validation set with MedSAM2"
)
parser.add_argument(
    "-data_root",
    type=str,
    default=None,
    help="Path to the data folder",
)
parser.add_argument(
    "-pred_save_dir",
    type=str,
    default="/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/check_results",
    help="Path to save the segmentation results",
)
parser.add_argument(
    "-sam2_checkpoint",
    type=str,
    default='/home/gxu/proj1/lesionSeg/utswlits3d_v2/checkpoint/SAM/sam2_hiera_tiny.pt',
    help="SAM2 pretrained model checkpoint",
)
parser.add_argument(
    "-model_cfg",
    type=str,
    default="sam2_hiera_t.yaml",
    help="Model config file"
)
parser.add_argument("-device", type=str, default="cuda:1", help="device")
parser.add_argument(
    "-bbox_shift",
    type=int,
    default=5,
    help="Bounding box perturbation",
)
parser.add_argument(
    "-num_workers",
    type=int,
    default=16,
    help="Number of workers for multiprocessing",
)
parser.add_argument("--visualize", default=True, help="Save the .nii.gz slice segmentation results")
args = parser.parse_args()


# ============================================================
# Utility functions
# ============================================================
def hist_match_ultrasound(src_img, ref_img):
    """
    Match histogram of src_img to ref_img
    Both inputs: H×W×3 uint8
    """
    # Convert to grayscale
    src_gray = src_img[:,:,0] #cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    ref_gray = ref_img[:,:,0] #cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    # Histogram matching
    matched = match_histograms(src_gray, ref_gray)

    # Convert back to 3-channel
    matched_3c = np.stack([matched]*3, axis=-1) #.astype(np.uint8)
    return matched_3c

def crop_by_bbox(img, mask, box, enlarge_ratio=0.0):
    """
    Crop image and mask using an enlarged bounding box.

    Args:
        img: H×W×C numpy array
        mask: H×W numpy array
        box: [x1, y1, x2, y2]
        enlarge_ratio: fraction of image size used for enlargement

    Returns:
        cropped_img, cropped_mask
    """
    H, W = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    # scale enlargement with image size
    enlarge_x = int(W * enlarge_ratio)
    enlarge_y = int(H * enlarge_ratio)

    x1 = max(0, x1 - enlarge_x)
    y1 = max(0, y1 - enlarge_y)
    x2 = min(W, x2 + enlarge_x)
    y2 = min(H, y2 + enlarge_y)

    # safety check
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox after enlargement: {x1,y1,x2,y2}")

    return img[y1:y2, x1:x2], mask[y1:y2, x1:x2]


def resize_exemplar(img, mask, resize=32):
    img32 = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_LINEAR)
    mask32 = cv2.resize(mask, (resize, resize), interpolation=cv2.INTER_NEAREST)
    return img32, mask32

@torch.no_grad()
## load model: sam3
def load_sam3(sam3_checkpoint=None, device="cuda:1"):
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=bpe_path, 
                                   device=device,
                                   enable_inst_interactivity=True, # False
                                   checkpoint_path=sam3_checkpoint,
                                   eval_mode=True)

    return model

@torch.no_grad()
def extract_sam_feature(img, model, device):
    """
    img: H×W (single-channel US)
    """
    # img_3c = np.stack([img, img, img], axis=-1)
    img_t = torch.from_numpy(img).float().permute(2, 0, 1)[None].to(device)
    img_t = F.interpolate(img_t, size=(1008, 1008), mode="bilinear")
    with torch.no_grad():
        feats_dict = model.backbone(img_t, "lesion")   # (1, C, H', W')
    feats = feats_dict['vision_features'] # 1x256x72x72 # dict_keys(['vision_features', 'vision_pos_enc', 'backbone_fpn', 'sam2_backbone_out', 'language_features', 'language_mask', 'language_embeds'])
    # feats = feats_dict["sam2_backbone_out"]["vision_features"] # dict_keys(['vision_features', 'vision_pos_enc', 'backbone_fpn'])
    # feat = feats.mean(dim=(2, 3))         # Global descriptor 1x256
    feat = F.normalize(feats, dim=1)
    # feat = feats.mean(dim=(0, 1))         # Global descriptor 72x72
    # feat = F.normalize(feat, dim=(0, 1))

    return feat

@torch.no_grad()
def extract_exemplar_prototype(img, mask, model, device):
    """
    img: H×W×3 numpy
    mask: H×W binary numpy
    return: (1, C) normalized lesion prototype
    """
    feats = extract_sam_feature(img, model, device)  # 1×C×Hf×Wf
    _, C, Hf, Wf = feats.shape

    mask_t = torch.from_numpy(mask).float().to(device)
    mask_t = F.interpolate(mask_t[None,None], size=(Hf, Wf), mode="nearest")

    masked_feats = feats * mask_t
    proto = masked_feats.sum(dim=(2,3)) / (mask_t.sum() + 1e-6)
    proto = F.normalize(proto, dim=1)

    return proto


@torch.no_grad()
def find_similar_region_conv(
    target_img,
    exemplar_proto,
    model,
    device,
    win_size,
):
    """
    target_img: H×W×3 numpy
    exemplar_proto: (1, C)
    win_size: lesion size in original image (pixels)
    """
    H, W, _ = target_img.shape

    # 1️⃣ Extract target feature map ONCE
    feat_map = extract_sam_feature(target_img, model, device) # 1x256x72x72
    _, C, Hf, Wf = feat_map.shape

    # 2️⃣ Cosine similarity via 1×1 conv
    kernel = exemplar_proto.view(C, 1, 1)
    sim_map = F.conv2d(feat_map, kernel.unsqueeze(0)).squeeze(0).squeeze(0)
    # sim_map: Hf×Wf

    # 3️⃣ Find max response
    max_idx = torch.argmax(sim_map)
    y_f, x_f = divmod(max_idx.item(), Wf)

    # 4️⃣ Map feature coords → image coords
    scale_x = W / Wf
    scale_y = H / Hf

    cx = int((x_f + 0.5) * scale_x)
    cy = int((y_f + 0.5) * scale_y)

    half = win_size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(W, cx + half)
    y2 = min(H, cy + half)

    best_box = [x1, y1, x2, y2]
    best_score = sim_map.max().item()

    return best_box, best_score


def norm_box(box_ori, W, H, device="cuda:1"):

    box_512 = box_ori / np.array([W, H, W, H]) * 512 ## adjust to 512x512 scale
    # convert the xyxy to cxcywh
    x1, y1, x2, y2 = box_512
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w  = x2 - x1
    h  = y2 - y1

    box_input_cxcywh = np.array([cx, cy, w, h])
    box_input_cxcywh = torch.from_numpy(box_input_cxcywh).to(device).view(-1, 4)
    norm_box_cxcywh = normalize_bbox(box_input_cxcywh, 512, 512).flatten().tolist()

    return norm_box_cxcywh

@torch.no_grad()
def find_similar_region_multiscale_topk(
    target_img,
    exemplar_proto,
    model,
    device,
    base_win_size,
    scales=(0.75, 1.0, 1.25),
    topk=5,
):
    """
    Multi-scale cosine similarity search with Top-K boxes.

    Returns:
        boxes: List[[x1,y1,x2,y2]]
        scores: List[float]
        heatmap_img: H×W numpy similarity heatmap
    """
    H, W, _ = target_img.shape

    # -------------------------------------------------
    # 1. Extract target feature map ONCE
    # -------------------------------------------------
    feat_map = extract_sam_feature(target_img, model, device)  # 1×C×Hf×Wf
    _, C, Hf, Wf = feat_map.shape

    # -------------------------------------------------
    # 2. Cosine similarity via 1×1 conv
    # -------------------------------------------------
    kernel = exemplar_proto.view(C, 1, 1)
    sim_map = F.conv2d(feat_map, kernel.unsqueeze(0))  # 1×1×Hf×Wf
    sim_map = sim_map.squeeze()                        # Hf×Wf

    # -------------------------------------------------
    # 3. Top-K locations in feature space
    # -------------------------------------------------
    sim_flat = sim_map.flatten()
    topk_scores, topk_idx = torch.topk(sim_flat, k=topk)

    # -------------------------------------------------
    # 4. Feature → Image coordinate mapping
    # -------------------------------------------------
    scale_x = W / Wf
    scale_y = H / Hf

    boxes = []
    scores = []

    for idx, score in zip(topk_idx, topk_scores):
        y_f = idx // Wf
        x_f = idx % Wf

        cx = int((x_f + 0.5) * scale_x)
        cy = int((y_f + 0.5) * scale_y)

        for s in scales:
            win = int(base_win_size * s)
            half = win // 2

            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(W, cx + half)
            y2 = min(H, cy + half)

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                scores.append(score.item())

    # -------------------------------------------------
    # 5. Heatmap (feature → image resolution)
    # -------------------------------------------------
    # heatmap = sim_map.cpu().numpy()
    heatmap = sim_map.detach().float().cpu().numpy()
    heatmap = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return boxes, scores, heatmap

def visualize_similarity_heatmap(img, heatmap, boxes=None, save_path=None):
    """
    img: H×W×3
    heatmap: H×W
    """
    plt.figure(figsize=(6,6))
    plt.imshow(img, alpha=0.6)
    plt.imshow(heatmap, cmap="jet", alpha=0.4)
    plt.colorbar()

    if boxes is not None:
        for box in boxes:
            x1,y1,x2,y2 = box
            plt.gca().add_patch(
                plt.Rectangle((x1,y1), x2-x1, y2-y1,
                              edgecolor='lime', facecolor='none', linewidth=2)
            )

    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

#######################################
"""
1. extract the cropped lesion mask(bounding box region) and the corresponding image patch for the first mask and first image
2. resize the lesion bbox and the image patch as 32x32
3. copy the the subsquent images and mask, extract this bbox as prompt the the subsquent image inference
4. Exclude the mask area when in evaluation for the subsequent prediction result
"""

## testing
image_size = 512
args.visualize = True
data_root = args.data_root
# bbox_shift = args.bbox_shift
args.bbox_shift = 0
args.noMask = False
device = "cuda:1"  # args.device
args.ch3 = True
flag_fix = False # False True
# tv_types = ['gtv', 'itv', 'ptv', 'alltv']

if flag_fix is True:
    box_func = get_all_bounding_boxes_with_ccl_box_fix 
else:
    box_func = get_all_bounding_boxes_with_ccl_box_rand

root_path = "/home/gxu/proj1/lesionSeg/sam3_brain"
model_path = "/home/gxu/proj1/lesionSeg/sam3_brain/checkpoint"
sam3_checkpoint = os.path.join(model_path, "sam3.pt")
BOX_NUM = 1 # number of box examplars
MODALITY_TYPES = ["Ultrasound"] # 
NUM = 0
USE_TEXT = True
USE_EXEMP = True

if USE_EXEMP and (not USE_TEXT):
    save_res_path = os.path.join(model_path, "use_copymask/busi/sam3_exemplar")
elif (not USE_EXEMP) and USE_TEXT:
    save_res_path = os.path.join(model_path, "use_copymask/busi/sam3_text")
else:
    save_res_path = os.path.join(model_path, "use_copymask/busi/sam3_text_exemplar")

for modality_type in MODALITY_TYPES:
    num_workers = args.num_workers

    save_eval_path = os.path.join(save_res_path, "eval_result")
    makedirs(save_eval_path, exist_ok=True)
    # save segmentation results
    pred_save_dir = os.path.join(save_res_path, "seg_result")
    makedirs(pred_save_dir, exist_ok=True)
    # copy the current testing script
    script_path = join(save_eval_path, os.path.basename(__file__))
    if not os.path.exists(script_path):
        shutil.copyfile(
            __file__, script_path
        )
    args.bbox_shift =  NUM # NUM
    print(f"Testing with bbox shift: {args.bbox_shift}")
    sam3_transforms = SAM3Transforms(resolution=512, mask_threshold=0)

    ## load data
    all_test_transform = None
    testLoader = load_data_busi(root_path, all_test_transform, batch_size=1, split="test", 
                                   modality_type=modality_type, use_v2=True)

    # load evaluation function
    dice_fun, hd95_fun, nsd_fun, asd_fun = get_dice_hd_nsd_asd()
    exemplar_box = None
    exemplar_img32 = None
    exemplar_mask32 = None
    is_first_case = True


    dice_all = [] # save all data
    hd95_all = []
    nsd_all = []
    asd_all = []
    iou_all = []
    acc_2d_all = []
    cnt_case = 0

    ## load model: sam3
    model = load_sam3(sam3_checkpoint=sam3_checkpoint, device=device)
    processor = Sam3Processor(model, confidence_threshold=0.5, device=device)

    for curImg, curGT, caseID in testLoader:
        cnt_case = cnt_case + 1
        print(f"cnt: {cnt_case}")

        # -------- Load data --------

        curImg = curImg.to(device)     # B×3×H×W
        curGT = curGT.to(device)     # B×H×W
        # print(f"Processing case {caseID[0]}, image shape: {curImg.shape}, GT shape: {curGT.shape}")
        # mask_2d = F.interpolate(mask_2d, size=(256, 256), mode="nearest", align_corners=None)

        B, C, H, W = curImg.shape

        # Convert to numpy
        curImg_np = curImg[0].permute(1, 2, 0).cpu().numpy()  # H×W×3
        curGT_np = curGT[0].cpu().numpy()                   # H×W

        # exclude no lesion case
        if curGT.sum() < 2:
            continue

        ## inference
        segs_dict = {}
        label_ids = np.unique(curGT_np)[1:]
        ## Simulate 3D box for each organ
        # segs_2d_temp = np.zeros_like(curGT_np, dtype=np.uint8)
        for label_id in label_ids:
            segs_2d_temp = np.zeros((H, W), dtype=np.uint8)

            # -------------------- EXAMPLAR --------------------
            if is_first_case:
                # binary mask
                mask_bin = (curGT_np == label_id).astype(np.uint8)
                # get bbox from GT (ONLY once)
                z_box_check = box_func(mask_bin, bbox_shift=args.bbox_shift)
                bbox_dict = z_box_check
                box_used = z_box_check[0] if z_box_check.ndim > 1 else z_box_check
                # crop exemplar
                crop_img, crop_mask = crop_by_bbox(curImg_np, curGT_np, box_used, enlarge_ratio=0.20)
                win_size = crop_img.shape[0] # HW3
                # extract feature
                # exemplar_feat = extract_sam_feature(
                #                     crop_img, model, device
                #                 ) # 1x256

                exemplar_feat = extract_exemplar_prototype(
                    crop_img,
                    crop_mask == label_id,
                    model,
                    device
                )

                # best_box, score = find_similar_region_conv(
                #     curImg_np,
                #     exemplar_feat,
                #     model,
                #     device,
                #     win_size=win_size,
                # )

                boxes, scores, heatmap = find_similar_region_multiscale_topk(
                    curImg_np,
                    exemplar_feat,
                    model,
                    device,
                    base_win_size=win_size,
                    scales=(0.75, 1.0, 1.25),
                    topk=5,
                )
                print("Best box:", boxes[0], "score:", scores[0])
                # best_box, score = find_similar_region(
                #                             curImg_np,
                #                             exemplar_feat,
                #                             model,
                #                             device,
                #                             win_size=win_size,
                #                         )
                # print("Best box:", best_box, "score:", score)
                print(f"reference bbox: {box_used}")
                # exemplar_img32, exemplar_mask32 = resize_exemplar(crop_img, crop_mask, resize=128)
                # exemplar_box = box_used.copy()
                is_first_case = False
            else:
                # 1. copy the crop_img, and crop_mask into the top left of the next loaded image/mask
                # 2. obtain the bbox location (the copied cropped mask)
                # 3. when do evaluaion, excluded the crop_mask from the prediction mask
                # -------------------------------------------------
                # 1. Copy exemplar patch into top-left corner
                # -------------------------------------------------
                # h_ex, w_ex = exemplar_img32.shape[:2]

                # curImg_np = curImg_np.copy()
                # # curGT_np = curGT_np.copy()

                # patched_img = hist_match_ultrasound(exemplar_img32, curImg_np)

                # curImg_np[0:h_ex, 0:w_ex, :] = patched_img
                # curImg_np = (curImg_np - curImg_np.min()) / (curImg_np.max() - curImg_np.min() + 1e-8)
                # best_box, score = find_similar_region(
                #                             curImg_np,
                #                             exemplar_feat,
                #                             model,
                #                             device,
                #                             win_size=win_size,
                #                         )

                # best_box, score = find_similar_region_conv(
                #     curImg_np,
                #     exemplar_feat,
                #     model,
                #     device,
                #     win_size=win_size,
                # )                
                # # print("Best box:", best_box, "score:", score)
                # box_used = best_box
                # # curGT_np[0:h_ex, 0:w_ex] = exemplar_mask32

                boxes, scores, heatmap = find_similar_region_multiscale_topk(
                    curImg_np,
                    exemplar_feat,
                    model,
                    device,
                    base_win_size=win_size,
                    scales=(0.75, 1.0, 1.25),
                    topk=5,
                )

                box_used = boxes[0]   # highest-score box
                print("Best box:", box_used, "score:", scores[0])


                # if True:
                #     visualize_similarity_heatmap(
                #         curImg_np,
                #         heatmap,
                #         boxes=boxes[:3],
                #         save_path=os.path.join(
                #             pred_save_dir, f"{caseID[0]}_similarity_heatmap.png"
                #         )
                #     )



                # -------------------------------------------------
                # 2. Obtain bbox from the pasted mask
                # -------------------------------------------------
                # pasted_mask = np.zeros_like(curGT_np, dtype=np.uint8)
                # pasted_mask[0:h_ex, 0:w_ex] = exemplar_mask32
                mask_bin = (curGT_np == label_id).astype(np.uint8)
                # get bbox from GT (ONLY once)
                z_box_check = box_func(mask_bin, bbox_shift=args.bbox_shift)
                print(f"reference bbox: {z_box_check}")
                # box_used = z_box_check[0] if z_box_check.ndim > 1 else z_box_check

            ## inference
            img_2d = curImg_np # HW3
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            # convert the shape to (1, 3, 512, 512)
            # img_512_tensor = sam3_transforms(img_3c)[None, ...].to(device) # 1x3xHxW
            img_512_tensor = sam3_transforms(img_3c).to(device) # 1x3xHxW

            # inference
            inference_state = processor.set_image(img_512_tensor)
            processor.reset_all_prompts(inference_state)
            if USE_TEXT:
                inference_state = processor.set_text_prompt(state=inference_state, prompt="tumor")
            
            ## get bbox
            box_ori = box_used
            norm_box_cxcywh = norm_box(box_ori, W, H, device=device)
            if USE_EXEMP:
                inference_state = processor.add_geometric_prompt(
                        state=inference_state, box=norm_box_cxcywh, label=True
                                )
            nb_objects = len(inference_state["scores"])
            print(f"found {nb_objects} object(s)")
            for n in range(nb_objects):
                img_2d_seg = inference_state["masks"][n][0].cpu().numpy() # HxW
                img_2d_seg = img_2d_seg.astype(np.uint8)
                if segs_2d_temp.shape != img_2d_seg.shape: # do interpolation if size not match
                    img_2d_seg = cv2.resize(img_2d_seg, (W, H), interpolation=cv2.INTER_NEAREST)
                    # print(f"case_cnt: {cnt_case}, Original shape {img_2d_seg.shape}, prediction shape {masks.shape[1:]}") 
                              
                segs_2d_temp[img_2d_seg>0] = 1

            segs_dict[label_id] = segs_2d_temp.copy()

        # cal dice, hd95, nsd and asd
        pd = torch.from_numpy(segs_2d_temp)[None, None, ...] #BCDHW
        gt = torch.from_numpy(curGT_np)[None, None, ...] 
        dice_case = dice_fun(pd, gt).numpy()[0,0]
        hd95_case = hd95_fun(pd, gt).numpy()[0,0]
        nsd_case = nsd_fun(pd, gt).numpy()[0,0]
        asd_case = asd_fun(pd, gt).numpy()[0,0]
        iou_case = iou_score(segs_2d_temp>0, curGT_np>0)
        print(f"case {caseID} dice: {dice_case}")
        det_acc_2d_case, _, _ = detection_accuracy_2d(curGT_np, segs_2d_temp)

        ## save each case results to txt
        with open(os.path.join(save_eval_path, "test_each_dice_hd95_nsd_asd_rand"+str(NUM)+".txt"), "a") as f:
            write_content = f"{caseID} dice:\t{dice_case:.4f}\thd95:\t{hd95_case:.4f}\tnsd:\t{nsd_case:.4f}\tasd:\t{asd_case:.4f}\t\n"
            f.write(write_content)

        dice_all.append(dice_case)
        hd95_all.append(hd95_case)
        nsd_all.append(nsd_case)
        asd_all.append(asd_case)
        iou_all.append(iou_case)
        acc_2d_all.append(det_acc_2d_case)
        
        ## save results
        if args.visualize:
            os.makedirs(pred_save_dir, exist_ok=True)

            # -------------------------
            # Save RGB input image
            # -------------------------
            save_rgb_image(
                curImg_np,
                os.path.join(pred_save_dir, f"{caseID[0]}_im.png"),
            )

            # -------------------------
            # Save prediction masks
            # -------------------------
            for label_id in label_ids:
                save_mask(
                    segs_dict[label_id],
                    os.path.join(pred_save_dir, f"{caseID[0]}_pd.png"),
                )

            # -------------------------
            # Save GT mask
            # -------------------------
            save_mask(
                curGT_np,
                os.path.join(pred_save_dir, f"{caseID[0]}_gt.png"),
            )



    print(f"mean dice: {np.array(dice_all).mean()}")
    ## save results
    with open(os.path.join(save_eval_path, "test_mean_dice_hd_nsd_asd.txt"), "a") as f:
        write_content = f"{'#'*60}\nmean dice\tmean hd95\tmean nsd\tmean asd\t\n"
        write_content = write_content + f"{np.array(dice_all).mean():.4f}±{np.array(dice_all).std():.2f}\t" \
                                        f"{np.array(hd95_all).mean():.4f}±{np.array(hd95_all).std():.2f}\t" \
                                        f"{np.array(nsd_all).mean():.4f}±{np.array(nsd_all).std():.4f}\t" \
                                        f"{np.array(asd_all).mean():.4f}±{np.array(asd_all).std():.4f}\t" \
                                        f"{np.array(iou_all).mean():.4f}±{np.array(iou_all).std():.4f}\t" \
                                        f"{np.array(acc_2d_all).mean():.4f}±{np.array(acc_2d_all).std():.4f}\t\n"                                      
        f.write(write_content)

    ## save format results
    with open(os.path.join(save_eval_path, "format_test_mean_dice_hd_nsd_asd.txt"), "a") as f:
        write_content = f"{'#'*60}\nmean dice\tmean nsd\tmean hd95\tmean asd\t\n"
        write_content = write_content + f"{np.array(dice_all).mean():.4f}±{np.array(dice_all).std():.4f}\t" \
                                        f"{np.array(nsd_all).mean():.4f}±{np.array(nsd_all).std():.4f}\t" \
                                        f"{np.array(hd95_all).mean():.4f}±{np.array(hd95_all).std():.4f}\t" \
                                        f"{np.array(asd_all).mean():.4f}±{np.array(asd_all).std():.4f}\t" \
                                        f"{np.array(iou_all).mean():.4f}±{np.array(iou_all).std():.4f}\t" \
                                        f"{np.array(acc_2d_all).mean():.4f}±{np.array(acc_2d_all).std():.4f}\t\n"
        write_content = write_content + f"{np.array(dice_all).mean():.2f}±{np.array(dice_all).std():.2f}\t" \
                                        f"{np.array(nsd_all).mean():.2f}±{np.array(nsd_all).std():.2f}\t" \
                                        f"{np.array(hd95_all).mean():.2f}±{np.array(hd95_all).std():.2f}\t" \
                                        f"{np.array(asd_all).mean():.2f}±{np.array(asd_all).std():.2f}\t" \
                                        f"{np.array(iou_all).mean():.2f}±{np.array(iou_all).std():.2f}\t" \
                                        f"{np.array(acc_2d_all).mean():.2f}±{np.array(acc_2d_all).std():.2f}\t\n"
        write_content = write_content + f"{np.array(dice_all).mean()*100:.2f}\t" \
                                        f"{np.array(nsd_all).mean():.2f}\t" \
                                        f"{np.array(hd95_all).mean():.2f}\t" \
                                        f"{np.array(asd_all).mean():.2f}\t" \
                                        f"{np.array(iou_all).mean()*100:.2f}\t" \
                                        f"{np.array(acc_2d_all).mean()*100:.2f}\n"
        f.write(write_content)

    ## save path
    print(f"results saved to {os.path.join(save_eval_path, "format_test_mean_dice_hd_nsd_asd.txt")}")