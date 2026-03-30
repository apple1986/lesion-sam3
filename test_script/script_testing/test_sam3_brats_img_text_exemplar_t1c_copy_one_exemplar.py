# copy one image exemplar as a reference to the other images
import sys
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2")
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2")
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

from sam2.utils.transforms import SAM2Transforms
from torch import multiprocessing as mp
from torchvision import transforms
from dataset.ustwlits import ToTensor
from data_load.brats_builder import load_data_brats_mets
from gp_knife.get_prompts import get_all_bounding_boxes_with_ccl_box_fix, get_all_bounding_boxes_with_ccl_box_rand, get_all_bounding_boxes_with_ccl
from gp_knife.eval_model import get_dice_hd_nsd_asd
from medpy.metric import dc
import cv2
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
torch.manual_seed(2025)
torch.cuda.manual_seed(2025)
np.random.seed(2025)
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


## build model architecture

@torch.no_grad()
## load model: sam3
def load_sam3(device):
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=bpe_path, enable_inst_interactivity=True, device=device)

    return model

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


#######################################
## testing
image_size = 512
args.visualize = True
data_root = args.data_root
# bbox_shift = args.bbox_shift
args.bbox_shift = 0
args.noMask = False
device = "cuda:1"  # args.device
args.model_cfg = "sam2_hiera_t.yaml"
args.sam2_checkpoint = "/home/gxu/proj1/lesionSeg/utswlits3d_v2/checkpoint/SAM/sam2_hiera_tiny.pt"
model_cfg = args.model_cfg
sam2_checkpoint = args.sam2_checkpoint
args.ch3 = True
flag_fix = False # False True
ONLY_COPY = False

if flag_fix is True:
    box_func = get_all_bounding_boxes_with_ccl_box_fix 
else:
    box_func = get_all_bounding_boxes_with_ccl_box_rand

root_path = "/home/gxu/proj1/lesionSeg/sam3_brain"
BOX_NUM = 1 # number of box examplars

for NUM in range(0, 1):
    num_workers = args.num_workers
    save_eval_path = os.path.join(root_path, "results/brats_mets/sam3_ori/sam3_brats_eval_result_shift"+str(NUM)+"_text_copy_one_exemplar_only")
    makedirs(save_eval_path, exist_ok=True)
    # save segmentation results
    pred_save_dir = os.path.join(root_path, "results/brats_mets/sam3_ori/sam3_brats_seg_result_shift"+str(NUM)+"_text_copy_one_exemplar_only")
    makedirs(pred_save_dir, exist_ok=True)
    # copy the current testing script
    script_path = join(save_eval_path, os.path.basename(__file__))
    if not os.path.exists(script_path):
        shutil.copyfile(
            __file__, script_path
        )
    args.bbox_shift =  NUM # NUM
    print(f"Testing with bbox shift: {args.bbox_shift}")
    sam2_transforms = SAM2Transforms(resolution=512, mask_threshold=0)

    ## load data
    all_test_transform = transforms.Compose([
            # CenterCrop(patch_size),
            ToTensor(),
    ])
    testLoader = load_data_brats_mets(root_path, all_test_transform, batch_size=1, split="test")

    # load evaluation function
    dice_fun, hd95_fun, nsd_fun, asd_fun = get_dice_hd_nsd_asd()

    dice_all = [] # save all data
    hd95_all = []
    nsd_all = []
    asd_all = []
    cnt_case = 0

    ## load model: sam3
    model = load_sam3(device)
    processor = Sam3Processor(model, confidence_threshold=0.5, device=device)

    for data_four in testLoader:
        cnt_case = cnt_case + 1
        
        curMR, curGT = data_four["curMR"], data_four["curGT"]
        caseID = data_four["caseID"][0]
        curMR = curMR.to(device) # BCHWD
        curGT = curGT.to(device) # BCHWD

        B, C, H, W, D = curMR.shape
        print(f"cnt case {cnt_case}, total slices: {D}")
        curGT_np = torch.permute(curGT, dims=(0,1,4,2,3)).cpu().numpy()[0,0,...] # DHW
        ## convert to 3 channels
        curMR = torch.cat((curMR, curMR, curMR), dim=1) # only use current MR image, BCHWD-->B3CHWD
        curMR_np = torch.permute(curMR, dims=(0, 4, 2, 3, 1)).cpu().numpy()[0,...] # DHW3

        ## inference
        segs_dict = {}
        label_ids = np.unique(curGT_np)[1:] # exclude 0
        ## Simulate 3D box for each organ
        for label_id in label_ids:
            segs_3d_temp = np.zeros_like(curGT_np, dtype=np.uint8)
            marker_data_id = (curGT_np == label_id).astype(np.uint8)
            marker_zids, _, _ = np.where(marker_data_id > 0)
            marker_zids = np.sort(np.unique(marker_zids))
            bbox_dict = {} # key: z_index, value: bbox
            # get all bbox for this volume case
            for z in marker_zids:
                z_box_check = box_func(marker_data_id[z, :, :], bbox_shift=args.bbox_shift)
                bbox_dict[z] = z_box_check

            ## randomly select a box and the corresponding mask and the image intensity
            ## get bbox
            ref_idx = marker_zids[np.random.randint(0, len(marker_zids))] # randomly select a slice to copy the image exemplar
            box_ori = bbox_dict[ref_idx]
            if box_ori.ndim > 1:
                # randomly select one box for inference
                rand_idx = np.random.randint(0, box_ori.shape[0])
                sel_box = box_ori[rand_idx,:]
                x1, y1, x2, y2 = sel_box[0], sel_box[1], sel_box[2], sel_box[3]
                sel_exmp_img = curMR_np[ref_idx, y1:y2, x1:x2, :] # hw3
                sel_exmp_msk = curGT_np[ref_idx, y1:y2, x1:x2, ] # hw
            else:
                sel_box = box_ori
                x1, y1, x2, y2 = sel_box[0], sel_box[1], sel_box[2], sel_box[3]
                sel_exmp_img = curMR_np[ref_idx, y1:y2, x1:x2, :] # hw3
                sel_exmp_msk = curGT_np[ref_idx, y1:y2, x1:x2, ] # hw
            
            ## inference
            for z in list(marker_zids):
                img_2d = curMR_np[z, :, :, :] # HW3
                # randomly paste the patch image and the correponding mask: sel_exmp_msk
                # keep that there is no overlap the current object in the mask
                H, W, _ = img_2d.shape
                ph, pw = sel_exmp_img.shape[:2]

                max_try = 50
                valid = False

                for _ in range(max_try):
                    px = np.random.randint(0, W - pw)
                    py = np.random.randint(0, H - ph)

                    # 当前 slice 的 GT mask
                    cur_mask_2d = curGT_np[z]  # (H, W)

                    # 确保 pasted 区域不与当前 GT 重叠
                    if cur_mask_2d[py:py+ph, px:px+pw].sum() == 0:
                        valid = True
                        used_box = np.array([px, py, px+pw, py+ph])
                        break

                if not valid:
                    continue  # 放弃本次 copy-paste

                # img_2d = img_2d.copy()
                img_2d[py:py+ph, px:px+pw, :] = sel_exmp_img
                # 构造 ignore mask
                ignore_mask = np.zeros((H, W), dtype=bool)
                ignore_mask[py:py+ph, px:px+pw] = sel_exmp_msk#.astype(bool)

                # mask_2d = curGT[..., z].float() # mask_2d = curGT_t[..., z].float() just for checking
                # mask_2d = F.interpolate(mask_2d, size=(256, 256), mode="nearest", align_corners=None)
                if len(img_2d.shape) == 2:
                    img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
                else:
                    img_3c = img_2d
                # convert the shape to (1, 3, 512, 512)
                # img_512_tensor = sam2_transforms(img_3c)[None, ...].to(device) # 1x3xHxW
                img_512_tensor = sam2_transforms(img_3c).to(device) # 1x3xHxW

                # # inference
                inference_state = processor.set_image(img_512_tensor)
                processor.reset_all_prompts(inference_state)
                inference_state = processor.set_text_prompt(state=inference_state, prompt="lesion")
               
                ## get bbox
                if ONLY_COPY:
                    box_ori = used_box
                else:
                    box_ori = bbox_dict[z]
                ## check if there are more than one bbox in a mask
                if box_ori.ndim > 1:
                    # randomly select one box for inference
                    rand_idx = list(np.random.randint(0, box_ori.shape[0], BOX_NUM))
                    for idx in rand_idx:
                        # norm box
                        norm_box_cxcywh = norm_box(box_ori[idx,:], W, H, device=device)
                        inference_state = processor.add_geometric_prompt(
                            state=inference_state, box=norm_box_cxcywh, label=True
                                    )
                        # add a copied bbox
                        norm_box_cxcywh = norm_box(used_box, W, H, device=device)
                        inference_state = processor.add_geometric_prompt(
                            state=inference_state, box=norm_box_cxcywh, label=True
                                    )                        
                    nb_objects = len(inference_state["scores"])
                    print(f"found {nb_objects} object(s)")
                    for n in range(nb_objects):
                        img_2d_seg = inference_state["masks"][n][0].cpu().numpy() # HxW
                        
                        img_2d_seg = img_2d_seg.astype(np.uint8)
                        if segs_3d_temp.shape[1:] != img_2d_seg.shape: # do interpolation if size not match
                            img_2d_seg = cv2.resize(img_2d_seg, (H, W), interpolation=cv2.INTER_NEAREST)
                            # print(f"case_cnt: {cnt_case}, Original shape {img_2d_seg.shape}, prediction shape {masks.shape[1:]}")                    
                        img_2d_seg[ignore_mask] = 0
                        segs_3d_temp[z, img_2d_seg>0] = 1

                else:
                    norm_box_cxcywh = norm_box(box_ori, W, H, device=device)
                    inference_state = processor.add_geometric_prompt(
                            state=inference_state, box=norm_box_cxcywh, label=True
                                    )
                    # add a copied bbox
                    norm_box_cxcywh = norm_box(used_box, W, H, device=device)
                    inference_state = processor.add_geometric_prompt(
                        state=inference_state, box=norm_box_cxcywh, label=True
                                )                       
                    # inference_state = processor.set_text_prompt(state=inference_state, prompt="lesion")
                    nb_objects = len(inference_state["scores"])
                    print(f"found {nb_objects} object(s)")
                    for n in range(nb_objects):
                        img_2d_seg = inference_state["masks"][n][0].cpu().numpy() # HxW
                        
                        img_2d_seg = img_2d_seg.astype(np.uint8)
                        if segs_3d_temp.shape[1:] != img_2d_seg.shape: # do interpolation if size not match
                            img_2d_seg = cv2.resize(img_2d_seg, (H, W), interpolation=cv2.INTER_NEAREST)
                            # print(f"case_cnt: {cnt_case}, Original shape {img_2d_seg.shape}, prediction shape {masks.shape[1:]}")                    
                        img_2d_seg[ignore_mask] = 0
                        segs_3d_temp[z, img_2d_seg>0] = 1

            segs_dict[label_id] = segs_3d_temp.copy()
        # cal dice, hd95, nsd and asd
        pd = torch.from_numpy(segs_3d_temp)[None, None, ...] #BCDHW
        gt = torch.from_numpy(curGT_np)[None, None, ...] 
        dice_case = dice_fun(pd, gt).numpy()[0,0]
        hd95_case = hd95_fun(pd, gt).numpy()[0,0]
        nsd_case = nsd_fun(pd, gt).numpy()[0,0]
        asd_case = asd_fun(pd, gt).numpy()[0,0]
        print(f"case {caseID} dice: {dice_case}")

        ## save each case results to txt
        with open(os.path.join(save_eval_path, "test_each_dice_hd95_nsd_asd_rand"+str(NUM)+".txt"), "a") as f:
            write_content = f"{caseID} dice:\t{dice_case:.4f}\thd95:\t{hd95_case:.4f}\tnsd:\t{nsd_case:.4f}\tasd:\t{asd_case:.4f}\t\n"
            f.write(write_content)

        dice_all.append(dice_case)
        hd95_all.append(hd95_case)
        nsd_all.append(nsd_case)
        asd_all.append(asd_case)
        
        ## save results
        curMR_np = curMR_np[..., 0] #DHW3-->DHW .transpose((1,2,0,3)
        if args.visualize:
            for label_id in label_ids:
                seg_sitk = sitk.GetImageFromArray(segs_dict[label_id])
                seg_sitk.SetSpacing((1, 1, 1))
                sitk.WriteImage(seg_sitk, join(pred_save_dir, caseID+"_pd_sam2.nii.gz"))

            # img_sitk = sitk.GetImageFromArray(curMR_np)
            # img_sitk.SetSpacing((1, 1, 1))
            # sitk.WriteImage(img_sitk, join(pred_save_dir, caseID+"_im_sam2.nii.gz"))
            gts_sitk = sitk.GetImageFromArray(curGT_np)
            gts_sitk.SetSpacing((1, 1, 1))
            sitk.WriteImage(gts_sitk, join(pred_save_dir, caseID+"_gt_sam2.nii.gz"))


    print(f"mean dice: {np.array(dice_all).mean()}")
    ## save results
    with open(os.path.join(save_eval_path, "test_mean_dice_hd_nsd_asd"+str(NUM)+".txt"), "a") as f:
        write_content = f"{'#'*60}\nmean dice\tmean hd95\tmean nsd\tmean asd\t\n"
        write_content = write_content + f"{np.array(dice_all).mean():.4f}±{np.array(dice_all).std():.2f}\t" \
                                        f"{np.array(hd95_all).mean():.4f}±{np.array(hd95_all).std():.2f}\t" \
                                        f"{np.array(nsd_all).mean():.4f}±{np.array(nsd_all).std():.4f}\t" \
                                        f"{np.array(asd_all).mean():.4f}±{np.array(asd_all).std():.4f}\t\n"
        f.write(write_content)

    ## save format results
    with open(os.path.join(save_eval_path, "format_test_mean_dice_hd_nsd_asd"+str(NUM)+".txt"), "a") as f:
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
                                        f"{np.array(hd95_all).mean():.2f}\t" \
                                        f"{np.array(asd_all).mean():.2f}\n"
        f.write(write_content)

