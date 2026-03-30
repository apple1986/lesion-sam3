# ap: use SAM3Transforms, without mean and std normalization, input image in [0, 1], the processor will handle the mean and std normalization
from ast import mod
import sys
from turtle import st

import test
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2")
sys.path.append("/home/gxu/proj1/lesionSeg/sam3_brain")
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

from ap_knife.sam_data_transforms import SAM3Transforms
from ap_knife.eval_metrics import dice_score, iou_score, detection_accuracy_2d, detection_accuracy_3d

from data_load.brats_builder import load_data_brats_mets
from data_load.all_data_loader import all_testsets
from ap_knife.get_prompts import get_all_bounding_boxes_with_ccl_box_fix, get_all_bounding_boxes_with_ccl_box_rand
from ap_knife.eval_metrics import get_dice_hd_nsd_asd
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


## build model architecture

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
device = "cuda:1"  # args.device
flag_fix = False # False True

if flag_fix is True:
    box_func = get_all_bounding_boxes_with_ccl_box_fix 
else:
    box_func = get_all_bounding_boxes_with_ccl_box_rand

root_path = "/home/gxu/proj1/lesionSeg/sam3_brain"
model_path = "/home/gxu/proj1/lesionSeg/sam3_brain/checkpoint"
save_root_path = os.path.join(model_path, "use_all_bbox")
sam3_checkpoint = os.path.join(model_path, "sam3.pt")
sam3_transforms = SAM3Transforms(resolution=512, mask_threshold=0) # no z-score normalization

# load model
model = load_sam3(sam3_checkpoint, device=device) # load the pretrained model     
state_dict = torch.load(sam3_checkpoint, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()
processor = Sam3Processor(model, confidence_threshold=0.5, device=device)
# load evaluation function
dice_fun, hd95_fun, nsd_fun, asd_fun = get_dice_hd_nsd_asd()
BOX_NUM = 1 # number of box examplars
## load data
# testLoader = load_data_brats_mets(root_path, transform_policy=None, batch_size=1, split="test", modality_type="v2") # use clip_norm_volum
for testset in all_testsets():
    if testset[0] == "brats":
        testLoader = testset[1](root_path, transform_policy=None, batch_size=1, split="test", modality_type="v2")
        print(f"Loaded test set: {testset[0]}")
    elif testset[0] == "isles":
        testLoader = testset[1](root_path, transform_policy=None, batch_size=1, split="test", modality_type=testset[3], use_v2=True)
        print(f"Loaded test set: {testset[0]}")
    elif testset[0] == "colon":
        testLoader = testset[1](root_path, transform_policy=None, batch_size=1, split="test", modality_type=testset[3], use_v2=True)
        print(f"Loaded test set: {testset[0]}")
    elif testset[0] == "kidney":
        testLoader = testset[1](root_path, transform_policy=None, batch_size=1, split="test", modality_type=testset[3], use_v2=True)
        print(f"Loaded test set: {testset[0]}")
    elif testset[0] == "liver":
        testLoader = testset[1](root_path, transform_policy=None, batch_size=1, split="test", modality_type=testset[3], use_v2=True)
        print(f"Loaded test set: {testset[0]}")
    elif testset[0] == "lung":
        testLoader = testset[1](root_path, transform_policy=None, batch_size=1, split="test", modality_type=testset[3], use_v2=True)
        print(f"Loaded test set: {testset[0]}")
    elif testset[0] == "pancreas":
        testLoader = testset[1](root_path, transform_policy=None, batch_size=1, split="test", modality_type=testset[3], use_v2=True)
        print(f"Loaded test set: {testset[0]}")
    elif testset[0] == "sts":
        testLoader = testset[1](root_path, transform_policy=None, batch_size=1, split="test", modality_type=testset[3], use_v2=True)
        print(f"Loaded test set: {testset[0]}")
    elif testset[0] == "busi":
        testLoader = testset[1](root_path, transform_policy=None, batch_size=1, split="test", modality_type=testset[3])
        print(f"Loaded test set: {testset[0]}")
    elif testset[0] == "kvasir":
        testLoader = testset[1](root_path, transform_policy=None, batch_size=1, split="test", modality_type=testset[3])
        print(f"Loaded test set: {testset[0]}")
    elif testset[0] == "isic":
        testLoader = testset[1](root_path, transform_policy=None, batch_size=1, split="test", modality_type=testset[3])
        print(f"Loaded test set: {testset[0]}")


    for NUM in range(0, 12, 2):
        print(NUM)
        # set save path
        save_res_path = os.path.join(save_root_path, f"shift_{NUM}")        
        save_eval_path = os.path.join(save_res_path, f"{testset[0]}/eval_result")
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
        args.bbox_shift =  NUM
        print(f"Testing with bbox shift: {args.bbox_shift}")
        
        # init all metrics
        dice_all = [] # save all data
        hd95_all = []
        nsd_all = []
        asd_all = []
        iou_all = []
        acc_2d_all = []
        acc_3d_all = []
        cnt_case = 0
        
        # load data and inference
        for data_four in testLoader:
            cnt_case = cnt_case + 1
            if testset[0] == "brats":
                curInput, curGT = data_four["curMR"], data_four["curGT"]
                caseID = data_four["caseID"][0]
                curGT = curGT.to(device) # BHWD
                B, C, H, W, D = curInput.shape
                print(f"cnt case {cnt_case}, total slices: {D}")
                curGT_np = torch.permute(curGT, dims=(0,1,4,2,3)).cpu().numpy()[0,0,...] # DHW
                ## convert to 3 channels
                curInput = torch.cat((curInput, curInput, curInput), dim=1) # only use current MR image, BCHWD-->B3CHWD
                curInput_np = torch.permute(curInput, dims=(0, 4, 2, 3, 1)).cpu().numpy()[0,...] # DHW3
            elif testset[0] == "isles":
                curInput, curGT = data_four["dwi"], data_four["curGT"]
                caseID = data_four["caseID"][0]
                curInput = curInput.to(device) # BCHWD
                curGT = curGT.to(device) # BHWD
                B, C, H, W, D = curInput.shape
                print(f"cnt case {cnt_case}, total slices: {D}")
                curGT_np = torch.permute(curGT, dims=(0,1,4,2,3)).cpu().numpy()[0,0,...] # DHW
                ## convert to 3 channels
                curInput = torch.cat((curInput, curInput, curInput), dim=1) # only use current MR image, BCHWD-->B3CHWD
                curInput_np = torch.permute(curInput, dims=(0, 4, 2, 3, 1)).cpu().numpy()[0,...] # DHW3 
            elif testset[0] in ["colon", "kidney", "liver", "lung", "pancreas"]:
                curCT, curGT = data_four["curCT"], data_four["curGT"]
                caseID = data_four["caseID"][0]
                curCT = curCT.to(device) # BCHWD
                curGT = curGT.to(device) # BHWD
                B, C, H, W, D = curCT.shape
                print(f"cnt case {cnt_case}, total slices: {D}")
                curGT_np = torch.permute(curGT, dims=(0,1,4,2,3)).cpu().numpy()[0,0,...] # DHW
                ## convert to 3 channels
                curCT = torch.cat((curCT, curCT, curCT), dim=1) # only use current CT image, BCHWD-->B3CHWD
                curCT_np = torch.permute(curCT, dims=(0, 4, 2, 3, 1)).cpu().numpy()[0,...] # DHW3
            elif testset[0] == "sts":
                curCT, curPT, curGT = data_four["curCT"], data_four["curPT"], data_four["curGT"]
                caseID = data_four["caseID"][0]
                curCT = curCT.to(device) # BCHWD
                curGT = curGT.to(device) # BHWD
                B, C, H, W, D = curCT.shape
                print(f"cnt case {cnt_case}, total slices: {D}")
                curGT_np = torch.permute(curGT, dims=(0,1,4,2,3)).cpu().numpy()[0,0,...] # DHW
                ## convert to 3 channels
                curCT = torch.cat((curCT, curCT, curCT), dim=1) # only use current CT image, BCHWD-->B3CHWD
                curCT_np = torch.permute(curCT, dims=(0, 4, 2, 3, 1)).cpu().numpy()[0,...] # DHW3            

            ## inference
            segs_dict = {}
            label_ids = np.unique(curGT_np)[1:]
            ## Simulate 3D box for each organ
            for label_id in label_ids:
                segs_3d_temp = np.zeros_like(curGT_np, dtype=np.uint8)
                marker_data_id = (curGT_np == label_id).astype(np.uint8)
                # marker_data_id = (preGT_np == label_id).astype(np.uint8) # use preGT to extract bbox
                marker_zids, _, _ = np.where(marker_data_id > 0)
                marker_zids = np.sort(np.unique(marker_zids))
                bbox_dict = {} # key: z_index, value: bbox
                for z in marker_zids:
                    z_box_check = box_func(marker_data_id[z, :, :], bbox_shift=args.bbox_shift)
                    bbox_dict[z] = z_box_check

                ## inference
                for z in list(marker_zids):
                    img_2d = curInput_np[z, :, :, :] # HW3
                    mask_2d = curGT[..., z].float() # mask_2d = curGT_t[..., z].float() just for checking
                    mask_2d = F.interpolate(mask_2d, size=(256, 256), mode="nearest", align_corners=None)
                    if len(img_2d.shape) == 2:
                        img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
                    else:
                        img_3c = img_2d
                    # convert the shape to (1, 3, 512, 512)
                    # img_512_tensor = sam3_transforms(img_3c)[None, ...].to(device) # 1x3xHxW
                    img_512_tensor = sam3_transforms(img_3c).to(device) # 1x3xHxW

                    # inference
                    inference_state = processor.set_image(img_512_tensor)
                    
                    ## get bbox
                    box_ori = bbox_dict[z]
                    ## check if there are more than one bbox in a mask
                    if box_ori.ndim > 1:
                        ## inference one by one
                        for num_box in range(box_ori.shape[0]):
                            box_512 = box_ori[num_box,:] / np.array([W, H, W, H]) * 512 ## adjust to 512x512 scale
                            masks, scores, logits = model.predict_inst(
                                                            inference_state,
                                                            box=box_512[None,:],
                                                            multimask_output=True,
                                                        )
                            sorted_ind = np.argsort(scores)[::-1]
                            img_2d_seg =  masks[sorted_ind[0]]
                            if segs_3d_temp.shape[1:] != img_2d_seg.shape: # do interpolation if size not match
                                img_2d_seg = cv2.resize(img_2d_seg, (H, W), interpolation=cv2.INTER_NEAREST)
                                # print(f"case_cnt: {cnt_case}, Original shape {img_2d_seg.shape}, prediction shape {masks.shape[1:]}")                    
                            segs_3d_temp[z, img_2d_seg>0] = 1
                    else:
                        box_512 = box_ori / np.array([W, H, W, H]) * 512 ## adjust to 512x512 scale
                        masks, scores, logits = model.predict_inst(
                                                        inference_state,
                                                        box=box_512[None,:],
                                                        multimask_output=True,
                                                    )
                        sorted_ind = np.argsort(scores)[::-1]
                        img_2d_seg =  masks[sorted_ind[0]]    
                        if segs_3d_temp.shape[1:] != img_2d_seg.shape: # do interpolation if size not match
                            img_2d_seg = cv2.resize(img_2d_seg, (H, W), interpolation=cv2.INTER_NEAREST)
                            # print(f"case_cnt: {cnt_case}, Original shape {img_2d_seg.shape}, prediction shape {masks.shape[1:]}")                                                
                        segs_3d_temp[z, img_2d_seg>0] = 1
                segs_dict[label_id] = segs_3d_temp.copy()

            # cal dice, hd95, nsd and asd
            pd = torch.from_numpy(segs_3d_temp)[None, None, ...] #BCDHW
            gt = torch.from_numpy(curGT_np)[None, None, ...] 
            dice_case = dice_fun(pd, gt).numpy()[0,0]
            hd95_case = hd95_fun(pd, gt).numpy()[0,0]
            nsd_case = nsd_fun(pd, gt).numpy()[0,0]
            asd_case = asd_fun(pd, gt).numpy()[0,0]
            iou_case = iou_score(segs_3d_temp>0, curGT_np>0)
            print(f"case {caseID} dice: {dice_case}")

            # calculate detection accuracy 2d and 3d
            detection_accuracy_2d_list = []
            for num_slice in range(segs_3d_temp.shape[0]):
                curGT_slice = curGT_np[num_slice, :, :]
                segs_2d_slice = segs_3d_temp[num_slice, :, :]
                if curGT_slice.sum() > 5:
                    detection_accuracy_2d_slice, _, _ = detection_accuracy_2d(curGT_slice, segs_2d_slice)
                    # print(f"case {caseID} slice {num_slice} detection accuracy 2d: {detection_accuracy_2d_slice}")
                    detection_accuracy_2d_list.append(detection_accuracy_2d_slice)

            det_acc_2d_case = np.array(detection_accuracy_2d_list).mean()
            det_acc_3d_case, _, _ = detection_accuracy_3d(curGT_np, segs_3d_temp)

            ## save each case results to txt
            with open(os.path.join(save_eval_path, "test_each_dice_hd95_nsd_asd_rand"+str(NUM)+".txt"), "a") as f:
                write_content = f"{caseID} dice:\t{dice_case:.4f}\thd95:\t{hd95_case:.4f}\tnsd:\t{nsd_case:.4f}\tasd:\t{asd_case:.4f}\tiou:\t{iou_case:.4f}\tacc_2d:\t{det_acc_2d_case:.4f}\tacc_3d:\t{det_acc_3d_case:.4f}\n"
                f.write(write_content)

            # save all case results
            dice_all.append(dice_case)
            hd95_all.append(hd95_case)
            nsd_all.append(nsd_case)
            asd_all.append(asd_case)
            iou_all.append(iou_case)
            acc_2d_all.append(det_acc_2d_case)
            acc_3d_all.append(det_acc_3d_case)
            
            ## save results
            curInput_np = curInput_np[..., 0] #DHW3-->DHW .transpose((1,2,0,3)
        print(f"mean dice: {np.array(dice_all).mean()}")
        # save results for the additional metrics
        with open(os.path.join(save_eval_path, "test_mean_all_metrics.txt"), "a") as f:
            write_content = f"{'#'*60}\nmean dice\tmean hd95\tmean nsd\tmean asd\tmean iou\tmean acc_2d\tmean acc_3d\n"
            write_content += f"{np.array(dice_all).mean():.4f}±{np.array(dice_all).std():.4f}\t" \
                            f"{np.array(hd95_all).mean():.4f}±{np.array(hd95_all).std():.4f}\t" \
                            f"{np.array(nsd_all).mean():.4f}±{np.array(nsd_all).std():.4f}\t" \
                            f"{np.array(asd_all).mean():.4f}±{np.array(asd_all).std():.4f}\t" \
                            f"{np.array(iou_all).mean():.4f}±{np.array(iou_all).std():.4f}\t" \
                            f"{np.array(acc_2d_all).mean():.4f}±{np.array(acc_2d_all).std():.4f}\t" \
                            f"{np.array(acc_3d_all).mean():.4f}±{np.array(acc_3d_all).std():.4f}\t\n"
            f.write(write_content)

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
        # copy the current testing txt to save_res_path
        src_file = os.path.join(save_eval_path, "format_test_mean_all_metrics.txt")
        shutil.copyfile(
            src_file, os.path.join(save_res_path, f"{testset[0]}_format_test_mean_all_metrics.txt")
        )