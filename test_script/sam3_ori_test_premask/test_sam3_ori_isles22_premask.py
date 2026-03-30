#p: test in t1c, t1c+t2f, t1c+t2f+t1n
import sys
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2")
sys.path.append("/home/gxu/proj1/lesionSeg/sam3_brain")
# sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2")
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

from torch import multiprocessing as mp
from torchvision import transforms
from dataset.ustwlits import ToTensor, CenterCrop
from data_load.islse22_builder import load_data_islse22
from gp_knife.get_prompts import get_all_bounding_boxes_with_ccl_box_fix, get_all_bounding_boxes_with_ccl_box_rand, get_all_bounding_boxes_with_ccl
from gp_knife.freq_func import set_random_seed
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
args.visualize = False
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
MODALITY_TYPES = ["DWI"] # "ADC", "DWI", ADC_DWI
NUM = 0
USE_TEXT = True
USE_EXEMP = True
previous_pred = None  # prior mask, HxW
model_path = "/home/gxu/proj1/lesionSeg/sam3_brain/checkpoint"
if USE_EXEMP and (not USE_TEXT):
    save_res_path = os.path.join(model_path, "use_premask/islse22_check/sam3_exemplar")
elif (not USE_EXEMP) and USE_TEXT:
    save_res_path = os.path.join(model_path, "use_premask/islse22_check/sam3_text")
else:
    save_res_path = os.path.join(model_path, "use_premask/islse22_check/sam3_text_exemplar")

for modality_type in MODALITY_TYPES:

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
    all_test_transform = transforms.Compose([
            # CenterCrop(patch_size),
            ToTensor(),
    ])
    testLoader = load_data_islse22(root_path, all_test_transform, batch_size=1, split="test", 
                                   modality_type=modality_type, use_v2=True)

    # load evaluation function
    dice_fun, hd95_fun, nsd_fun, asd_fun = get_dice_hd_nsd_asd()

    dice_all = [] # save all data
    hd95_all = []
    nsd_all = []
    asd_all = []
    iou_all = []
    acc_2d_all = []
    acc_3d_all = []
    cnt_case = 0

    ## load model: sam3
    model = load_sam3(sam3_checkpoint=sam3_checkpoint, device=device)
    processor = Sam3Processor(model, confidence_threshold=0.5, device=device)

    for data_four in testLoader:
        cnt_case = cnt_case + 1

        if modality_type == "ADC": 
            curMR, curGT = data_four["adc"], data_four["curGT"]
            caseID = data_four["caseID"][0]
            curMR = curMR.to(device) # BCHWD
            curGT = curGT.to(device) # BHWD
            B, C, H, W, D = curMR.shape
            print(f"cnt case {cnt_case}, total slices: {D}")
            curGT_np = torch.permute(curGT, dims=(0,1,4,2,3)).cpu().numpy()[0,0,...] # DHW
            ## convert to 3 channels
            curMR = torch.cat((curMR, curMR, curMR), dim=1) # only use current MR image, BCHWD-->B3CHWD
            curMR_np = torch.permute(curMR, dims=(0, 4, 2, 3, 1)).cpu().numpy()[0,...] # DHW3
        elif modality_type == "DWI":
            curMR, curGT = data_four["dwi"], data_four["curGT"]
            caseID = data_four["caseID"][0]
            curMR = curMR.to(device) # BCHWD
            curGT = curGT.to(device) # BHWD
            B, C, H, W, D = curMR.shape
            print(f"cnt case {cnt_case}, total slices: {D}")
            curGT_np = torch.permute(curGT, dims=(0,1,4,2,3)).cpu().numpy()[0,0,...] # DHW
            ## convert to 3 channels
            curMR = torch.cat((curMR, curMR, curMR), dim=1) # only use current MR image, BCHWD-->B3CHWD
            curMR_np = torch.permute(curMR, dims=(0, 4, 2, 3, 1)).cpu().numpy()[0,...] # DHW3
        elif modality_type == "ADC_DWI":
            curMR, curGT, MR_DWI,  = data_four["adc"], data_four["curGT"], data_four["dwi"]
            caseID = data_four["caseID"][0]
            curMR = curMR.to(device) # BCHWD
            curGT = curGT.to(device) # BHWD
            MR_DWI = MR_DWI.to(device)
            B, C, H, W, D = curMR.shape
            print(f"cnt case {cnt_case}, total slices: {D}")
            curGT_np = torch.permute(curGT, dims=(0,1,4,2,3)).cpu().numpy()[0,0,...] # DHW
            ## convert to 3 channels
            curMR = torch.cat((curMR, MR_DWI, curMR), dim=1) # only use current MR image, BCHWD-->B3CHWD
            curMR_np = torch.permute(curMR, dims=(0, 4, 2, 3, 1)).cpu().numpy()[0,...] # DHW3
    
        # exclude no lesion case
        if curGT.sum() < 2:
            continue


        ## inference
        segs_dict = {}
        label_ids = np.unique(curGT_np)[1:]
        ## Simulate 3D box for each organ
        # segs_3d_temp = np.zeros_like(curGT_np, dtype=np.uint8)
        previous_pred = None
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
                img_2d = curMR_np[z, :, :, 0] # HW
                if previous_pred is None:
                    # First slice: no prior
                    prior = np.zeros_like(img_2d)
                else:
                    prior = previous_pred

                # Stack: [current, current, previous_pred]
                img_3c = np.stack(
                    [img_2d, img_2d, prior],
                    axis=-1
                )  # H x W x 3

                # img_512_tensor = sam3_transforms(img_3c)[None, ...].to(device) # 1x3xHxW
                img_512_tensor = sam3_transforms(img_3c).to(device) # 1x3xHxW

                # inference
                inference_state = processor.set_image(img_512_tensor)
                processor.reset_all_prompts(inference_state)
                if USE_TEXT:
                    inference_state = processor.set_text_prompt(state=inference_state, prompt="lesion")
                
                ## get bbox
                box_ori = bbox_dict[z]
                ## check if there are more than one bbox in a mask
                if box_ori.ndim > 1:
                    # randomly select one box for inference
                    rand_idx = list(np.random.randint(0, box_ori.shape[0], BOX_NUM))
                    for idx in rand_idx:
                        # norm box
                        norm_box_cxcywh = norm_box(box_ori[idx,:], W, H, device=device)
                        if USE_EXEMP:
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
                        segs_3d_temp[z, img_2d_seg>0] = 1
                    # ✅ update prior using union of objects
                    previous_pred = (segs_3d_temp[z] > 0).astype(np.float32)
                else:
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
                        if segs_3d_temp.shape[1:] != img_2d_seg.shape: # do interpolation if size not match
                            img_2d_seg = cv2.resize(img_2d_seg, (H, W), interpolation=cv2.INTER_NEAREST)
                            # print(f"case_cnt: {cnt_case}, Original shape {img_2d_seg.shape}, prediction shape {masks.shape[1:]}")                    
                        segs_3d_temp[z, img_2d_seg>0] = 1
                    # ✅ update prior using union of objects
                    previous_pred = (segs_3d_temp[z] > 0).astype(np.float32)
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

    ## save path
    print(f"results saved to {os.path.join(save_eval_path, "format_test_mean_all_metrics.txt")}")
