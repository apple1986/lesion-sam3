import sys
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

from sam2.build_sam import build_sam2
from ap_knife.sam_data_transforms import SAM3Transforms
from ap_knife.eval_metrics import dice_score, iou_score, detection_accuracy_2d, detection_accuracy_3d


from torch import multiprocessing as mp
from torchvision import transforms
from dataset.ustwlits import ToTensor, CenterCrop
from data_load.utswdataset_builder import load_data
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
parser.add_argument("-device", type=str, default="cuda:0", help="device")
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
class SAM2Wrapper(nn.Module):
    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.sam2_model = model
        # freeze prompt encoder
        for param in self.sam2_model.sam_prompt_encoder.parameters():
            param.requires_grad = False
        

    def forward(self, image, box):
        """
        image: (B, 3, 512, 512)
        box: (B, 2, 2)
        """
        _features = self._image_encoder(image)
        img_embed, high_res_features = _features["image_embed"], _features["high_res_feats"]
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_coords = box_torch.reshape(-1, 2, 2) # (B, 4) to (B, 2, 2)
                box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=image.device)
                box_labels = box_labels.repeat(box_torch.size(0), 1)
            concat_points = (box_coords, box_labels)

            sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                points=concat_points,
                boxes=None,
                masks=None,
            )
        low_res_masks_logits, _, _, _ = self.sam2_model.sam_mask_decoder(
            image_embeddings=img_embed, # (1, 256, 64, 64)
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        return low_res_masks_logits


    def _image_encoder(self, input_image):
        backbone_out = self.sam2_model.forward_image(input_image)
        _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)
        if self.sam2_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed
        bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        feats = [
            feat.permute(1, 2, 0).view(input_image.size(0), -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
        ][::-1]

        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        return _features
   
@torch.no_grad()
def medsam_inference(
    medsam_model,
    features,
    box_512,
    H, W,
    masks=None,
    ):
    img_embed, high_res_features = features["image_embed"], features["high_res_feats"]
    box_torch = torch.as_tensor(box_512, dtype=torch.float32, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_coords = box_torch.reshape(-1, 2, 2) # (B, 4) to (B, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=img_embed.device)
        box_labels = box_labels.repeat(box_torch.size(0), 1)
    concat_points = (box_coords, box_labels)

    sparse_embeddings, dense_embeddings = medsam_model.sam2_model.sam_prompt_encoder(
        points=concat_points,
        boxes=None,
        masks=masks,
    )
    low_res_masks_logits, _, _, _ = medsam_model.sam2_model.sam_mask_decoder(
        image_embeddings=img_embed, # (1, 256, 64, 64)
        image_pe=medsam_model.sam2_model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_features, # two: Bx64x128x128; Bx32x256x256
    )

    low_res_pred = torch.sigmoid(low_res_masks_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

    return medsam_seg

## load model: sam3
def load_sam3():
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=bpe_path, enable_inst_interactivity=True)

    return model

def norm_box(box_ori, W, H, device="cuda:0"):

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
args.bbox_shift = 10
args.noMask = False
device = "cuda:1"  # args.device
args.model_cfg = "sam2_hiera_t.yaml"
args.sam2_checkpoint = "/home/gxu/proj1/lesionSeg/utswlits3d_v2/checkpoint/SAM/sam2_hiera_tiny.pt"
model_cfg = args.model_cfg
sam2_checkpoint = args.sam2_checkpoint
args.ch3 = True
flag_fix = False # False True
# tv_types = ['gtv', 'itv', 'ptv', 'alltv']
tv_types = ['itv']

if flag_fix is True:
    box_func = get_all_bounding_boxes_with_ccl_box_fix 
else:
    box_func = get_all_bounding_boxes_with_ccl_box_rand

root_path = "/home/gxu/proj1/lesionSeg/sam3_brain"
model_path = "/home/gxu/proj1/lesionSeg/sam3_brain/checkpoint"
USE_TEXT = True
USE_EXEMP = True
BOX_NUM = 1 # number of box examplars

if USE_EXEMP and (not USE_TEXT):
    save_res_path = os.path.join(model_path, "utsw_liver/sam3_exemplar")
elif (not USE_EXEMP) and USE_TEXT:
    save_res_path = os.path.join(model_path, "utsw_liver/sam3_text")
else:
    save_res_path = os.path.join(model_path, "utsw_liver/sam3_text_exemplar")
for tv_type in tv_types:
    for NUM in range(0, 1):
        num_workers = args.num_workers
        save_eval_path = os.path.join(save_res_path, "eval_result_"+tv_type)
        makedirs(save_eval_path, exist_ok=True)
        # save segmentation results
        pred_save_dir = os.path.join(save_res_path, "seg_result_"+tv_type)
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
        testLoader = load_data(root_path, all_test_transform, batch_size=1, split="liver_all", tv_type=tv_type)

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
        model = load_sam3()
        processor = Sam3Processor(model, confidence_threshold=0.5)

        for data_four in testLoader:
            cnt_case = cnt_case + 1
            
            curMR, preMR, preGT, curGT = data_four["curMR"], data_four["preMR"], data_four["preGT"], data_four["curGT"]
            caseID = data_four["caseID"][0]
            curMR = curMR.to(device) # BCHWD
            preMR = preMR.to(device) # BCHWD
            curGT = curGT.to(device) # BHWD
            preGT = preGT.to(device) # B1HWD

            B, C, H, W, D = curMR.shape
            print(f"cnt case {cnt_case}, total slices: {D}")
            curGT_np = torch.permute(curGT, dims=(0,3,1,2)).cpu().numpy()[0,...] # DHW
            # preGT_np = torch.permute(preGT, dims=(0,1,4,2,3)).cpu().numpy()[0,0,...] # DHW
            ## convert to 3 channels
            curMR = torch.cat((curMR, preMR, preGT), dim=1)
            curMR_np = torch.permute(curMR, dims=(0, 4, 2, 3, 1)).cpu().numpy()[0,...] # DHW3

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
                    img_2d = curMR_np[z, :, :, :] # HW3
                    mask_2d = preGT[..., z].float() # mask_2d = curGT_t[..., z].float() just for checking
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

                img_sitk = sitk.GetImageFromArray(curMR_np)
                img_sitk.SetSpacing((1, 1, 1))
                sitk.WriteImage(img_sitk, join(pred_save_dir, caseID+"_im_sam2.nii.gz"))
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

