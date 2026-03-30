import torch
import torch.nn.functional as F
from tqdm import tqdm
from ap_knife.eval_metrics import dice_score, iou_score
from sam3.model.sam3_image_processor import Sam3Processor
import numpy as np
from ap_knife.get_prompts import (
    get_all_bounding_boxes_with_ccl_box_rand as gen_box_rand
)
from sam3.visualization_utils import normalize_bbox


def norm_box(box_ori, W, H, device="cuda:1"):

    box_scaled = box_ori / np.array([W, H, W, H]) * 1008 ## adjust to 512x512 scale
    # convert the xyxy to cxcywh
    x1, y1, x2, y2 = box_scaled
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w  = x2 - x1
    h  = y2 - y1

    box_input_cxcywh = np.array([cx, cy, w, h])
    box_input_cxcywh = torch.from_numpy(box_input_cxcywh).to(device).view(-1, 4)
    norm_box_cxcywh = normalize_bbox(box_input_cxcywh, 1008, 1008).flatten().tolist()

    return norm_box_cxcywh

def evaluate_sam3(model, val_loader, device="cuda:1", threshold=0.5):
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=threshold, device=device)
    

    dice_list = []
    iou_list = []

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Validation"):
            imgs = imgs.to(device)      # (B,3,1008,1008)
            masks = masks.to(device)    # (B,1008,1008)

            for i in range(imgs.shape[0]):
                img = imgs[i]#.unsqueeze(0) # torch.Size([3, 1008, 1008])
                gt  = masks[i]#.unsqueeze(0).unsqueeze(0)  # (H,W)
                gt_mask = gt.bool() # HW
                if gt.sum() < 10:
                    continue

                # inference
                inference_state = processor.set_image(img)
                processor.reset_all_prompts(inference_state)
                inference_state = processor.set_text_prompt(state=inference_state, prompt="lesion")

                ## get bounding box
                curGT_np = masks[i].cpu().numpy()
                marker_data_id = (curGT_np == 1).astype(np.uint8) # HW
                box_ori = gen_box_rand(marker_data_id, bbox_shift=0)
                ## check if there are more than one bbox in a mask
                if box_ori.ndim > 1:
                    rand_idx = list(np.random.randint(0, box_ori.shape[0], 1))
                    for idx in rand_idx:
                        norm_box_cxcywh = norm_box(box_ori[idx,:], 1008, 1008, device=device)
                        inference_state = processor.add_geometric_prompt(
                                state=inference_state, box=norm_box_cxcywh, label=True
                                        )
                    nb_objects = len(inference_state["scores"])
                    # print(f"found {nb_objects} object(s)")
                    for n in range(nb_objects):
                        pred_mask = inference_state["masks"][n][0]

                else:
                    norm_box_cxcywh = norm_box(box_ori, 1008, 1008, device=device)
                    inference_state = processor.add_geometric_prompt(
                            state=inference_state, box=norm_box_cxcywh, label=True
                                    )
                    nb_objects = len(inference_state["scores"])
                    if nb_objects == 0:
                        pred_mask = torch.zeros_like(gt_mask, dtype=bool)
                    # print(f"found {nb_objects} object(s)")
                    for n in range(nb_objects):
                        pred_mask = inference_state["masks"][n][0]#.cpu().numpy() # HxW
         
                
                dice = dice_score(pred_mask, gt_mask)
                iou  = iou_score(pred_mask, gt_mask)
                print(f"dice: {dice}, iou: {iou}")

                dice_list.append(dice.item())
                iou_list.append(iou.item())

    model.train()
    return {
        "dice": sum(dice_list) / len(dice_list),
        "iou": sum(iou_list) / len(iou_list),
    }