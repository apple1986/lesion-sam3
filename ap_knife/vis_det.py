import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision.utils import make_grid

def draw_boxes_on_image(img, boxes, gt_boxes=None):
    """
    img: H×W×3 numpy (0-1)
    boxes: (num_q,4) in cxcywh normalized
    gt_boxes: (N,4) in xyxy absolute
    """
    H, W = img.shape[:2]
    canvas = (img * 255).astype(np.uint8).copy()

    # draw GT
    if gt_boxes is not None:
        for b in gt_boxes:
            x1,y1,x2,y2 = map(int,b)
            cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,255,0),2)

    # draw each query box
    for q,b in enumerate(boxes):
        cx,cy,w,h = b
        x1 = int((cx-w/2)*W)
        y1 = int((cy-h/2)*H)
        x2 = int((cx+w/2)*W)
        y2 = int((cy+h/2)*H)

        # color per query (just for visibility)
        color = (255, 50*q % 255, 150)
        cv2.rectangle(canvas,(x1,y1),(x2,y2),color,1)

    return canvas


def visualize_batch(imgs, pred_boxes, gt_boxes, epoch, step, SAVE_VIS):
    vis = []
    for i in range(len(imgs)):
        vis.append(
            draw_boxes_on_image(imgs[i], pred_boxes[i], gt_boxes[i])
        )

    grid = make_grid(
        torch.from_numpy(np.stack(vis)).permute(0,3,1,2),
        nrow=4
    )
    grid = grid.permute(1,2,0).numpy()

    cv2.imwrite(
        f"{SAVE_VIS}/epoch_s{step}.png",
        cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    )
