import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from scipy.optimize import linear_sum_assignment
@torch.no_grad()
def hungarian_match(pred_boxes, gt_boxes):
    """
    pred_boxes: (N,4)
    gt_boxes:   (M,4)
    return: list of (pred_idx, gt_idx)
    """

    N, M = len(pred_boxes), len(gt_boxes)

    cost = torch.zeros(N, M, device=pred_boxes.device)

    # cost = 1 - IoU + L1
    for i in range(N):
        for j in range(M):
            cost[i,j] = giou_loss(
                pred_boxes[i:i+1],
                gt_boxes[j:j+1]
            ) + F.l1_loss(pred_boxes[i], gt_boxes[j])

    cost = cost.cpu().numpy()
    row, col = linear_sum_assignment(cost)

    return list(zip(row, col))

def cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5*w, cy - 0.5*h, cx + 0.5*w, cy + 0.5*h], dim=-1)

def giou_loss(pred, target):
    p = cxcywh_to_xyxy(pred)
    t = cxcywh_to_xyxy(target)

    x1 = torch.max(p[:,0], t[:,0])
    y1 = torch.max(p[:,1], t[:,1])
    x2 = torch.min(p[:,2], t[:,2])
    y2 = torch.min(p[:,3], t[:,3])

    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area_p = (p[:,2]-p[:,0]).clamp(0)*(p[:,3]-p[:,1]).clamp(0)
    area_t = (t[:,2]-t[:,0]).clamp(0)*(t[:,3]-t[:,1]).clamp(0)
    union = area_p + area_t - inter + 1e-6
    iou = inter / union

    cx1 = torch.min(p[:,0], t[:,0])
    cy1 = torch.min(p[:,1], t[:,1])
    cx2 = torch.max(p[:,2], t[:,2])
    cy2 = torch.max(p[:,3], t[:,3])
    area_c = (cx2-cx1).clamp(0)*(cy2-cy1).clamp(0) + 1e-6

    giou = iou - (area_c - union)/area_c
    return (1-giou).mean()



def box_iou_cxcywh(pred, gt):
    p = cxcywh_to_xyxy(pred)
    t = cxcywh_to_xyxy(gt)

    x1 = torch.max(p[...,0], t[...,0])
    y1 = torch.max(p[...,1], t[...,1])
    x2 = torch.min(p[...,2], t[...,2])
    y2 = torch.min(p[...,3], t[...,3])

    inter = (x2-x1).clamp(0) * (y2-y1).clamp(0)

    area_p = (p[...,2]-p[...,0]).clamp(0)*(p[...,3]-p[...,1]).clamp(0)
    area_t = (t[...,2]-t[...,0]).clamp(0)*(t[...,3]-t[...,1]).clamp(0)

    union = area_p + area_t - inter + 1e-6

    return inter / union        # shape same as pred


def quality_loss(pred_score, pred_box, gt_box):

    with torch.no_grad():
        iou = box_iou_cxcywh(pred_box, gt_box)   # shape [1]

    return F.mse_loss(torch.sigmoid(pred_score), iou)



def focal_obj_loss(logits, targets, alpha=0.25, gamma=2.0):
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob*targets + (1-prob)*(1-targets)
    loss = ce*((1-p_t)**gamma)
    a_t = alpha*targets + (1-alpha)*(1-targets)
    return (a_t*loss).mean()

def varifocal_loss(pred, iou):
    p = torch.sigmoid(pred)
    return (iou * (iou - p).abs()).mean()