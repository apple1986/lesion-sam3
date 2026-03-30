import monai
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDiceMetric, SurfaceDistanceMetric
import numpy as np
from scipy import ndimage
## define evaluation functions
def get_dice_hd_nsd_asd():
    dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch") # each segmented object
    hd95_metric_batch = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch", get_not_nans=False)
    nsd_metric_batch = SurfaceDiceMetric(include_background=False, class_thresholds=[0.5,], reduction="mean_batch", get_not_nans=False) # each segmented object
    asd_metric_batch  = SurfaceDistanceMetric(include_background=False, reduction="mean_batch",get_not_nans=False) # each segmented object
    return dice_metric_batch, hd95_metric_batch, nsd_metric_batch, asd_metric_batch



def dice_score(pred, gt, eps=1e-6):
    inter = (pred & gt).sum()
    union = pred.sum() + gt.sum()
    return (2 * inter + eps) / (union + eps)

def iou_score(pred, gt, eps=1e-6):
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return (inter + eps) / (union + eps)


# detection accuracy 2D
def detection_accuracy_2d(gt_mask, pred_mask):
    """
    Compute 2D slice-level detection accuracy.

    Args:
        gt_mask   (H, W): binary ground truth mask
        pred_mask (H, W): binary prediction mask

    Returns:
        accuracy: float
        n_gt:     number of GT objects
        n_detect: number of correctly detected GT objects
    """

    # Label connected components
    gt_labeled, n_gt = ndimage.label(gt_mask)
    pred_labeled, n_pred = ndimage.label(pred_mask)

    if n_gt == 0:
        return 1.0, 0, 0  # no GT → perfect detection by convention

    # Compute centroids of predicted objects
    pred_centroids = ndimage.center_of_mass(
        pred_mask, pred_labeled, range(1, n_pred + 1)
    )

    # Track detected GT objects
    detected_gt_ids = set()

    for cy, cx in pred_centroids:
        if np.isnan(cx) or np.isnan(cy):
            continue

        y, x = int(round(cy)), int(round(cx))

        # Ensure index safety
        if 0 <= y < gt_mask.shape[0] and 0 <= x < gt_mask.shape[1]:
            gt_id = gt_labeled[y, x]
            if gt_id > 0:
                detected_gt_ids.add(gt_id)

    n_detect = len(detected_gt_ids)
    accuracy = n_detect / n_gt

    return accuracy, n_gt, n_detect

def detection_accuracy_3d(gt_vol, pred_vol, connectivity=1):
    """
    Compute 3D volume-level lesion detection accuracy.

    Args:
        gt_vol   (D, H, W): binary ground-truth volume
        pred_vol (D, H, W): binary predicted volume
        connectivity: 1 (6-connectivity) or 2 (18/26-connectivity)

    Returns:
        accuracy: float
        n_gt: number of GT lesions
        n_detect: number of detected GT lesions
    """

    # Define connectivity
    structure = ndimage.generate_binary_structure(3, connectivity)

    # Label GT and prediction volumes
    gt_labeled, n_gt = ndimage.label(gt_vol, structure=structure)
    pred_labeled, n_pred = ndimage.label(pred_vol, structure=structure)

    if n_gt == 0:
        return 1.0, 0, 0  # no GT lesions

    # Compute centroids of predicted objects
    pred_centroids = ndimage.center_of_mass(
        pred_vol, pred_labeled, range(1, n_pred + 1)
    )

    detected_gt_ids = set()

    for cz, cy, cx in pred_centroids:
        if np.isnan(cx) or np.isnan(cy) or np.isnan(cz):
            continue

        z, y, x = int(round(cz)), int(round(cy)), int(round(cx))

        # Boundary check
        if (0 <= z < gt_vol.shape[0] and
            0 <= y < gt_vol.shape[1] and
            0 <= x < gt_vol.shape[2]):

            gt_id = gt_labeled[z, y, x]
            if gt_id > 0:
                detected_gt_ids.add(gt_id)

    n_detect = len(detected_gt_ids)
    accuracy = n_detect / n_gt

    return accuracy, n_gt, n_detect