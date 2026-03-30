import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np

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
    feat = F.normalize(feats, dim=1) # 1x256x72x72
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


def extract_exemplar_prototype_v2(img, mask, model, device, top_ratio=0.5):
    feats = extract_sam_feature(img, model, device)
    feats = F.normalize(feats, dim=1)

    _, C, Hf, Wf = feats.shape

    mask_t = torch.from_numpy(mask).float().to(device)
    mask_t = F.interpolate(mask_t[None,None], size=(Hf, Wf), mode="nearest")

    masked = feats * mask_t

    # --- robust selection ---
    flat = masked.view(C, -1)
    valid = mask_t.view(-1) > 0

    if valid.sum() < 5:
        return F.normalize(flat.mean(dim=1, keepdim=True), dim=0)[None]

    vals = flat[:, valid]

    # select top energy pixels
    energy = vals.norm(dim=0)
    k = max(1, int(len(energy) * top_ratio))
    idx = torch.topk(energy, k=k).indices

    proto = vals[:, idx].mean(dim=1, keepdim=True).T
    proto = F.normalize(proto, dim=1)

    return proto


@torch.no_grad()
def extract_exemplar_prototype_v3(
    img,
    mask,
    model,
    device,
    top_ratio=0.5,
    min_pixels=5,
):
    """
    Robust Top-P prototype extraction.

    Args:
        img: H×W×3 numpy
        mask: H×W binary
        top_ratio: keep top % highest-energy pixels

    Returns:
        proto: (1, C) normalized vector
    """

    # -------------------------------------------------
    # 1. Extract feature
    # -------------------------------------------------
    feats = extract_sam_feature(img, model, device)   # 1×C×Hf×Wf
    feats = F.normalize(feats, dim=1)                 # channel-wise norm

    _, C, Hf, Wf = feats.shape

    # -------------------------------------------------
    # 2. Resize mask to feature space
    # -------------------------------------------------
    mask_t = torch.from_numpy(mask).float().to(device)
    mask_t = F.interpolate(
        mask_t[None, None],
        size=(Hf, Wf),
        mode="nearest"
    )  # 1×1×Hf×Wf

    # -------------------------------------------------
    # 3. Collect valid pixels
    # -------------------------------------------------
    masked = feats * mask_t                           # 1×C×Hf×Wf

    flat = masked.view(1, C, -1)                      # 1×C×N
    valid = mask_t.view(-1) > 0                       # N

    # ---------- fallback ----------
    if valid.sum() < min_pixels:
        proto = flat.mean(dim=2)                      # 1×C
        return F.normalize(proto, dim=1)

    vals = flat[:, :, valid]                          # 1×C×Nv

    # -------------------------------------------------
    # 4. Top-P pooling by feature energy
    # -------------------------------------------------
    # L2 energy per pixel
    energy = torch.norm(vals, dim=1)                  # 1×Nv

    k = max(1, int(energy.numel() * top_ratio))

    idx = torch.topk(energy, k=k, dim=1).indices      # 1×k

    # gather top pixels
    top_vals = torch.gather(
        vals,
        2,
        idx.unsqueeze(1).expand(-1, C, -1)
    )                                                 # 1×C×k

    # -------------------------------------------------
    # 5. Prototype = mean of top pixels
    # -------------------------------------------------
    proto = top_vals.mean(dim=2)                      # 1×C
    proto = F.normalize(proto, dim=1)

    return proto



## proposal bounding box
def find_proposal_bbox(
    target_img,
    exemplar_proto,
    model,
    device,
    scales=(0.75, 1.0, 1.25),
    topk=3,
    peak_ratio=0.6,
):
    """
    Adaptive window from similarity map instead of fixed base_win_size.

    Steps:
    1. Compute cosine similarity map
    2. Get Top-K peaks
    3. For each peak, estimate adaptive region via thresholding around peak
    4. Convert to bbox and apply multi-scale expansion

    Returns:
        boxes: List[[x1,y1,x2,y2]]
        scores: List[float]
        heatmap_img: H×W numpy similarity heatmap
    """
    H, W, _ = target_img.shape

    # -------------------------------------------------
    # 1. Feature extraction
    # -------------------------------------------------
    feat_map = extract_sam_feature(target_img, model, device)  # 1×C×Hf×Wf
    _, C, Hf, Wf = feat_map.shape

    # -------------------------------------------------
    # 2. Cosine similarity
    # -------------------------------------------------
    kernel = exemplar_proto.view(C, 1, 1)
    sim_map = F.conv2d(feat_map, kernel.unsqueeze(0))  # 1×1×Hf×Wf
    sim_map = sim_map.squeeze()                        # Hf×Wf

    # -------------------------------------------------
    # 3. Top-K peaks
    # -------------------------------------------------
    sim_flat = sim_map.flatten()
    topk_scores, topk_idx = torch.topk(sim_flat, k=topk)

    scale_x = W / Wf
    scale_y = H / Hf

    boxes = []
    scores = []

    sim_np = sim_map.detach().float().cpu().numpy()

    # -------------------------------------------------
    # 4. For each peak → adaptive bbox
    # -------------------------------------------------
    for idx, score in zip(topk_idx, topk_scores):

        y_f = int(idx // Wf)
        x_f = int(idx % Wf)

        peak_val = sim_np[y_f, x_f]

        # ---- adaptive threshold around this peak ----
        thresh = peak_val * peak_ratio

        # binary mask in feature space
        local_mask = (sim_np >= thresh).astype(np.uint8)

        # keep only component connected to the peak
        num, labels = cv2.connectedComponents(local_mask)

        peak_label = labels[y_f, x_f]
        comp = (labels == peak_label).astype(np.uint8)

        ys, xs = np.where(comp > 0)

        # fallback: single point if too small
        if len(xs) < 3:
            xs = np.array([x_f])
            ys = np.array([y_f])

        # bbox in feature space
        x1f, x2f = xs.min(), xs.max()
        y1f, y2f = ys.min(), ys.max()

        # map to image space
        cx = int((x1f + x2f + 1) / 2 * scale_x)
        cy = int((y1f + y2f + 1) / 2 * scale_y)

        # estimated window size from similarity extent
        est_w = int((x2f - x1f + 1) * scale_x)
        est_h = int((y2f - y1f + 1) * scale_y)

        base_size = max(est_w, est_h)

        # -------------------------------------------------
        # 5. Apply multi-scale expansion
        # -------------------------------------------------
        for s in scales:
            win = int(base_size * s)
            half = win // 2

            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(W, cx + half)
            y2 = min(H, cy + half)

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                scores.append(score.item())

    # -------------------------------------------------
    # 6. Heatmap for visualization
    # -------------------------------------------------
    heatmap = sim_np
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
