import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np

import torch
import torch.nn.functional as F
import numpy as np
import cv2


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
    img: H×W×3 or H×W (will be converted to 3ch)
    return:
        {
          'backbone_fpn': [1×256×288×288,
                           1×256×144×144,
                           1×256×72×72]
        }
    """

    # ---- handle grayscale ----
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    img_t = torch.from_numpy(img).float().permute(2, 0, 1)[None].to(device)

    # SAM3 expects 1008×1008
    img_t = F.interpolate(img_t, size=(1008, 1008), mode="bilinear", align_corners=False)

    with torch.no_grad():
        feats_dict = model.backbone(img_t, "lesion")

    fpn_list = feats_dict["backbone_fpn"]

    # ---- normalize each level ----
    fpn_norm = [F.normalize(f, dim=1) for f in fpn_list]

    return fpn_norm


@torch.no_grad()
def extract_exemplar_prototype(
    img, mask, model, device, top_ratio=0.4
):

    feats_list = extract_sam_feature(img, model, device)
    mask_t = torch.from_numpy(mask).float().to(device)

    protos = []

    for feats in feats_list:
        _, C, Hf, Wf = feats.shape

        m = F.interpolate(mask_t[None,None], size=(Hf,Wf), mode="nearest")

        # ---- keep raw for energy ----
        raw = feats.clone()

        # cosine space
        feats = F.normalize(feats, dim=1)

        flat = feats.view(C, -1)
        raw_flat = raw.view(C, -1)

        valid = m.view(-1) > 0

        # ---------- small object fallback ----------
        if valid.sum() < 5:
            proto = flat.mean(dim=1, keepdim=True).T
            protos.append(F.normalize(proto, dim=1))
            continue

        vals = flat[:, valid]
        raw_vals = raw_flat[:, valid]

        # ---- energy BEFORE normalization ----
        energy = raw_vals.norm(dim=0)

        k = min(
            max(8, int(len(energy) * top_ratio)),
            256
        )

        idx = torch.topk(energy, k=k).indices

        # ---- foreground proto ----
        proto = vals[:, idx].mean(dim=1, keepdim=True).T
        proto = F.normalize(proto, dim=1)

        # ---- background contrast ----
        if (~valid).sum() > 10:
            bg = flat[:, ~valid]
            bg_proto = bg.mean(1, keepdim=True).T
            bg_proto = F.normalize(bg_proto, dim=1)

            proto = F.normalize(proto - 0.3 * bg_proto, dim=1)

        protos.append(proto)

    # ---- adaptive weights ----
    weights = torch.tensor([0.2, 0.3, 0.5], device=device)

    protos = torch.stack(protos, dim=0)

    fused = (protos * weights.view(-1,1,1)).sum(0)
    fused = F.normalize(fused, dim=1)

    return fused


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
    Multi-scale FPN aware proposal search.

    feat_map is a LIST:
        [1×256×288×288,
         1×256×144×144,
         1×256×72×72]

    We:
      1. compute similarity on each level
      2. upsample all to 288×288
      3. average → final sim map
    """

    H, W, _ = target_img.shape

    # -------------------------------------------------
    # 1. Feature extraction → LIST
    # -------------------------------------------------
    feat_list = extract_sam_feature(target_img, model, device)

    # exemplar prototype
    C = exemplar_proto.shape[1]
    kernel = exemplar_proto.view(C, 1, 1)

    # -------------------------------------------------
    # 2. similarity per level
    # -------------------------------------------------
    sim_maps = []

    for feat in feat_list:
        # feat: 1×256×h×w
        sim = F.conv2d(feat, kernel.unsqueeze(0))  # 1×1×h×w
        sim = sim.squeeze(0).squeeze(0)            # h×w
        sim_maps.append(sim)

    # -------------------------------------------------
    # 3. fuse to highest resolution (288×288)
    # -------------------------------------------------
    target_size = sim_maps[0].shape   # 288×288

    sim_maps_up = [
        F.interpolate(
            s[None, None],
            size=target_size,
            mode="bilinear",
            align_corners=False
        )[0, 0]
        for s in sim_maps
    ]

    sim_map = torch.stack(sim_maps_up, dim=0).mean(0)   # 288×288

    # -------------------------------------------------
    # 4. Top-K peaks
    # -------------------------------------------------
    Hf, Wf = sim_map.shape

    sim_flat = sim_map.flatten()
    topk_scores, topk_idx = torch.topk(sim_flat, k=topk)

    scale_x = W / Wf
    scale_y = H / Hf

    boxes = []
    scores = []

    sim_np = sim_map.detach().float().cpu().numpy()

    # -------------------------------------------------
    # 5. Adaptive bbox around each peak
    # -------------------------------------------------
    for idx, score in zip(topk_idx, topk_scores):

        y_f = int(idx // Wf)
        x_f = int(idx % Wf)

        peak_val = sim_np[y_f, x_f]

        thresh = peak_val * peak_ratio

        local_mask = (sim_np >= thresh).astype(np.uint8)

        num, labels = cv2.connectedComponents(local_mask)

        peak_label = labels[y_f, x_f]
        comp = (labels == peak_label).astype(np.uint8)

        ys, xs = np.where(comp > 0)

        # fallback
        if len(xs) < 3:
            xs = np.array([x_f])
            ys = np.array([y_f])

        x1f, x2f = xs.min(), xs.max()
        y1f, y2f = ys.min(), ys.max()

        # map to image space
        cx = int((x1f + x2f + 1) / 2 * scale_x)
        cy = int((y1f + y2f + 1) / 2 * scale_y)

        est_w = int((x2f - x1f + 1) * scale_x)
        est_h = int((y2f - y1f + 1) * scale_y)

        base_size = max(est_w, est_h)

        # multi-scale expansion
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
    # 6. Heatmap
    # -------------------------------------------------
    heatmap = cv2.resize(sim_np, (W, H), interpolation=cv2.INTER_LINEAR)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return boxes, scores, heatmap


# import torch
# import torch.nn.functional as F
# import numpy as np

# @torch.no_grad()
# def extract_exemplar_prototype(
#     img,
#     mask,
#     model,
#     device,
#     top_ratio=0.4,
#     N=1
# ):
#     """
#     Return spatial prototype kernel: 1 x 256 x N x N

#     From multi-level SAM features:
#         [1x256x288x288, 1x256x144x144, 1x256x72x72]

#     Steps:
#     1. Energy-based foreground selection
#     2. Find centroid in each level
#     3. Crop NxN patch as local descriptor
#     4. Fuse across levels
#     """

#     feats_list = extract_sam_feature(img, model, device)
#     mask_t = torch.from_numpy(mask).float().to(device)

#     kernels = []

#     for lvl, feats in enumerate(feats_list):

#         B, C, Hf, Wf = feats.shape

#         # ---- resize mask ----
#         m = F.interpolate(mask_t[None,None], size=(Hf, Wf), mode="nearest")

#         # raw + cosine
#         raw = feats.clone()
#         feats = F.normalize(feats, dim=1)

#         flat = feats.view(C, -1)
#         raw_flat = raw.view(C, -1)

#         valid = m.view(-1) > 0

#         # ===== 1) fallback =====
#         if valid.sum() < 5:
#             # center crop as safe prior
#             cy, cx = Hf//2, Wf//2
#             pad = N//2

#             y1, y2 = max(0,cy-pad), min(Hf,cy+pad+1)
#             x1, x2 = max(0,cx-pad), min(Wf,cx+pad+1)

#             patch = feats[:,:,y1:y2,x1:x2]

#             patch = F.interpolate(
#                 patch, size=(N,N), mode="bilinear", align_corners=False
#             )

#             kernels.append(F.normalize(patch, dim=1))
#             continue

#         # ===== 2) energy selection =====
#         vals = flat[:, valid]
#         raw_vals = raw_flat[:, valid]

#         energy = raw_vals.norm(dim=0)

#         k = min(
#             max(8, int(len(energy) * top_ratio)),
#             256
#         )

#         idx = torch.topk(energy, k=k).indices

#         # ===== 3) get spatial coords =====
#         all_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
#         sel = all_idx[idx]

#         ys = sel // Wf
#         xs = sel % Wf

#         # centroid of energetic region
#         cy = int(ys.float().mean().round())
#         cx = int(xs.float().mean().round())

#         # ===== 4) crop NxN patch =====
#         pad = N // 2

#         y1, y2 = max(0,cy-pad), min(Hf,cy+pad+1)
#         x1, x2 = max(0,cx-pad), min(Wf,cx+pad+1)

#         patch = feats[:,:,y1:y2,x1:x2]

#         # ensure exact NxN
#         patch = F.interpolate(
#             patch, size=(N,N), mode="bilinear", align_corners=False
#         )

#         # ===== 5) background contrast =====
#         if (~valid).sum() > 20:
#             bg = flat[:, ~valid].mean(1)
#             bg = F.normalize(bg[None,:,None,None], dim=1)

#             patch = F.normalize(patch - 0.2 * bg, dim=1)

#         kernels.append(patch)

#     # =====================================================
#     # 6) Fuse multi-level kernels
#     # =====================================================

#     weights = torch.tensor([0.2, 0.3, 0.5], device=device)

#     kernels = torch.stack(kernels, dim=0)   # 3 x 1 x 256 x N x N

#     fused = (kernels * weights.view(-1,1,1,1,1)).sum(0)

#     fused = F.normalize(fused, dim=1)

#     return fused   # 👉 1 x 256 x N x N

# @torch.no_grad()
# def find_proposal_bbox(
#     target_img,
#     exemplar_proto,   # 1×256×N×N  ← NEW
#     model,
#     device,
#     scales=(0.75, 1.0, 1.25),
#     topk=3,
#     peak_ratio=0.6,
# ):
#     """
#     Multi-scale FPN aware proposal search with SPATIAL prototype.

#     exemplar_proto: 1×256×N×N

#     feat_map is a LIST:
#         [1×256×288×288,
#          1×256×144×144,
#          1×256×72×72]

#     Steps:
#       1. correlate on each level with NxN kernel
#       2. upsample to highest resolution
#       3. average → final sim map
#       4. adaptive connected-component bbox
#     """

#     H, W, _ = target_img.shape

#     # -------------------------------------------------
#     # 1. Extract multi-level features
#     # -------------------------------------------------
#     feat_list = extract_sam_feature(target_img, model, device)

#     # -------------------------------------------------
#     # 2. Spatial correlation per level
#     # -------------------------------------------------
#     sim_maps = []

#     for feat in feat_list:
#         # feat: 1×256×h×w
#         # proto:1×256×N×N

#         sim = F.conv2d(
#             feat,
#             exemplar_proto,      # ✔ correct spatial kernel
#             padding="same"
#         )                        # 1×1×h×w

#         sim = sim.squeeze(0).squeeze(0)   # h×w
#         sim_maps.append(sim)

#     # -------------------------------------------------
#     # 3. Fuse to highest resolution
#     # -------------------------------------------------
#     target_size = sim_maps[0].shape   # 288×288

#     sim_maps_up = [
#         F.interpolate(
#             s[None, None],
#             size=target_size,
#             mode="bilinear",
#             align_corners=False
#         )[0, 0]
#         for s in sim_maps
#     ]

#     sim_map = torch.stack(sim_maps_up, dim=0).mean(0)   # Hf×Wf

#     # -------------------------------------------------
#     # 4. Top-K peaks
#     # -------------------------------------------------
#     Hf, Wf = sim_map.shape

#     sim_flat = sim_map.flatten()
#     topk_scores, topk_idx = torch.topk(sim_flat, k=topk)

#     scale_x = W / Wf
#     scale_y = H / Hf

#     boxes = []
#     scores = []

#     sim_np = sim_map.detach().float().cpu().numpy()

#     # -------------------------------------------------
#     # 5. Adaptive bbox around each peak
#     # -------------------------------------------------
#     for idx, score in zip(topk_idx, topk_scores):

#         y_f = int(idx // Wf)
#         x_f = int(idx % Wf)

#         peak_val = sim_np[y_f, x_f]

#         # threshold region around this peak
#         thresh = peak_val * peak_ratio

#         local_mask = (sim_np >= thresh).astype(np.uint8)

#         num, labels = cv2.connectedComponents(local_mask)

#         peak_label = labels[y_f, x_f]
#         comp = (labels == peak_label).astype(np.uint8)

#         ys, xs = np.where(comp > 0)

#         # ---- fallback ----
#         if len(xs) < 3:
#             xs = np.array([x_f])
#             ys = np.array([y_f])

#         x1f, x2f = xs.min(), xs.max()
#         y1f, y2f = ys.min(), ys.max()

#         # ---- map to image space ----
#         cx = int((x1f + x2f + 1) / 2 * scale_x)
#         cy = int((y1f + y2f + 1) / 2 * scale_y)

#         est_w = int((x2f - x1f + 1) * scale_x)
#         est_h = int((y2f - y1f + 1) * scale_y)

#         base_size = max(est_w, est_h)

#         # ---- multi-scale expansion ----
#         for s in scales:
#             win = int(base_size * s)
#             half = win // 2

#             x1 = max(0, cx - half)
#             y1 = max(0, cy - half)
#             x2 = min(W, cx + half)
#             y2 = min(H, cy + half)

#             if x2 > x1 and y2 > y1:
#                 boxes.append([x1, y1, x2, y2])
#                 scores.append(score.item())

#     # -------------------------------------------------
#     # 6. Heatmap for visualization
#     # -------------------------------------------------
#     heatmap = cv2.resize(sim_np, (W, H), interpolation=cv2.INTER_LINEAR)
#     heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

#     return boxes, scores, heatmap





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
