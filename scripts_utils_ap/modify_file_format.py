import os
import numpy as np
from pathlib import Path

# ==================================================
# Paths & config
# ==================================================
model_name = "brats_mets_ft_part_vis_det_loss_seg_det"
root_path = "/home/gxu/proj1/lesionSeg/sam3_brain/checkpoint_exp/"+model_name+"/results/sam3_eval_text_exemplar"
txt_path = Path(root_path, "test_each_dice_hd95_nsd_asd_rand0.txt")

save_eval_path = root_path
NUM = 0   # or your experiment ID

# ==================================================
# 1. Read file & clean trailing empty lines
# ==================================================
with txt_path.open("r") as f:
    lines = [ln.rstrip("\n") for ln in f.readlines()]

while lines and lines[-1].strip() == "":
    lines.pop()


EXPECTED = 73
REMOVE_ID = "00223"

# --------------------------------------------------
# Decide whether to process or use directly
# --------------------------------------------------
if len(lines) >= EXPECTED:
    print(f"Detected {len(lines)} lines > {EXPECTED}, processing tail...")

    # ==================================================
    # 2. Take last EXPECTED lines
    # ==================================================
    last_lines = lines[-EXPECTED:]

    # ==================================================
    # 3. Sanity check IDs
    # ==================================================
    assert "00189" in last_lines[0], f"First line mismatch: {last_lines[0]}"
    assert "00801" in last_lines[-1], f"Last line mismatch: {last_lines[-1]}"

    # ==================================================
    # 4. Remove specific case
    # ==================================================
    filtered_lines = [ln for ln in last_lines if REMOVE_ID not in ln]
    assert len(filtered_lines) == EXPECTED - 1, (
        f"Expected {EXPECTED - 1} lines, got {len(filtered_lines)}"
    )

    # ==================================================
    # 5. Overwrite original file
    # ==================================================
    with txt_path.open("w") as f:
        for ln in filtered_lines:
            f.write(ln + "\n")

    lines_to_use = filtered_lines

    print("✔ File updated successfully")
    print(f"  Lines written: {len(filtered_lines)}")
    print(f"  Removed ID: {REMOVE_ID}")

else:
    print(f"Detected {len(lines)} lines < {EXPECTED}, using file directly")
    lines_to_use = lines


# ==================================================
# 6. Parse metrics
# ==================================================
dice_all, hd95_all, nsd_all, asd_all = [], [], [], []

for ln in lines_to_use:
    parts = ln.replace(",", " ").split()

    metrics = {}
    for i in range(len(parts) - 1):
        key = parts[i].rstrip(":")
        val = parts[i + 1]
        if key in {"dice", "hd95", "nsd", "asd"}:
            metrics[key] = float(val)

    # sanity check
    missing = {"dice", "hd95", "nsd", "asd"} - metrics.keys()
    if missing:
        raise ValueError(f"Missing metrics {missing} in line:\n{ln}")

    dice_all.append(metrics["dice"])
    hd95_all.append(metrics["hd95"])
    nsd_all.append(metrics["nsd"])
    asd_all.append(metrics["asd"])


# ==================================================
# 7. Save results (standard format)
# ==================================================
## save results
with open(os.path.join(save_eval_path, "test_mean_dice_hd_nsd_asd"+str(NUM)+".txt"), "w") as f:
    write_content = f"{'#'*60}\nmean dice\tmean hd95\tmean nsd\tmean asd\t\n"
    write_content = write_content + f"{np.array(dice_all).mean():.4f}±{np.array(dice_all).std():.2f}\t" \
                                    f"{np.array(hd95_all).mean():.4f}±{np.array(hd95_all).std():.2f}\t" \
                                    f"{np.array(nsd_all).mean():.4f}±{np.array(nsd_all).std():.4f}\t" \
                                    f"{np.array(asd_all).mean():.4f}±{np.array(asd_all).std():.4f}\t\n"
    f.write(write_content)

## save format results
with open(os.path.join(save_eval_path, "format_test_mean_dice_hd_nsd_asd"+str(NUM)+".txt"), "w") as f:
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

print("✔ Statistics saved successfully")
check_file = os.path.join(save_eval_path, "format_test_mean_dice_hd_nsd_asd"+str(NUM)+".txt")
print(f"  Path: {check_file}")
