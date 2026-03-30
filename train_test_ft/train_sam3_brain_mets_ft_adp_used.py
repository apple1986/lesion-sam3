# ap: freeze visual & text encoders, use Hydra config for optimizer, scheduler, training, different lrs for different parts
import sys, os, shutil
from sympy import use
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.extend([
    "/home/gxu/proj1/lesionSeg/utswlits3d_v2",
    "/home/gxu/proj1/lesionSeg/utswlits3d_v2/sam_methods/MedSAM2"
])

from data_load.brats_builder import load_data_brats_mets
from data_load.sam3_data import MRISliceDataset_v2, convert_to_batched_datapoint
from ap_sam3.eval_sam3 import evaluate_sam3

import sam3
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
from sam3 import build_sam3_image_model
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sam3.train.utils.train_utils import register_omegaconf_resolvers
# from sam3.train.optim.optimizer import construct_optimizer

# ------------------------
# Load Hydra config
# ------------------------
cfg = OmegaConf.load("train_configs/mets_ft_adp.yaml")

if not dist.is_initialized():
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method="tcp://127.0.0.1:29501",
        world_size=1,
        rank=0,
    )

# ------------------------
# Utility: plot loss curve
# ------------------------
def plot_loss(steps, losses, ckpt_dir):
    plt.figure(figsize=(8,5))
    plt.plot(steps, losses, linewidth=1)
    plt.xlabel("Optimizer Step")
    plt.ylabel("Total Loss")
    plt.title("SAM3 Training Loss")
    plt.grid(True)
    plt.savefig(os.path.join(ckpt_dir, "loss_curve.png"))
    plt.close()

# ------------------------
# Build SAM3 model
# ------------------------
def load_sam3(device, use_adapter=False):
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        eval_mode=False,
        enable_inst_interactivity=False,
        device=device,
    )
    if use_adapter:
        from ap_knife.adapters import insert_mlp_adapters_sam3
        insert_mlp_adapters_sam3(model, bottleneck=64, start_block=20)
        # freeze entire model
        for p in model.parameters():
            p.requires_grad = False

        # unfreeze adapters only
        for blk in model.backbone.vision_backbone.trunk.blocks:
            if hasattr(blk, "adapter"):
                for p in blk.adapter.parameters():
                    p.requires_grad = True
    model = model.to(device)
    return model

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
# ------------------------
# Main training
# ------------------------
def main():
    # seting hyper-parameters and save path
    USE_ADAPTER = True

    root_path = "/home/gxu/proj1/lesionSeg/sam3_brain"
    save_folder = cfg.training.save_folder
    ckpt_dir = os.path.join(root_path, "checkpoint_exp", save_folder)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Copy script for reproducibility
    shutil.copyfile(__file__, os.path.join(ckpt_dir, os.path.basename(__file__)))
    shutil.copyfile(os.path.join(root_path, "train_configs/mets_ft_adp.yaml"), os.path.join(ckpt_dir, "mets_ft_adp_used.yaml"))

    # Load loss, optimizer, scheduler from config
    loss_fn = instantiate(cfg.loss, _convert_="all")

    device = cfg.training.device
    batch_size = cfg.training.batch_size
    num_epochs = cfg.training.epochs
    ACCUM_STEPS = cfg.training.accum_steps
    scaler = torch.amp.GradScaler(enabled=cfg.training.mixed_precision == "fp16")

    # ------------------------
    # Load datasets
    # ------------------------
    train_dataset = MRISliceDataset_v2(load_data_brats_mets(root_path, split="train", modality_type="v2"))
    val_dataset = MRISliceDataset_v2(load_data_brats_mets(root_path, split="val", modality_type="v2"))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"✅ Training slices: {len(train_dataset)}, Validation slices: {len(val_dataset)}")

    # ------------------------
    # Build model
    # ------------------------
    model = load_sam3(device, use_adapter=USE_ADAPTER)
    for p in model.backbone.vision_backbone.parameters():
        p.requires_grad = not cfg.model.freeze_vision_encoder
    for p in model.backbone.language_backbone.parameters():
        p.requires_grad = not cfg.model.freeze_text_encoder
    model.train()

    if USE_ADAPTER: # only for the adapter finetuning case
        for p in model.parameters():
            p.requires_grad = False

        for blk in model.backbone.vision_backbone.trunk.blocks:
            if hasattr(blk, "adapter"):
                for p in blk.adapter.parameters():
                    p.requires_grad = True

    total, trainable = count_params(model)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    
    # ------------------------
    # Optimizer & Scheduler
    # ------------------------
    adapter_lr = cfg.scratch.adapter_lr

    adapter_params = []
    geometry_params = []
    transformer_params = []
    segmentation_params = []
    scoring_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "adapter" in name:
            adapter_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            # ======================
            # Adapter (HIGH LR)
            # ======================
            {
                "params": adapter_params,
                "lr": adapter_lr,
                # "weight_decay": 0.0,   # recommended for adapters
            },
        ],
        weight_decay=1e-4,
    )

    # checking learning rates
    for i, pg in enumerate(optimizer.param_groups):
        if any(p is adapter_params[0] for p in pg["params"]):
            print(f"Adapter group {i} LR = {pg['lr']}")

    # CRITICAL: remove frozen params
    for pg in optimizer.param_groups:
        pg["params"] = [p for p in pg["params"] if p.requires_grad]

    num_optim_params = sum(
        p.numel() for pg in optimizer.param_groups for p in pg["params"]
    )
    print(f"Params in optimizer: {num_optim_params:,}")

    optimizer.zero_grad(set_to_none=True)
    gradient_clipper = (
            instantiate(cfg.optim.gradient_clip) if cfg.optim else None
        )


    # ------------------------
    # Training loop
    # ------------------------
    loss_history = []
    step_history = []
    global_step = 0
    dice_val_best = -1.0
    accum_loss = 0.0
    accum_count = 0
    steps_per_epoch = len(train_loader) * batch_size // ACCUM_STEPS
    total_steps = steps_per_epoch * num_epochs

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        if epoch == 5: # reduce learning rate
            for pg in optimizer.param_groups:
                pg["lr"] *= 0.5
            print(f"🔻 Learning rate halved at epoch {epoch+1}")
            # checking learning rates
            for i, pg in enumerate(optimizer.param_groups):
                if any(p is adapter_params[0] for p in pg["params"]):
                    print(f"Adapter group {i} LR = {pg['lr']}")
            print("&"*50)

        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)

            # Forward each slice independently
            for i in range(imgs.shape[0]):
                img = imgs[i].unsqueeze(0)
                gt = masks[i].unsqueeze(0).unsqueeze(0)

                img_bd = convert_to_batched_datapoint(img, gt)

                with torch.amp.autocast(device, dtype=torch.float16):
                    out = model(img_bd)
                    find_targets = [model.back_convert(t) for t in img_bd.find_targets]
                    loss = loss_fn(out, find_targets)["core_loss"] / ACCUM_STEPS

                scaler.scale(loss).backward()
                iter_loss = loss.item()
                epoch_loss += iter_loss
                accum_loss += iter_loss
                accum_count += 1

                if accum_count == ACCUM_STEPS:
                    # Clipping gradients and detecting diverging gradients
                    if gradient_clipper is not None:
                        scaler.unscale_(optimizer)
                        gradient_clipper(model=model)     

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step = global_step + 1

                    # save loss
                    # print(f"[Step {global_step}] Loss: {accum_loss:.4f}")
                    with open(os.path.join(ckpt_dir, "train_loss_record_iter.txt"), "a") as f:
                            write_content = f'Epoch: [{epoch+1}/{num_epochs}]\tStep: [{global_step}]\tLoss: {accum_loss:.4f}\n'
                            f.write(write_content)

                    lrs = [pg["lr"] for pg in optimizer.param_groups]
                    # print(f"Step {global_step}: lr = {optimizer.param_groups[0]['lr']:.6e}")

                    loss_history.append(accum_loss)
                    step_history.append(global_step)
                    accum_loss = 0.0
                    accum_count = 0

                    pbar.set_postfix({
                                    "Step": global_step,
                                    "Loss": f"{loss_history[-1]:.4f}",
                                    "LR": f"{lrs[0]:.2e}"
                                })

        # Epoch-level logging
        print(f"✅ Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss:.4f}")
        with open(os.path.join(ckpt_dir, "train_loss_record_epoch.txt"), "a") as f:
            f.write(f"Epoch: [{epoch+1}/{num_epochs}]\tLoss: {epoch_loss:.4f}\n")

        # ------------------------
        # Validation
        # ------------------------
        metrics = evaluate_sam3(model, val_loader, device)
        dice_val_cur = metrics['dice']
        print(f"[Epoch {epoch+1}] Dice: {dice_val_cur:.4f}, IoU: {metrics['iou']:.4f}")

        with open(os.path.join(ckpt_dir, "eval_record_epoch.txt"), "a") as f:
            f.write(f"Epoch: [{epoch+1}/{num_epochs}], Dice: {dice_val_cur:.4f}, IoU: {metrics['iou']:.4f}\n")

        if dice_val_cur > dice_val_best:
            dice_val_best = dice_val_cur
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "sam3_mri_best_val.pth"))

    # Save the final model
    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"sam3_mri_epoch_{epoch+1}.pth"))

    # Plot final loss curve
    plot_loss(step_history, loss_history, ckpt_dir)


if __name__ == "__main__":
    register_omegaconf_resolvers()
    main()
    if dist.is_initialized():
        dist.destroy_process_group()
