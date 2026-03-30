import torch
from sam3 import build_sam3_image_model
import sys
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2")
sys.path.append("/home/gxu/proj1/lesionSeg/sam3_brain")

@torch.no_grad()
def load_sam3(sam3_checkpoint=None, device="cuda:1", use_adapter=False, use_deform_adapter=False, 
              use_dd_adapter=False, start_block=20, end_block=None):
    bpe_path = f"/home/gxu/proj1/lesionSeg/sam3_brain/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=bpe_path, 
                                   device=device,
                                   enable_inst_interactivity=True, # False
                                   checkpoint_path=sam3_checkpoint,
                                   eval_mode=True)
    if use_adapter:
        if use_deform_adapter:
            from ap_knife.adapters import insert_deform_adapters_sam3
            insert_deform_adapters_sam3(model, bottleneck=64, start_block=20)
        elif use_dd_adapter:
            from ap_knife.adapters import insert_dd_adapters_sam3
            insert_dd_adapters_sam3(model, start_block=start_block, end_block=end_block)
        else:
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
    state_dict = torch.load(sam3_checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model
