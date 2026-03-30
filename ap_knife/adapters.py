import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

class BottleneckAdapter(nn.Module):
    def __init__(self, dim=1024, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim, bias=False)

        # zero-init so pretrained behavior is preserved
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))

def insert_mlp_adapters_sam3(
    model,
    bottleneck=64,
    start_block=20,   # only last blocks
):
    trunk = model.backbone.vision_backbone.trunk
    blocks = trunk.blocks

    for i in range(start_block, len(blocks)):
        blk = blocks[i]

        if hasattr(blk, "adapter"):
            continue

        blk.adapter = BottleneckAdapter(
            dim=blk.mlp.fc2.out_features,
            bottleneck=bottleneck
        )

        # wrap forward safely
        old_forward = blk.forward

        def forward_with_adapter(x, blk=blk, old_forward=old_forward):
            x = old_forward(x)
            x = blk.adapter(x)
            return x

        blk.forward = forward_with_adapter

class DeformConvAdapter(nn.Module):
    def __init__(self, dim, bottleneck=64, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2

        # channel reduction
        self.down = nn.Conv2d(dim, bottleneck, kernel_size=1, bias=False)
        self.act = nn.GELU()

        # offset prediction
        self.offset = nn.Conv2d(
            bottleneck,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            padding=padding
        )

        # deformable conv
        self.deform = DeformConv2d(
            bottleneck,
            bottleneck,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        # channel restore
        self.up = nn.Conv2d(bottleneck, dim, kernel_size=1, bias=False)

        # zero-init → preserve pretrained behavior
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x):
        """
        x: [B, H, W, C]   (BHWC)
        """
        identity = x

        # BHWC → BCHW
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.down(x)
        x = self.act(x)

        offset = self.offset(x)
        x = self.deform(x, offset)

        x = self.up(x)

        # BCHW → BHWC
        x = x.permute(0, 2, 3, 1).contiguous()

        return identity + x

def insert_deform_adapters_sam3(
    model,
    bottleneck=64,
    start_block=20,
):
    trunk = model.backbone.vision_backbone.trunk
    blocks = trunk.blocks

    for i in range(start_block, len(blocks)):
        blk = blocks[i]

        if hasattr(blk, "adapter"):
            continue

        dim = blk.mlp.fc2.out_features  # usually 1024

        blk.adapter = DeformConvAdapter(
            dim=dim,
            bottleneck=bottleneck
        )

        old_forward = blk.forward

        def forward_with_adapter(x, blk=blk, old_forward=old_forward):
            x = old_forward(x)    # x is BHWC
            x = blk.adapter(x)    # still BHWC
            return x

        blk.forward = forward_with_adapter


# DD-Adapter insertion functions for SAM-3

class DD_Adapter(nn.Module):
    def __init__(self, input_dim, adapter_dim=64):
        super().__init__()

        self.pw1 = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // 2, kernel_size=1),
            nn.GELU(),
        )

        self.dw2_1 = nn.Sequential(
            nn.Conv2d(
                input_dim // 2,
                input_dim // 2,
                kernel_size=3,
                padding=1,
                dilation=1,
                groups=input_dim // 2,
            ),
            nn.GELU(),
        )

        self.dw2_2 = nn.Sequential(
            nn.Conv2d(
                input_dim // 2,
                input_dim // 2,
                kernel_size=3,
                padding=3,
                dilation=3,
                groups=input_dim // 2,
            ),
            nn.GELU(),
        )

        self.pw2 = nn.Sequential(
            nn.Conv2d(input_dim // 2, input_dim, kernel_size=1),
            nn.GELU(),
        )

        # zero-init to preserve pretrained behavior
        nn.init.zeros_(self.pw2[0].weight)

    def forward(self, x):
        """
        x: [B, H, W, C]  (BHWC)
        """
        identity = x

        # BHWC → BCHW
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.pw1(x)
        x = self.dw2_1(x) + self.dw2_2(x)
        x = self.pw2(x)

        # BCHW → BHWC
        x = x.permute(0, 2, 3, 1).contiguous()

        return identity + x

def insert_dd_adapters_sam3(
    model,
    start_block=20,
    end_block=None,
):
    """
    Insert DD-Adapter into SAM-3 vision backbone blocks.
    """
    trunk = model.backbone.vision_backbone.trunk
    blocks = trunk.blocks

    if end_block is None:
        end_block = len(blocks)

    for i in range(start_block, end_block):
        blk = blocks[i]

        # avoid double insertion
        if hasattr(blk, "adapter"):
            continue

        dim = blk.mlp.fc2.out_features  # usually 1024

        blk.adapter = DD_Adapter(input_dim=dim)

        old_forward = blk.forward

        def forward_with_adapter(x, blk=blk, old_forward=old_forward):
            x = old_forward(x)    # BHWC
            x = blk.adapter(x)    # BHWC
            return x

        blk.forward = forward_with_adapter
