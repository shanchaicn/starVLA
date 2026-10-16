"""Two cameras -> 32 tokens each -> 64 spatial tokens, with explicit view/position."""
import types
import torch
from torch import nn
from torch.nn import functional as F

class DepthCNN(nn.Module):
    output_dim = 128
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(2,32,5,2,2),nn.GroupNorm(4,32),nn.SiLU(),
                                 nn.Conv2d(32,64,3,2,1),nn.GroupNorm(8,64),nn.SiLU(),
                                 nn.Conv2d(64,128,3,2,1),nn.GroupNorm(8,128),nn.SiLU())

    def forward(self, depth, mask, rgb=None, intrinsics=None):
        b,v,h,w = depth.shape
        if v != 2: raise ValueError("Expected agentview and wrist")
        # Metric information is preserved in log1p; invalid values have a separate channel.
        d = torch.where(mask,depth,0)
        x = torch.stack((torch.log1p(d),mask.to(d.dtype)),2).reshape(b*v,2,h,w)
        x = self.cnn(x)
        return F.adaptive_avg_pool2d(x,(4,8)).flatten(2).transpose(1,2).reshape(b,64,-1)

class FrozenLingBot(nn.Module):
    def __init__(self, checkpoint, num_tokens=1200):
        super().__init__()
        from mdm.model.v2 import MDMModel
        # Audit exact weights rather than silently accepting missing pretrained tensors.
        ckpt = torch.load(checkpoint,map_location="cpu",weights_only=True)
        self.model = MDMModel(**ckpt["model_config"])
        self.model.load_state_dict(ckpt["model"],strict=True)
        self.model.enable_pytorch_native_sdpa()
        # Upstream forward_feat sends a list to NestedTensorBlock. Its list path
        # still requires xFormers even after the official SDPA switch. Process each
        # list member independently: equivalent to block-diagonal attention in eval,
        # preserves all pretrained parameters and avoids compiled CUDA extensions.
        from mdm.model.dinov2_rgbd.layers.block import NestedTensorBlock
        for block in self.model.encoder.backbone.modules():
            if isinstance(block, NestedTensorBlock):
                block.forward = types.MethodType(_dense_nested_forward, block)
        self.model.requires_grad_(False).eval()
        self.output_dim = self.model.encoder.output_projections[0].out_channels
        self.num_tokens = num_tokens

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, depth, mask, rgb=None, intrinsics=None):
        b,v,h,w = depth.shape
        if rgb is None or rgb.shape != (b,v,3,h,w):
            raise ValueError("LingBot needs aligned RGB in [B,2,3,H,W]")
        if intrinsics is None or intrinsics.shape != (b,v,3,3) or not torch.isfinite(intrinsics).all():
            raise ValueError("Missing/invalid pixel intrinsics")
        if (intrinsics[...,0,0].abs() < 1e-6).any() or (intrinsics[...,1,1].abs() < 1e-6).any():
            raise ValueError("Invalid focal length")
        # Official forward_feat bypasses neck/depth_head. K is validated, but this path
        # does not consume it; do not pretend to feed intrinsics into the encoder.
        with torch.no_grad():
            feat,_ = self.model.forward_feat(
                rgb.reshape(b*v,3,h,w).float(), num_tokens=self.num_tokens,
                depth=torch.where(mask,depth,0).reshape(b*v,h,w).float())
            feat = F.adaptive_avg_pool2d(feat,(4,8)).flatten(2).transpose(1,2)
        return feat.reshape(b,64,-1)

class LayerwiseDepthFusion(nn.Module):
    def __init__(self, feature_dim, hidden_dim, layers):
        super().__init__()
        self.projector = nn.Sequential(nn.LayerNorm(feature_dim),nn.Linear(feature_dim,hidden_dim))
        self.position = nn.Parameter(torch.randn(1,64,hidden_dim)*.02)
        self.gates = nn.Parameter(torch.full((layers,),.1))

    def forward(self, vl_layers, tokens, attention_mask, valid_views):
        tokens = self.projector(tokens) + self.position
        valid = valid_views.repeat_interleave(32,dim=1)
        if attention_mask is None:
            attention_mask = torch.ones(vl_layers[0].shape[:2],device=tokens.device,dtype=torch.bool)
        out=[torch.cat((vl.float(),torch.tanh(self.gates[i])*tokens),dim=1) for i,vl in enumerate(vl_layers)]
        return out,torch.cat((attention_mask.bool(),valid),dim=1)


def _dense_nested_forward(block, inputs):
    from mdm.model.dinov2_rgbd.layers.block import Block
    if block.training:
        raise RuntimeError("Dense LingBot adapter is frozen/eval only")
    if isinstance(inputs,list):
        return [Block.forward(block,x) for x in inputs]
    return Block.forward(block,inputs)
