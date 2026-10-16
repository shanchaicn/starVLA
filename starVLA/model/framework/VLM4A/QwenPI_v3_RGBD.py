"""Additive QwenPI_v3 extension; the upstream RGB framework remains untouched."""
import numpy as np
import torch
from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.rgbd.depth import DepthCNN, FrozenLingBot, LayerwiseDepthFusion

@FRAMEWORK_REGISTRY.register("QwenPI_v3_RGBD")
class QwenPI_v3_RGBD(Qwen_PI_v3):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        cfg=self.config.framework.rgbd
        if cfg.encoder == "cnn": self.depth_encoder=DepthCNN()
        elif cfg.encoder == "lingbot":
            self.depth_encoder=FrozenLingBot(cfg.checkpoint,cfg.get("num_tokens",1200))
        else: raise ValueError(f"Unknown depth encoder: {cfg.encoder}")
        self.depth_fusion=LayerwiseDepthFusion(self.depth_encoder.output_dim,self.action_dit_hidden_dim,self.num_action_dit_layers)
        self.qwen_vl_interface.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.qwen_vl_interface.eval()
        return self

    from starVLA.rgbd.backbone import encode_frozen_qwen as _encode_vl_hidden_states

    def _conditions(self, examples):
        device=next(self.depth_fusion.parameters()).device
        instructions=self.add_discretized_state_to_instruction([e["lang"] for e in examples],[e["state"] for e in examples])
        vl,mask=self._encode_vl_hidden_states([e["image"] for e in examples],instructions)
        depth=torch.as_tensor(np.stack([e["depth"] for e in examples]),device=device,dtype=torch.float32)
        valid=torch.as_tensor(np.stack([e["depth_mask"] for e in examples]),device=device,dtype=torch.bool)
        if depth.ndim!=4 or depth.shape[1]!=2 or valid.shape!=depth.shape:
            raise ValueError("Depth and mask must be [B,2,H,W]")
        if not torch.isfinite(depth).all() or (depth<0).any(): raise ValueError("Expected sanitized metric depth")
        rgb=K=None
        if isinstance(self.depth_encoder,FrozenLingBot):
            rgb=torch.as_tensor(np.stack([[np.asarray(im) for im in e["image"]] for e in examples]),device=device,dtype=torch.float32).permute(0,1,4,2,3)/255
            K=torch.as_tensor(np.stack([e["intrinsics"] for e in examples]),device=device,dtype=torch.float32)
        tokens=self.depth_encoder(depth,valid,rgb,K)
        return self.depth_fusion(vl,tokens,mask,valid.flatten(2).any(-1))

    def forward(self, examples=None, **kwargs):
        vl,mask=self._conditions(examples)
        actions=torch.as_tensor(np.stack([e["action"] for e in examples]),device=vl[0].device,dtype=torch.float32)
        if actions.shape[1:]!=(self.action_horizon,7): raise ValueError("Wrong action chunk shape")
        repeat=int(self.config.trainer.get("repeated_diffusion_steps",1))
        return {"action_loss":self.action_model([h.repeat(repeat,1,1) for h in vl],actions.repeat(repeat,1,1),
                                                encoder_attention_mask=mask.repeat(repeat,1))}
    @torch.inference_mode()
    def predict_action(self, examples=None, **kwargs):
        vl,mask=self._conditions(examples)
        action=self.action_model.predict_action(vl,encoder_attention_mask=mask)
        return {"normalized_actions":action.float().cpu().numpy()}
