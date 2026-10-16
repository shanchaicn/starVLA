"""Frozen-backbone execution adapter; does not change the QwenPI_v3 architecture."""
import torch
from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3

def encode_frozen_qwen(self, batch_images, instructions):
    inputs=self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images,instructions=instructions)
    with torch.no_grad():
        outputs=self.qwen_vl_interface.model.model(**inputs,output_hidden_states=True,return_dict=True,use_cache=False)
    hidden=list(outputs.hidden_states[-self.num_action_dit_layers:])
    return self._project_vl_hidden_for_action([h.float() for h in hidden]),inputs.get("attention_mask")

class FrozenRGBBaseline(Qwen_PI_v3):
    """Original RGB/state/action behavior, with frozen VLM and float32 projectors/head."""
    _encode_vl_hidden_states=encode_frozen_qwen
    def __init__(self,config):
        super().__init__(config)
        self.qwen_vl_interface.requires_grad_(False).eval()
    def train(self,mode=True):
        super().train(mode);self.qwen_vl_interface.eval();return self
