import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("STARVLA_DISABLE_DEEPSPEED", "1")

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from transformers import GPT2Config, GPT2LMHeadModel

from starVLA.model.modules.vlm.lora_utils import (
    apply_lora_to_vlm_backbone,
    get_lora_settings,
    resolve_vlm_lora_adapter_dir,
    save_vlm_lora_adapter,
)
from starVLA.training.train_starvla import VLATrainer, _strip_lora_module_from_freeze_modules
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups


class TinyInterface(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = GPT2LMHeadModel(
            GPT2Config(n_layer=1, n_head=1, n_embd=16, vocab_size=32, n_positions=8)
        )


class TinyVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen_vl_interface = TinyInterface()
        self.action_model = nn.Linear(32, 4)

    def forward(self, input_ids, labels):
        output = self.qwen_vl_interface.model(input_ids=input_ids, labels=labels)
        action = self.action_model(output.logits.mean(dim=1))
        return output.loss + action.square().mean()


def lora_config(adapter_name="default", adapter_dir_name="vlm_lora_adapter"):
    return OmegaConf.create(
        {
            "framework": {
                "vlm": {
                    "lora": {
                        "enabled": True,
                        "r": 2,
                        "alpha": 4,
                        "dropout": 0.0,
                        "target_modules": ["c_attn"],
                        "adapter_name": adapter_name,
                        "adapter_dir_name": adapter_dir_name,
                        "save_adapter_only": False,
                    }
                }
            },
            "trainer": {
                "freeze_modules": "qwen_vl_interface",
                "learning_rate": {"base": 1e-4, "qwen_vl_interface": 2e-4},
            },
        }
    )


class VlmLoraActionHeadTest(unittest.TestCase):
    def test_backward_and_optimizer_include_lora_and_full_action_head(self):
        cfg = lora_config()
        model, settings = apply_lora_to_vlm_backbone(TinyVLA(), cfg)
        cfg.trainer.freeze_modules = _strip_lora_module_from_freeze_modules(
            cfg.trainer.freeze_modules, settings.module_path
        )

        groups = build_param_lr_groups(model, cfg)
        optimized_ids = {id(param) for group in groups for param in group["params"]}
        input_ids = torch.randint(0, 32, (2, 8))
        model(input_ids=input_ids, labels=input_ids).backward()

        lora_params = [(name, param) for name, param in model.named_parameters() if "lora_" in name]
        action_params = list(model.action_model.parameters())
        base_vlm_params = [
            param
            for name, param in model.qwen_vl_interface.model.named_parameters()
            if "lora_" not in name
        ]

        self.assertTrue(lora_params)
        self.assertTrue(all(param.requires_grad and param.grad is not None for _, param in lora_params))
        self.assertTrue(all(param.requires_grad and param.grad is not None for param in action_params))
        self.assertTrue(all(not param.requires_grad and param.grad is None for param in base_vlm_params))
        self.assertTrue(all(id(param) in optimized_ids for _, param in lora_params))
        self.assertTrue(all(id(param) in optimized_ids for param in action_params))
        self.assertTrue(all(id(param) not in optimized_ids for param in base_vlm_params))
        self.assertFalse(settings.save_adapter_only)

        state_dict = model.state_dict()
        self.assertTrue(any("lora_" in key for key in state_dict))
        self.assertIn("action_model.weight", state_dict)

    def test_nested_freeze_pattern_is_preserved(self):
        freeze_modules = "qwen_vl_interface,qwen_vl_interface.model.visual,action_model.encoder"
        stripped = _strip_lora_module_from_freeze_modules(
            freeze_modules, "qwen_vl_interface.model"
        )
        self.assertEqual(stripped, "qwen_vl_interface.model.visual,action_model.encoder")

    def test_custom_adapter_name_and_directory_round_trip(self):
        cfg = lora_config(adapter_name="bridge_task", adapter_dir_name="bridge_adapter")
        model, settings = apply_lora_to_vlm_backbone(TinyVLA(), cfg)
        with tempfile.TemporaryDirectory(dir="/root/data/tmp") as tmp:
            full_checkpoint = Path(tmp) / "steps_5_pytorch_model.pt"
            torch.save(model.state_dict(), full_checkpoint)
            adapter_root = Path(tmp) / "steps_5_bridge_adapter"
            save_vlm_lora_adapter(
                model,
                str(adapter_root),
                adapter_name=settings.adapter_name,
                module_path=settings.module_path,
            )
            resolved = resolve_vlm_lora_adapter_dir(
                full_checkpoint,
                adapter_dir_name=settings.adapter_dir_name,
                adapter_name=settings.adapter_name,
            )
            self.assertEqual(resolved, adapter_root / "bridge_task")
            self.assertTrue((resolved / "adapter_config.json").is_file())

    def test_resume_prefers_checkpoint_adapter_and_detects_custom_directory(self):
        cfg = lora_config(adapter_name="bridge_task", adapter_dir_name="bridge_adapter")
        model, _ = apply_lora_to_vlm_backbone(TinyVLA(), cfg)
        with tempfile.TemporaryDirectory(dir="/root/data/tmp") as tmp:
            checkpoint_dir = Path(tmp)
            adapter_root = checkpoint_dir / "steps_12_bridge_adapter"
            save_vlm_lora_adapter(
                model,
                str(adapter_root),
                adapter_name="bridge_task",
                module_path="qwen_vl_interface.model",
            )

            finder = TrainerUtils()
            finder.accelerator = SimpleNamespace(print=lambda *args, **kwargs: None)
            latest, step = finder._get_latest_checkpoint(str(checkpoint_dir))
            self.assertEqual(Path(latest), adapter_root)
            self.assertEqual(step, 12)

            trainer = VLATrainer.__new__(VLATrainer)
            trainer.config = cfg
            trainer._deferred_lora_checkpoint_load = None
            trainer._deferred_lora_state_dict = None
            self.assertTrue(trainer._defer_lora_checkpoint_load_if_needed(latest, None))
            self.assertIsNone(trainer._deferred_lora_checkpoint_load[0])
            self.assertEqual(
                Path(get_lora_settings(cfg).adapter_path),
                adapter_root / "bridge_task",
            )

    def test_full_checkpoint_with_embedded_lora_is_deferred_without_adapter_dir(self):
        cfg = lora_config()
        model, _ = apply_lora_to_vlm_backbone(TinyVLA(), cfg)
        with tempfile.TemporaryDirectory(dir="/root/data/tmp") as tmp:
            checkpoint = Path(tmp) / "steps_7_pytorch_model.pt"
            torch.save(model.state_dict(), checkpoint)

            trainer = VLATrainer.__new__(VLATrainer)
            trainer.config = cfg
            trainer._deferred_lora_checkpoint_load = None
            trainer._deferred_lora_state_dict = None
            self.assertTrue(trainer._defer_lora_checkpoint_load_if_needed(checkpoint, None))
            self.assertIsNotNone(trainer._deferred_lora_state_dict)

    def test_vlact_transfer_keeps_fresh_action_head(self):
        source = TinyVLA()
        target = TinyVLA()
        with torch.no_grad():
            source.qwen_vl_interface.model.transformer.wte.weight.fill_(0.25)
            source.action_model.weight.fill_(9.0)
            target.action_model.weight.fill_(1.0)

        TrainerUtils.load_pretrained_state_dict(
            target,
            source.state_dict(),
            ignore_modules="action_model",
        )

        self.assertTrue(
            torch.allclose(
                target.qwen_vl_interface.model.transformer.wte.weight,
                source.qwen_vl_interface.model.transformer.wte.weight,
            )
        )
        self.assertTrue(torch.all(target.action_model.weight == 1.0))


if __name__ == "__main__":
    unittest.main()
