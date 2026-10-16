"""CPU structural tests use a tiny mocked VLM and the real StarVLA Action DiT."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image
from omegaconf import OmegaConf
from starVLA.rgbd.preprocessing import metric_depth, normalize_actions, normalize_state, validate_contract
from starVLA.rgbd.depth import DepthCNN,LayerwiseDepthFusion


def test_dataset_aligns_previous_proprio_with_current_visual_action(tmp_path):
    from starVLA.rgbd.dataset import Libero3D
    task = tmp_path / "task"
    episode = task / "episode_000"
    episode.mkdir(parents=True)
    metadata = {
        "state_semantics": ["eef_pos_x_y_z", "eef_axis_angle_x_y_z", "gripper_qpos_2"],
        "task_name": "TEST_SCENE_move_object",
        "episodes": [{"episode_dir": "episode_000", "npz_file": "data.npz", "num_steps": 4}],
    }
    (task / "metadata.json").write_text(json.dumps(metadata))
    state = np.arange(32, dtype=np.float32).reshape(4, 8)
    actions = np.arange(28, dtype=np.float32).reshape(4, 7) / 100
    images = np.stack([np.full((2, 2, 3), i, np.uint8) for i in range(4)])
    np.savez(episode / "data.npz", state=state, actions=actions,
             agentview_rgb=images, wrist_rgb=images)
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"tasks": [{"relative_dir": "task", "smoke_episodes": [0]}]}))
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"state": {"min": [0] * 8, "max": [100] * 8}}))
    dataset = Libero3D(tmp_path, selection, stats, horizon=2)
    assert [t for _, t in dataset.index] == [1, 2]
    item = dataset[0]
    np.testing.assert_array_equal(item["action"], actions[1:3])
    np.testing.assert_array_equal(np.asarray(item["image"][0]), images[1])
    np.testing.assert_allclose(item["state"], 2 * state[0:1] / 100 - 1)

def test_depth_units_and_invalid_mask():
    d,m=metric_depth(np.array([[0,1000,np.nan,np.inf,-1]],np.float32),encoding="millimeters")
    np.testing.assert_array_equal(m,[[False,True,False,False,False]])
    assert d[0,1]==1 and np.isfinite(d).all()
    d,m=metric_depth(np.array([0,.5,1]),encoding="mujoco_zbuffer",near=.1,far=10)
    np.testing.assert_allclose(d[:2],[.1,.1980198],rtol=1e-5)
    assert not m[2]
    with pytest.raises(ValueError):metric_depth(np.ones(2),encoding=None)

def test_semantic_guards():
    with pytest.raises(ValueError):validate_contract({"depth_units_verified":True})
    with pytest.raises(ValueError):normalize_actions(np.zeros((2,8)))
    x=np.array([[0,0,0,0,0,0,.04,-.04]],np.float32)
    out=normalize_state(x,{"min":[-1]*8,"max":[1]*8})
    assert out.shape==(1,8) and out[0,6]>0 and out[0,7]<0

def test_cnn_depth_sensitivity_and_all_invalid_mask():
    torch.manual_seed(2)
    encoder=DepthCNN();fusion=LayerwiseDepthFusion(128,32,2)
    d=torch.rand(2,2,32,32);m=torch.ones_like(d,dtype=torch.bool)
    tokens=encoder(d,m)
    assert tokens.shape==(2,64,128)
    assert not torch.allclose(tokens,encoder(d.flip(0),m))
    vl=[torch.randn(2,5,32) for _ in range(2)]
    layers,mask=fusion(vl,tokens,None,torch.tensor([[True,False],[False,False]]))
    assert mask.shape==(2,69) and not mask[1,5:].any()
    loss=sum(x.square().mean() for x in layers);loss.backward()
    assert encoder.cnn[0].weight.grad.abs().sum()>0
    assert fusion.gates.grad.abs().sum()>0

def make_model(monkeypatch,depth=True):
    import starVLA.model.framework.VLM4A.QwenPI_v3 as base
    from starVLA.model.framework.VLM4A.QwenPI_v3_RGBD import QwenPI_v3_RGBD
    class FakeVLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model=nn.Linear(1,1)
            self.model.config=SimpleNamespace(hidden_size=32,num_hidden_layers=2)
    monkeypatch.setattr(base,"get_vlm_model",lambda config:FakeVLM())
    def encode(self,images,instructions):
        h=torch.ones(len(images),3,32)
        return self._project_vl_hidden_for_action([h,h*2]),torch.ones(len(images),3,dtype=torch.bool)
    monkeypatch.setattr(QwenPI_v3_RGBD,"_encode_vl_hidden_states",encode)
    monkeypatch.setattr(base.Qwen_PI_v3,"_encode_vl_hidden_states",encode)
    cfg=OmegaConf.create({"framework":{"name":"QwenPI_v3_RGBD" if depth else "QwenPI_v3",
        "rgbd":{"encoder":"cnn"},"action_model":{"state_dim":8,"action_horizon":4,"num_inference_timesteps":2,
        "num_target_vision_tokens":2,"diffusion_model_cfg":{"action_dit_hidden_dim":32,"attention_head_dim":8,"dropout":0.,"final_dropout":False}}},
        "trainer":{"repeated_diffusion_steps":1},"datasets":{"vla_data":{"obs_image_size":[32,32]}}})
    return (QwenPI_v3_RGBD(cfg) if depth else base.Qwen_PI_v3(cfg))

@pytest.mark.parametrize("depth",[False,True])
def test_real_action_head_backward_update_predict_restore(monkeypatch,depth,tmp_path):
    torch.manual_seed(7);model=make_model(monkeypatch,depth)
    e={"image":[Image.new("RGB",(32,32))]*2,"lang":"move bowl",
       "state":np.zeros((1,8),np.float32),"action":np.zeros((4,7),np.float32),
       "depth":np.ones((2,32,32),np.float32),"depth_mask":np.ones((2,32,32),bool)}
    before=model.action_model.action_decoder.layer2.weight.detach().clone()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    model.train();loss=model(examples=[e])["action_loss"];assert torch.isfinite(loss)
    loss.backward()
    if depth:
        assert model.depth_fusion.projector[1].weight.grad.abs().sum()>0
        assert model.depth_encoder.cnn[0].weight.grad.abs().sum()>0
        assert all(p.grad is None for p in model.qwen_vl_interface.parameters())
    opt.step()
    assert not torch.equal(before,model.action_model.action_decoder.layer2.weight)
    model.eval();torch.manual_seed(123)
    pred=model.predict_action(examples=[e])["normalized_actions"]
    assert pred.shape==(1,4,7) and np.isfinite(pred).all()
    path=tmp_path/'test.pt';torch.save(model.state_dict(),path)
    clone=make_model(monkeypatch,depth);clone.load_state_dict(torch.load(path,weights_only=True));clone.eval()
    torch.manual_seed(123)
    np.testing.assert_array_equal(pred,clone.predict_action(examples=[e])["normalized_actions"])

def test_real_lingbot_encoder_no_decoder_and_trainable_fusion(monkeypatch):
    from pathlib import Path
    from starVLA.rgbd.depth import FrozenLingBot
    checkpoint=Path('/root/data/models/lingbot-depth-v0.5/model.pt')
    if not checkpoint.exists():pytest.skip("Official weights not downloaded")
    model=make_model(monkeypatch,True)
    model.depth_encoder=FrozenLingBot(str(checkpoint),num_tokens=64)
    model.depth_fusion=LayerwiseDepthFusion(model.depth_encoder.output_dim,32,2)
    def forbid(*args,**kwargs):raise AssertionError("Depth decoder must never run")
    monkeypatch.setattr(model.depth_encoder.model.neck,'forward',forbid)
    if hasattr(model.depth_encoder.model,'depth_head'):
        monkeypatch.setattr(model.depth_encoder.model.depth_head,'forward',forbid)
    e={"image":[Image.new("RGB",(32,32),color=(100,120,140))]*2,"lang":"synthetic plane",
       "state":np.zeros((1,8),np.float32),"action":np.zeros((4,7),np.float32),
       "depth":np.full((2,32,32),1.2,np.float32),"depth_mask":np.ones((2,32,32),bool),
       "intrinsics":np.repeat(np.array([[32,0,16],[0,32,16],[0,0,1]],np.float32)[None],2,0)}
    rgb=torch.full((1,2,3,32,32),.5)
    d=torch.full((1,2,32,32),1.2);mask=torch.ones_like(d,dtype=torch.bool)
    K=torch.from_numpy(e["intrinsics"])[None]
    f1=model.depth_encoder(d,mask,rgb,K)
    f2=model.depth_encoder(d*1.5,mask,rgb,K)
    assert not torch.allclose(f1,f2), "Pretrained features must respond to numeric depth"
    model.train()
    before=model.depth_fusion.projector[1].weight.detach().clone()
    opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
    loss=model(examples=[e])["action_loss"];loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in model.depth_encoder.parameters())
    assert model.depth_fusion.projector[1].weight.grad.abs().sum()>0
    assert model.depth_fusion.gates.grad.abs().sum()>0
    opt.step();assert not torch.equal(before,model.depth_fusion.projector[1].weight)
    model.eval()
    pred=model.predict_action(examples=[e])["normalized_actions"]
    assert pred.shape==(1,4,7) and np.isfinite(pred).all()

def test_transport_preserves_metric_depth_and_mask():
    from deployment.model_server.tools import msgpack_numpy
    x={"depth":np.array([[[.12,1.23456]]],np.float32),"depth_mask":np.array([[[True,False]]])}
    y=msgpack_numpy.unpackb(msgpack_numpy.Packer().pack(x))
    assert y["depth"].dtype==np.float32 and y["depth_mask"].dtype==bool
    np.testing.assert_array_equal(x["depth"],y["depth"])

def test_pixel_transform_updates_intrinsics():
    from starVLA.rgbd.simulation import transform_camera
    rgb=np.zeros((4,6,3),np.uint8);d=np.arange(24,dtype=np.float32).reshape(4,6)
    K=np.array([[4,0,2],[0,4,1],[0,0,1]],np.float32)
    _,turned,K2=transform_camera(rgb,d,K,"rotate_180")
    np.testing.assert_array_equal(turned,d[::-1,::-1])
    p=np.array([.3,.2,1]);uv=K@p;uv2=K2@p
    np.testing.assert_allclose(uv2[:2],[5-uv[0],3-uv[1]])


def test_production_compact_checkpoint_optimizer_and_identity(monkeypatch,tmp_path):
    from starVLA.rgbd.checkpoint import save_checkpoint,restore_checkpoint
    model=make_model(monkeypatch,True)
    cfg=OmegaConf.create({"rgbd_experiment":{}})
    for name in ["selected_tasks","normalization","contract"]:
        path=tmp_path/(name+".json");path.write_text("{}")
        cfg.rgbd_experiment[name]=str(path)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    target=model.depth_fusion.gates
    target.sum().backward();optimizer.step()
    before=target.detach().clone()
    gen=torch.Generator().manual_seed(17)
    path=tmp_path/"compact.pt"
    save_checkpoint(path,model,optimizer,9,cfg,gen)
    with torch.no_grad():target.add_(1)
    step=restore_checkpoint(path,model,cfg,optimizer,gen)
    assert step==9 and torch.equal(before,target)
    ck=torch.load(path,weights_only=False)
    assert not any(k.startswith("qwen_vl_interface.") for k in ck["learned"])
    from pathlib import Path
    Path(cfg.rgbd_experiment.contract).write_text('{"changed": true}')
    with pytest.raises(ValueError,match="identity"):
        restore_checkpoint(path,model,cfg)
