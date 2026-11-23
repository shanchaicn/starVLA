"""Shape/key-only inspection of legacy transfer candidates; never loads a policy."""
import json
import struct
from pathlib import Path
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


def audit_legacy_checkpoint(checkpoint_path, vlm_directory, new_action_head):
    expected = {}
    for file in sorted(Path(vlm_directory).glob('*.safetensors')):
        with file.open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            header = json.loads(stream.read(size))
        expected.update({key: item['shape'] for key, item in header.items() if key != '__metadata__'})
    if not expected:
        raise ValueError('No local backbone headers found')
    with FakeTensorMode():
        old = torch.load(checkpoint_path, map_location='cpu', weights_only=True, mmap=True)
    if not isinstance(old, dict) or not all(isinstance(x, torch.Tensor) for x in old.values()):
        raise ValueError('Expected audited flat tensor state dictionary')
    prefix = 'qwen_vl_interface.model.'
    backbone = {key[len(prefix):]: list(tensor.shape) for key, tensor in old.items() if key.startswith(prefix)}
    missing = sorted(set(expected)-set(backbone))
    wrong = {key: [expected[key], backbone[key]] for key in set(expected)&set(backbone) if expected[key] != backbone[key]}
    extra = sorted(set(backbone)-set(expected))
    if missing or wrong or extra != ['lm_head.weight']:
        raise ValueError(f'Backbone identity/shape contract differs: {missing}, {wrong}, {extra}')
    config = json.loads((Path(vlm_directory)/'config.json').read_text())
    if not config.get('tie_word_embeddings') or backbone['lm_head.weight'] != expected['model.language_model.embed_tokens.weight']:
        raise ValueError('Extra lm_head must be explicitly accounted for by tied embeddings')
    current = new_action_head.state_dict()
    incompatible = {key: {'legacy': list(old['action_model.'+key].shape), 'new': list(value.shape)}
                    for key, value in current.items() if old['action_model.'+key].shape != value.shape}
    if set(incompatible) != {'model.fc2.weight', 'model.fc2.bias'} or new_action_head.action_dim != 8:
        raise ValueError('Unexpected head shape mismatch; refuse implicit adaptation')
    return {'decision': 'PARTIAL_BACKBONE_NEW_8D_ACTION_HEAD', 'checkpoint': str(checkpoint_path),
            'backbone_header_keys': len(expected), 'backbone_common_shapes_match': True,
            'extra_tied_lm_head': extra, 'head_incompatible_shapes': incompatible,
            'all_legacy_action_head_keys_excluded': [key for key in old if key.startswith('action_model.')],
            'full_checkpoint_load_allowed': False, 'head_tensors_loaded_from_legacy': 0,
            'preferred_initialization': 'original local Qwen3-VL-4B-Instruct plus fully fresh action head',
            'scope': 'metadata only; no payload numerical equivalence, tied-value equality or VLM forward verified'}
