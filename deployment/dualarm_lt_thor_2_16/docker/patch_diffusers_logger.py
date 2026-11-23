#!/usr/bin/env python3
"""Fix the vendored diffusers TorchAO optional-import warning during image build."""

from pathlib import Path

path = Path("/app/server/runtime/eval_deps/diffusers/quantizers/torchao/torchao_quantizer.py")
source = path.read_text()
definition = "logger = logging.get_logger(__name__)\n"
anchor = (
    "if (\n"
    "    is_torch_available()\n"
    '    and is_torch_version(">=", "2.6.0")\n'
)
if source.count(definition) != 1 or source.count(anchor) != 1:
    raise SystemExit("Unexpected diffusers source; refusing to patch")
if source.index(definition) > source.index(anchor):
    source = source.replace(definition, "", 1).replace(anchor, definition + "\n" + anchor, 1)
    path.write_text(source)
print("diffusers torchao logger initialized before optional import")
