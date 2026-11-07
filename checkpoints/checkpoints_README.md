# Checkpoints

## EXP006 — Final Frozen Model

| Property | Value |
|---|---|
| File | `EXP006_v1_FROZEN_best_macro_f1.pt` |
| SHA-256 | `18acb7f8...` |
| DS2 MacroF1 | 0.402 |
| INCART MacroF1 | 0.502 |
| Parameters | 403,700 |

The checkpoint file exceeds GitHub's 100 MB file size limit and is therefore
not stored in this repository.

**Download:** Available upon request to the corresponding author.
The checkpoint will be made publicly available via a persistent link
upon paper acceptance.

## Loading the checkpoint

```python
import torch
from model.mobilevit1d import MobileViT1d

ckpt = torch.load("EXP006_v1_FROZEN_best_macro_f1.pt", map_location="cpu")
model = MobileViT1d(num_classes=4, proj_dim=128)
model.load_state_dict(ckpt["model_state"])
model.eval()
```

## Checkpoint keys

```python
ckpt.keys()
# dict_keys(['epoch', 'model_state', 'optimizer_state', 'val_macro_f1', 'config'])
```
