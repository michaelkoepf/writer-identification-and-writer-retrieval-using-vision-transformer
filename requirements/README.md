# `requirements`
Some notes on `requirements.txt`:   

| Requirement | Note |
| ------------- |-------------|
| `vit-pytorch` | Used Vision Transformer (ViT) implementation (available: https://github.com/lucidrains/vit-pytorch) |
| `timm` | Provides also an Vision Transformer (ViT) implementation, but is only used for `timm.loss.cross_entropy.LabelSmoothingCrossEntropy` (available: https://github.com/rwightman/pytorch-image-models)  |
| `patool` | Used to extract `.rar` archives of downloaded datasets. Note, that an installation of `rar`, `unrar` or `7z` is required on the target system to extract `.rar` archives. |