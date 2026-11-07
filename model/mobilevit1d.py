"""
1D-MobileViT Architecture — EXP006
Edge-Deployable ECG Arrhythmia Classification
Submitted to IEEE Journal of Biomedical and Health Informatics (JBHI)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, groups=1, act=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, stride=stride,
                      padding=k // 2, groups=groups, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU6(inplace=True) if act else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class InvertedResidual1d(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand=6):
        super().__init__()
        mid = in_ch * expand
        self.use_skip = (stride == 1 and in_ch == out_ch)
        self.block = nn.Sequential(
            Conv1dBnRelu(in_ch, mid, k=1),
            Conv1dBnRelu(mid, mid, k=3, stride=stride, groups=mid),
            nn.Conv1d(mid, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
        )

    def forward(self, x):
        return x + self.block(x) if self.use_skip else self.block(x)


class TransformerBlock1d(nn.Module):
    def __init__(self, dim, heads, ff_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        x2 = self.norm1(x)
        x = x + self.attn(x2, x2, x2, need_weights=False)[0]
        x = x + self.ff(self.norm2(x))
        return x


class MobileViT1dBlock(nn.Module):
    def __init__(self, in_ch, dim, heads, ff_dim, patch_size=2):
        super().__init__()
        self.patch = patch_size
        self.local_conv = Conv1dBnRelu(in_ch, in_ch, k=3)
        self.proj_in = nn.Conv1d(in_ch, dim, 1, bias=False)
        self.transformer = TransformerBlock1d(dim, heads, ff_dim)
        self.proj_out = nn.Conv1d(dim, in_ch, 1, bias=False)
        self.fuse = Conv1dBnRelu(in_ch * 2, in_ch, k=1, act=True)

    def forward(self, x):
        B, C, L = x.shape
        P = self.patch
        pad = (P - L % P) % P
        if pad:
            x = F.pad(x, (0, pad))
        L_pad = x.shape[2]
        n_patches = L_pad // P

        local = self.local_conv(x)
        h = self.proj_in(local)
        h = h.reshape(B, -1, n_patches, P).permute(
            0, 2, 3, 1).reshape(B * n_patches, P, -1)
        h = self.transformer(h)
        dim = h.shape[-1]
        h = h.reshape(B, n_patches, P, dim).permute(
            0, 3, 1, 2).reshape(B, dim, L_pad)
        h = self.proj_out(h)

        if pad:
            h = h[:, :, :L]
            local = local[:, :, :L]
            x = x[:, :, :L]

        out = self.fuse(torch.cat([local, h], dim=1))
        return out


class MobileViT1d(nn.Module):
    """
    1D-MobileViT backbone for ECG arrhythmia classification.

    Input:  (B, 1, 260) — single-lead ECG, 260 samples @ 360 Hz
    Output: logits (B, num_classes), projection (B, proj_dim)

    Parameters: 403,700
    MACs:        6.6M
    """

    def __init__(self, num_classes=4, proj_dim=128):
        super().__init__()
        self.stem = Conv1dBnRelu(1, 16, k=3, stride=2)
        self.local1 = InvertedResidual1d(16, 32, stride=2, expand=6)
        self.local2 = InvertedResidual1d(32, 64, stride=2, expand=6)
        self.mvit1 = MobileViT1dBlock(
            64, dim=64, heads=4, ff_dim=128, patch_size=2)
        self.local3 = InvertedResidual1d(64, 128, stride=2, expand=4)
        self.mvit2 = MobileViT1dBlock(
            128, dim=128, heads=4, ff_dim=192, patch_size=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.proj_head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, proj_dim)
        )
        self.cls_head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.local1(x)
        x = self.local2(x)
        x = self.mvit1(x)
        x = self.local3(x)
        x = self.mvit2(x)
        feat = self.gap(x).squeeze(-1)
        logits = self.cls_head(feat)
        proj = F.normalize(self.proj_head(feat), dim=1)
        return logits, proj


class MobileViT1dInference(nn.Module):
    """
    Inference wrapper — drops projection head, returns logits only.
    Used for ONNX export and Android deployment.
    """

    def __init__(self, backbone: MobileViT1d):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        logits, _ = self.backbone(x)
        return logits


if __name__ == "__main__":
    model = MobileViT1d(num_classes=4, proj_dim=128)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    dummy = torch.randn(1, 1, 260)
    logits, proj = model(dummy)
    print(f"Logits shape:    {logits.shape}")
    print(f"Projection shape: {proj.shape}")
