"""Independent implementations of the three published/paper neural architectures."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


class NonlocalAttention(nn.Module):
    """P-NN Fig. 6, including unscaled dot product and zero-initialized residual."""

    def __init__(self):
        super().__init__()
        self.q = nn.Conv1d(32, 8, 1, bias=False)
        self.k = nn.Conv1d(32, 8, 1, bias=False)
        self.v = nn.Conv1d(32, 8, 1, bias=False)
        self.out = nn.Conv1d(8, 32, 1, bias=False)
        self.weight = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        z = x.flatten(2)
        # softmax over keys is the transpose of the paper's column-wise A.
        dtype = torch.bfloat16 if z.is_cuda and z.dtype == torch.float32 else z.dtype
        q = self.k(z).transpose(1, 2)[:, None].to(dtype).contiguous()
        k = self.q(z).transpose(1, 2)[:, None].to(dtype).contiguous()
        v = self.v(z).transpose(1, 2)[:, None].to(dtype).contiguous()
        if z.is_cuda:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                a = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        else:
            a = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        return (z + self.weight * self.out(a[:, 0].transpose(1, 2).to(z.dtype))).reshape_as(x)


class PNN(nn.Module):
    def __init__(self, sensors, bins, features=9, **_):
        super().__init__()
        self.features = features
        self.image = nn.Sequential(
            nn.Conv2d(1, 32, 9, padding="same"),
            nn.ReLU(),
            NonlocalAttention(),
            nn.Conv2d(32, 32, 4, padding="same"),
            nn.ReLU(),
        )

        def branch():
            return nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(),
            )

        self.power = branch()
        self.delay = branch()
        self.head = nn.Sequential(
            nn.Linear(32 * sensors * bins + 32 * sensors * features, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 2),
        )

    def forward(self, x):
        image, power, delay = x
        return self.head(
            torch.cat(
                [
                    self.image(image[:, None]).flatten(1),
                    self.power(power[:, None]).flatten(1),
                    self.delay(delay[:, None]).flatten(1),
                ],
                1,
            )
        )


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.weight


class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim=None):
        super().__init__()
        self.heads = heads
        self.d = head_dim or dim // heads
        self.qkv = nn.Linear(dim, 3 * heads * self.d)
        self.out = nn.Linear(heads * self.d, dim)

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, self.d).permute(2, 0, 3, 1, 4)
        return self.out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, -1))


class SwiBlock(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.n1 = RMSNorm(dim)
        self.n2 = RMSNorm(dim)
        self.attn = Attention(dim, 6)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.value = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        z = self.n2(x)
        return x + self.proj(F.silu(self.gate(z)) * self.value(z))


class SST(nn.Module):
    def __init__(self, sensors, bins, size="small", **_):
        super().__init__()
        layers, dim, hidden = {
            "small": (6, 48, 54),
            "medium": (10, 72, 94),
            "large": (16, 96, 231),
        }[size]
        self.embed = nn.Linear(bins, dim)
        self.blocks = nn.Sequential(*[SwiBlock(dim, hidden) for _ in range(layers)])
        self.norm = RMSNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 2))

    def forward(self, x):
        return self.head(self.norm(self.blocks(self.embed(x[0])).mean(1)))


class Cao(nn.Module):
    def __init__(self, sensors, bins, **_):
        super().__init__()
        dim = 256
        self.embed = nn.Linear(bins, dim)
        self.index_w = nn.Parameter(torch.randn(sensors, dim) * 0.02)
        self.index_b = nn.Parameter(torch.zeros(sensors, dim))
        self.register_buffer("indices", torch.arange(1, sensors + 1)[:, None])
        # Literal reported D=d_k=256, H=8: attention has 2048 internal features.
        self.attn = Attention(dim, 8, head_dim=256)
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.head = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 2))

    def forward(self, x):
        z = self.embed(x[0]) + self.indices * self.index_w + self.index_b
        z = self.n1(z + self.attn(z))
        z = self.n2(z + self.ff(z))
        return self.head(z.mean(1))


MODELS = {"pnn": PNN, "sst": SST, "cao": Cao}


class Encoder:
    """Fit all scaling statistics exclusively on the training part of each fold."""

    def __init__(self, method, features=9, power_scaling="per_bin", **_):
        self.method = method
        self.features = features
        self.power_scaling = power_scaling

    def fit(self, power):
        self.scale = max(float(np.mean(power.sum(-1))), 1e-30)
        if self.method == "pnn":
            order = np.argsort(-power, axis=-1, kind="stable")[..., : self.features]
            values = np.take_along_axis(power, order, -1) / self.scale
            self.pmean = float(values.mean())
            self.pstd = max(float(values.std()), 1e-8)
            self.dmean = float(order.mean())
            self.dstd = max(float(order.std()), 1e-8)
        elif self.method == "cao" and self.power_scaling == "global":
            self.scale = max(float(np.sqrt(np.mean(power**2))), 1e-30)
        elif self.method == "cao":
            z = power / self.scale
            self.mean = z.mean(0)
            self.std = np.maximum(z.std(0), 0.01)
        return self

    def transform(self, power):
        p = power / self.scale
        if self.method == "pnn":
            order = np.argsort(-p, axis=-1, kind="stable")[..., : self.features]
            values = (np.take_along_axis(p, order, -1) - self.pmean) / self.pstd
            delays = (order - self.dmean) / self.dstd
            image = np.zeros_like(p)
            np.put_along_axis(image, order, values, -1)
            return tuple(np.asarray(z, dtype=np.float32) for z in (image, values, delays))
        if self.method == "sst":
            p = 100.0 * np.maximum(p.sum(-1, keepdims=True), 1e-30) ** (-0.8) * p
            return (np.sqrt(p).astype(np.float32),)
        if self.power_scaling == "global":
            return (p.astype(np.float32),)
        return (((p - self.mean) / self.std).astype(np.float32),)
