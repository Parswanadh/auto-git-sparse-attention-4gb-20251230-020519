import torch
import torch.nn as nn
import math

class SparseAttention(nn.Module):
    """Block-diagonal sparse attention for 4GB VRAM"""
    def __init__(self, d_model=512, n_heads=8, block_size=256):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.block_size = block_size
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        attn = torch.zeros(B, self.n_heads, N, N, device=x.device)
        n_blocks = (N + self.block_size - 1) // self.block_size
        for b in range(n_blocks):
            start = b * self.block_size
            end = min((b + 1) * self.block_size, N)
            attn[:, :, start:end, start:end] = 1.0

        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v.transpose(2, 3)).transpose(2, 3)
        return self.out(out.reshape(B, N, C))

class SparseTransformer(nn.Module):
    def __init__(self, vocab_size=50000, d_model=512, n_heads=8, n_layers=6, block_size=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([SparseAttention(d_model, n_heads, block_size) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = x + layer(x)
        return self.head(self.norm(x))

if __name__ == "__main__":
    model = SparseTransformer()
    print(f"Parameters: {sum(p.numel() for p in model())/1e6:.1f}M")
