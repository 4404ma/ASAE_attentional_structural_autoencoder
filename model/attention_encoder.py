import torch
from torch import nn
from utils import ImageToSequence


class AttentionEncoder(nn.Module):
    def __init__(self, batch_size=1, image_size=64, patch_size=4, in_channels=3, stride=2, attention_mode='multi_head', num_heads=12, mlp_ratio=4., proj_ratio=2., embed_dim=768, **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention_mode = attention_mode
        self.image_to_sequence = ImageToSequence(batch_size=batch_size, image_size=image_size, patch_size=patch_size, in_channels=in_channels, stride=stride)
        output_dim = embed_dim if attention_mode == 'multi_head' else embed_dim * 2

        # transfer images to sequences
        width_num = height_num = (image_size - patch_size) // stride + 1
        sequence_len = width_num * height_num
        patch_len = int(in_channels * patch_size**2 + 2)
        hidden_dim = int(mlp_ratio * patch_len)
        multi_proj_dim = int(proj_ratio * embed_dim)

        # Pointwise MLP (get patch_embedding)
        self.pointwise_mlp = nn.Sequential(
            nn.Linear(patch_len, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

        # multi-head attention
        self.ruv = nn.Linear(embed_dim, embed_dim * 3)
        self.multi_proj = [nn.Sequential(
            nn.Linear(sequence_len * embed_dim // num_heads, multi_proj_dim),
            nn.Linear(multi_proj_dim, 1)
        )] * num_heads

        # single attention
        self.proj = nn.Linear(sequence_len * embed_dim, 12)

    def forward(self, x):
        x = self.image_to_sequence(x) # [B, C, H, W] -> [B, N, patch_len]
        B, N, D = x.shape
        x = x.reshape(-1, x.shape[2]).type(torch.FloatTensor)
        x = self.pointwise_mlp(x) # [B * N, output_dim]
        if self.attention_mode == 'multi_head':
            ruv = self.ruv(x).reshape(B, N, 3, self.num_heads, self.embed_dim // self.num_heads).permute(2, 3, 0, 1, 4) # [3, 12, B, N, 768/12]
            r, u, v = ruv.unbind(0)

            attn = (r @ u.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            x = (attn @ v).reshape(self.num_heads, B, -1)

            q = torch.FloatTensor(self.batch_size, 12)
            for k in range(self.num_heads):
                q[:, k] = self.multi_proj[k](x[k]).reshape(-1)
        elif self.attention_mode == 'single':
            u, v = x[:, :self.embed_dim], x[:, self.embed_dim:]
            attn = (u @ u.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            x = (attn @ v).reshape(B, -1)
            q = self.proj(x)

        return q