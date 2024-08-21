
import torch
from torch import nn , einsum
from einops import rearrange, repeat
from blocks import MBConv, Transformer, PreNorm


class CrossAttention(nn.Module):
    """
    Args:
        dim (int): Number of input dimensions.
        heads (int): Number of heads. Default: 6
        dim_head (int): Dimension for each head in the cross-attention layer
        dropout (float): Dropout rate. Default: 0.0
    """
    def __init__(self, dim, heads = 6, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(dim, inner_dim , bias = False)
        self.to_v = nn.Linear(dim, inner_dim , bias = False)
        self.to_q = nn.Linear(dim, inner_dim , bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_qkv):
        b, n, _, h = *x_qkv.shape, self.heads

        k = self.to_k(x_qkv)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)

        v = self.to_v(x_qkv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        q = self.to_q(x_qkv[:, 0].unsqueeze(1))
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)


        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class MFCA(nn.Module):
    """
    Args:
        small_embedding_dim (int): Number of linear projection output dimension for the small patches.
        small_depth (int): Number of transformer blocks for the small branch. Default: 3
        small_no_heads (int): Number of attention heads used for the small branch layers. Default: 6
        small_head_dim (int): Dimension for each head for the small branch. Default: 16
        small_mlp_expand (int): Ratio of mlp hidden dim for the small branch. Default: 12
        large_embedding_dim (int): Number of linear projection output dimension for the large patches.
        large_depth (int): Number of transformer blocks for the large branch. Default: 3
        large_no_heads (int): Number of attention heads used for the large branch layers. Default: 6
        large_head_dim (int): Dimension for each head for the large branch. Default: 64
        large_mlp_expand (int): Ratio of mlp hidden dim for the large branch. Default: 4
        cross_attn_depth (int): Number of cross-attention layers. Default: 3
        cross_attn_heads (int): Number of attention heads used within the cross-attention layers. Default: 6
    """
    def __init__(self,small_embedding_dim, small_depth, small_no_heads,small_head_dim,small_mlp_expand,
                 large_embedding_dim,large_depth,large_no_heads,large_head_dim,large_mlp_expand,
                     cross_attn_depth=4,cross_attn_heads=6):
        super().__init__()
        self.transformer_encoder_small = Transformer(small_embedding_dim, small_depth, small_no_heads, small_head_dim, small_mlp_expand)
        self.transformer_encoder_large = Transformer(large_embedding_dim, large_depth, large_no_heads, large_head_dim, large_mlp_expand)


        self.cross_attention_layers=nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attention_layers.append(nn.ModuleList([
                nn.Linear(small_embedding_dim, large_embedding_dim),
                nn.Linear(large_embedding_dim, small_embedding_dim),
                PreNorm(large_embedding_dim, CrossAttention(large_embedding_dim, heads = cross_attn_heads, dim_head = large_head_dim)),
                nn.Linear(large_embedding_dim, small_embedding_dim),
                nn.Linear(small_embedding_dim, large_embedding_dim),
                PreNorm(small_embedding_dim, CrossAttention(small_embedding_dim, heads = cross_attn_heads, dim_head = small_head_dim)),

            ]))


    def forward(self, FV1,FV2):

        xs = self.transformer_encoder_small(FV1)
        xl = self.transformer_encoder_large(FV2)

        for f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l in self.cross_attention_layers:

            small_class = xs[:, 0]
            x_small = xs[:, 1:]
            large_class = xl[:, 0]
            x_large = xl[:, 1:]

            # cross attention for the intermediate feature maps
            cal_q = f_ls(large_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_small), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = g_sl(cal_out)
            xl = torch.cat((cal_out, x_large), dim=1)

            # cross attention for the last feature maps
            cal_q = f_sl(small_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_large), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_ls(cal_out)
            xs = torch.cat((cal_out, x_small), dim=1)

        return xs, xl
