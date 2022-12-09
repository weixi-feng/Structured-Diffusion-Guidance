from inspect import isfunction
import math
import pdb
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from structured_stable_diffusion.modules.diffusionmodules.util import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., struct_attn=False, save_map=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.struct_attn = struct_attn
        self.save_map = save_map

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)

        if isinstance(context, list):
            if self.struct_attn:
                out = self.struct_qkv(q, context, mask)
            else:
                context = torch.cat([context[0], context[1]['k'][0]], dim=0) # use key tensor for context
                out = self.normal_qkv(q, context, mask)
        else:
            context = default(context, x)
            out = self.normal_qkv(q, context, mask)

        return self.to_out(out)
    
    def struct_qkv(self, q, context, mask):
        """
        context: list of [uc, list of conditional context]
        """
        uc_context = context[0]
        context_k, context_v = context[1]['k'], context[1]['v']

        if isinstance(context_k, list) and isinstance(context_v, list):
            out = self.multi_qkv(q, uc_context, context_k, context_v, mask)
        else:
            raise NotImplementedError
        return out

    def multi_qkv(self, q, uc_context, context_k, context_v, mask):
        h = self.heads

        assert uc_context.size(0) == context_k[0].size(0) == context_v[0].size(0)
        true_bs = uc_context.size(0) * h

        k_uc, v_uc = self.get_kv(uc_context)
        k_c = [self.to_k(c_k) for c_k in context_k]
        v_c = [self.to_v(c_v) for c_v in context_v]
        
        q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)

        k_uc = rearrange(k_uc, 'b n (h d) -> (b h) n d', h=h)            
        v_uc = rearrange(v_uc, 'b n (h d) -> (b h) n d', h=h)

        k_c  = [rearrange(k, 'b n (h d) -> (b h) n d', h=h) for k in k_c] # NOTE: modification point
        v_c  = [rearrange(v, 'b n (h d) -> (b h) n d', h=h) for v in v_c]

        # get composition
        sim_uc = einsum('b i d, b j d -> b i j', q[:true_bs], k_uc) * self.scale
        sim_c  = [einsum('b i d, b j d -> b i j', q[true_bs:], k) * self.scale for k in k_c]

        attn_uc = sim_uc.softmax(dim=-1)
        attn_c  = [sim.softmax(dim=-1) for sim in sim_c]

        if self.save_map and sim_uc.size(1) != sim_uc.size(2):
            self.save_attn_maps(attn_c)

        # get uc output
        out_uc = einsum('b i j, b j d -> b i d', attn_uc, v_uc)
        
        # get c output        
        n_keys, n_values = len(k_c), len(v_c)
        if n_keys == n_values:
            out_c = sum([einsum('b i j, b j d -> b i d', attn, v) for attn, v in zip(attn_c, v_c)]) / len(v_c)
        else:
            assert n_keys == 1 or n_values == 1
            out_c = sum([einsum('b i j, b j d -> b i d', attn, v) for attn in attn_c for v in v_c]) / (n_keys * n_values)

        out = torch.cat([out_uc, out_c], dim=0)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)  

        return out

    def normal_qkv(self, q, context, mask):
        h = self.heads
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        if self.save_map and sim.size(1) != sim.size(2):
            self.save_attn_maps(attn.chunk(2)[1])

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        
        return out

    def get_kv(self, context):
        return self.to_k(context), self.to_v(context)

    def save_attn_maps(self, attn):
        h = self.heads
        if isinstance(attn, list):
            height = width = int(math.sqrt(attn[0].size(1)))
            self.attn_maps = [rearrange(m.detach(), '(b x) (h w) l -> b x h w l', x=h, h=height, w=width)[...,:20].cpu() for m in attn]
        else:
            height = width = int(math.sqrt(attn.size(1)))
            self.attn_maps = rearrange(attn.detach(), '(b x) (h w) l -> b x h w l', x=h, h=height, w=width)[...,:20].cpu()


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True, struct_attn=False, save_map=False):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout,
                                    struct_attn=struct_attn, save_map=save_map)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, struct_attn=False, save_map=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, struct_attn=struct_attn, save_map=save_map)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in