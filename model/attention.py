import torch
import torch.nn as nn

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return attn

class X_align(nn.Module):
    def __init__(self, dim, aligned_dim=512):
        super().__init__()
        self.align = nn.Linear(dim, aligned_dim)

    def forward(self, x):
        x = self.align(x)

        return x

def cross_modal_attn(vattn, aattn, mode):
    def _cal_attn_a2b_mode0(attna, attnb):
        attna = attna[:, :, None, :]
        attnb = attnb[:, :, None, :].transpose(-2, -1)
        cm_attn = attna * attnb
        return cm_attn

    if mode == 'mean':
        cm_attn = _cal_attn_a2b_mode0(vattn, aattn)
        cvattn = (cm_attn).mean(dim=-2) / vattn.mean(dim=-1, keepdim=True)
        caattn = (cm_attn).mean(dim=-1) / aattn.mean(dim=-1, keepdim=True)
    elif mode == 'softmax':
        cm_attn = _cal_attn_a2b_mode0(vattn, aattn)
        cvattn = (cm_attn).mean(dim=-2).softmax(dim=-1)
        caattn = (cm_attn).mean(dim=-1).softmax(dim=-1)

    else:
        raise NotImplementedError()

    return cvattn, caattn

def align_feats(feat, attn):
    attn = attn.mean(dim=1).unsqueeze(dim=-1)
    eng_org = ((feat.mean(dim=1)) ** 2).sum(dim=1, keepdim=True)  # taking the patch mean; then calculate energy; [B, 1]
    eng_aft = (((feat * attn).mean(dim=1)) ** 2).sum(dim=1, keepdim=True)  # multiply the patch attn; patch mean; calculate energy; # [B, 1]
    scalar = (eng_org / eng_aft).sqrt().unsqueeze(dim=-1)
    new_feat = scalar * attn * feat

    return new_feat

def cal_cross_att(t_embed, s_embed, t_att, s_att):
    weights_t = t_att(t_embed).mean(dim=-2)
    weights_s = s_att(s_embed).mean(dim=-2)
    cvattn, caattn = cross_modal_attn(vattn=weights_t, aattn=weights_s, mode='mean')

    t_fea = align_feats(t_embed, cvattn)
    s_fea = align_feats(s_embed, caattn)

    return t_fea, s_fea
