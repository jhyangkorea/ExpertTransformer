__all__=['ExpertAttentionMultiHead','ExpertFFN','ExpertTransformerBlock']
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.router import RouterCompression, MeanRouter, CLSAttentionRouter, FFNRouter, RouterLossEMA
from layers.expert_fusion import AttentionGlobalContextExpertFusion, FFNGlobalContextExpertFusion
import sys
from torch import Tensor
from typing import Callable, Optional





class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights















class ExpertAttentionMultiHead(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        self.to(self.device)
        
    def forward(self, x, key_padding_mask=None, attn_mask=None):
        # x: [batch_size, seq_len, embed_dim]
        # MHA expects (query, key, value) all shaped [batch_size, seq_len, embed_dim]
        attn_output, _ = self.mha(x, x, x)  # Self-attention
        return self.out_proj(attn_output)

class ExpertFFN(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, x):
        return self.ffn(x)


class ExpertAttentionMultiHead(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1,bias=True):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        #self.mha = _MultiheadAttention(embed_dim, num_heads)       
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        attn_output, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)  # self-attention
        out = self.out_proj(attn_output)
        out = self.dropout(out)
        return out


class ExpertFFN(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.act = nn.GELU()  # standard transformer uses GELU
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# simple helper: transpose for BatchNorm option
class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


# === EXACT PatchTST-style Transpose helper ===
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous: bool = False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        x = x.transpose(*self.dims)
        return x.contiguous() if self.contiguous else x




class ExpertTransformerBlock(nn.Module):

    def __init__(self, embed_dim, num_attention_experts, num_ffn_experts, local_segment_length, local_stride, local_start, dist_interval, dist_start, num_heads = 4, num_active_experts = 2, num_patch = 42, dropout=0.3):
        super().__init__()
        #use this if multihead
        self.num_attention_experts = num_attention_experts
        self.num_ffn_experts = num_ffn_experts
        self.attention = ExpertAttentionMultiHead(embed_dim, num_heads, dropout=dropout)
        self.ffn = ExpertFFN(embed_dim,dropout=dropout)
        self.attn_experts = nn.ModuleList([ExpertAttentionMultiHead(embed_dim, num_heads, dropout=dropout) for _ in range(num_attention_experts)])
        self.ffn_experts = nn.ModuleList([ExpertFFN(embed_dim,dropout=dropout) for _ in range(num_ffn_experts)])
        self.router_compression = RouterCompression(embed_dim, local_segment_length, local_stride, local_start, dist_interval, dist_start)
        self.num_patch = num_patch
        self.mean_router = MeanRouter(num_patch,embed_dim,self.num_attention_experts)  
        self.cls_attention_router = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ffn_router = FFNRouter(embed_dim,num_ffn_experts).to(self.device)
        self.attention_fusion = AttentionGlobalContextExpertFusion(k=num_active_experts).to(self.device)
        self.ffn_fusion = FFNGlobalContextExpertFusion(k=num_active_experts).to(self.device)
        self.attention_expert_freq = torch.zeros(num_attention_experts, device=self.device)
        self.ffn_expert_freq = torch.zeros(num_ffn_experts, device=self.device)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm1 = nn.Sequential(
            Transpose(1, 2), nn.BatchNorm1d(embed_dim), Transpose(1, 2)
        )
        self.norm2 = nn.Sequential(
            Transpose(1, 2), nn.BatchNorm1d(embed_dim), Transpose(1, 2)
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn_router_logits = None
        self.attention_router_logits = None
        self.ffn_router_selection = []
        self.attention_router_selection = []
    
    def forward(self, x, key_padding_mask=None, attn_mask=None):
        feats_even, pos_even = self.router_compression.even_step_features(x, local_segment_length=self.router_compression.local_segment_length, local_stride=self.router_compression.local_stride, local_start=self.router_compression.local_start )
        feats_odd,  pos_odd  = self.router_compression.odd_step_features(x, dist_interval=self.router_compression.dist_interval, dist_start=self.router_compression.dist_start)
        
        attention_router_out = self.mean_router(x)
        expert_attn_out, attention_expert_freq = self.attention_fusion(x, self.attn_experts, attention_router_out, key_padding_mask, attn_mask)
        self.attention_router_selection.append(torch.argmax(attention_router_out, dim=1).detach().cpu())
        
        x = self.norm1(x + self.dropout(expert_attn_out))

        #FFN Experts
        ffn_router_out = self.ffn_router(x)
        expert_ffn_out, ffn_expert_freq = self.ffn_fusion(x, self.ffn_experts, ffn_router_out)
        self.ffn_router_selection.append(torch.argmax(ffn_router_out, dim=2).detach().cpu())

        x = self.norm2(x + self.dropout(expert_ffn_out))

        self.ffn_router_logits = self.ffn_router.logits
        self.attention_router_logits = self.mean_router.logits

        self.attention_expert_freq += attention_expert_freq.detach()
        self.ffn_expert_freq += ffn_expert_freq.detach()

        return x, torch.mean(attention_router_out,dim=0), torch.mean(ffn_router_out,dim=0)












if __name__ == "__main__":
    x=torch.arange(64,dtype=torch.float32).view(2,4,8)
    batch_size, seq_len, embed_dim = x.size()
    num_attention_experts = 4
    num_ffn_experts = 4
    local_segment_length = 2
    local_stride = 3
    local_start = 0
    dist_interval = 4
    dist_start = 0
    expert_transformer_block = ExpertTransformerBlock(embed_dim, num_attention_experts, num_ffn_experts, local_segment_length, local_stride, local_start, dist_interval, dist_start, num_heads = 4, num_active_experts = 2)
    y=expert_transformer_block(x)
    print("x ", x)
    print("y ", y)
    print("attention exper freq", expert_transformer_block.attention_expert_freq)
    print("ffn exper freq", expert_transformer_block.ffn_expert_freq)