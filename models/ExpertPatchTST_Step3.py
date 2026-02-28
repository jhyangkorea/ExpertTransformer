__all__ = ['ExpertPatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import ExpertPatchTST_backbone
from layers.PatchTST_layers import series_decomp
import sys

class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto', padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type='flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size

        num_attention_experts = 2#configs.num_attention_experts
        num_ffn_experts = 2#configs.num_ffn_experts
        num_active_experts = 2#configs.num_active_experts        
        print("num_attention_experts ",num_attention_experts,flush=True,file=sys.stderr)
        
        self.expert_loss_weight = 0
        # model(s)
        self.decomposition = decomposition
        self.ffn_router_logits = None
        self.attention_router_logits = None
        self.z_loss_weight = 1        
        self.z_loss = None
        
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = ExpertPatchTST_backbone(
                c_in=c_in, context_window=context_window, target_window=target_window,
                patch_len=patch_len, stride=stride, max_seq_len=max_seq_len,
                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v,
                d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                store_attn=store_attn, pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                head_dropout=head_dropout, padding_patch=padding_patch,
                pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                revin=revin, affine=affine, subtract_last=subtract_last,
                verbose=verbose, num_attention_experts = num_attention_experts, num_ffn_experts=num_ffn_experts, num_active_experts=num_active_experts, **kwargs
            )
            self.expert_loss_weight = self.model_trend.expert_loss_weight            
            self.model_res = ExpertPatchTST_backbone(
                c_in=c_in, context_window=context_window, target_window=target_window,
                patch_len=patch_len, stride=stride, max_seq_len=max_seq_len,
                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v,
                d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                store_attn=store_attn, pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                head_dropout=head_dropout, padding_patch=padding_patch,
                pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                revin=revin, affine=affine, subtract_last=subtract_last,
                verbose=verbose, num_attention_experts = num_attention_experts, num_ffn_experts=num_ffn_experts, num_active_experts=num_active_experts,**kwargs
            )
        else:
            self.model = ExpertPatchTST_backbone(
                c_in=c_in, context_window=context_window, target_window=target_window,
                patch_len=patch_len, stride=stride, max_seq_len=max_seq_len,
                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v,
                d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                store_attn=store_attn, pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                head_dropout=head_dropout, padding_patch=padding_patch,
                pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                revin=revin, affine=affine, subtract_last=subtract_last,
                verbose=verbose, num_attention_experts = num_attention_experts, num_ffn_experts=num_ffn_experts, num_active_experts=num_active_experts,**kwargs
            )
            self.expert_loss_weight = self.model.expert_loss_weight       
    def forward(self, x):  # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)
            
            res, res_aux = self.model_res(res_init)         # [B,C,T], scalar
            trend, trend_aux = self.model_trend(trend_init) # [B,C,T], scalar
            
            
            x = res + trend
            aux_loss = res_aux + trend_aux
            #aux_loss = torch.zeros_like(aux_loss)
            x = x.permute(0,2,1)   # [B, T, C]
        else:
            x = x.permute(0,2,1)   # [B, C, T]
            x, aux_loss = self.model(x)
            self.ffn_router_logits = self.model.ffn_router_logits
            self.attention_router_logits = self.model.attention_router_logits
            #aux_loss = torch.zeros_like(aux_loss)
            x = x.permute(0,2,1)   # [B, T, C]
        
        return x, self.model.z_loss