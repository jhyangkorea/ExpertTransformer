__all__ = ['AttentionGlobalContextExpertFusion','FFNGlobalContextExpertFusion']
import torch
import torch.nn as nn
import sys 

class AttentionGlobalContextExpertFusion(nn.Module):

    def __init__(self,k=1):
        super().__init__()
        self.k = k
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.to(self.device)
        
    def top_k_mask(self, routing_scores):
        values, indices = torch.topk(routing_scores, k=self.k, dim=1)
        #print("values at topk ",values,flush=True,file=sys.stderr)
        
        mask = torch.zeros_like(routing_scores, dtype=torch.bool, device = self.device)
        #indices is a tensor of shape [batch_size, k] containing indices to be set to True
        indices = indices.to(self.device)
        values = values.to(self.device)
        mask.scatter_(1, indices, True) #scatter_ is an in-place version of scatter, tensor.scatter_(dim, index, value) dim: the dimension along which to index.
        scale = 1 / (values.sum(dim=1, keepdim=True) + 1e-8)
        return mask, scale

    def forward(self, x, experts, routing_scores, key_padding_mask=None, attn_mask=None):
        """
        expert_outputs: (batch_size, num_experts, output_dim)
        routing_scores: (batch_size, num_experts)
        binary_mask: (batch_size, num_experts)
        """        
        # default binary mask
        binary_mask, scale = self.top_k_mask(routing_scores)
        routing_scores = routing_scores.to(self.device)                        
        masked_scores = scale * routing_scores * binary_mask  # (batch_size, num_experts)
        masked_scores = masked_scores#.to(x.device)
            
        
        # Efficient hard routing: only compute active experts
        batch_size, seq_len , output_dim = x.size()
        num_experts = len(experts)
        final_output = torch.zeros(batch_size, seq_len, output_dim, device = self.device)# device=x.device)
        
        for i in range(num_experts):
            active_indices = (binary_mask[:, i] == 1).nonzero(as_tuple=True)[0]

            if active_indices.numel() > 0:
                expert_output = experts[i](x,key_padding_mask=key_padding_mask,attn_mask=attn_mask)
                weight = masked_scores[active_indices, i].view(-1, 1, 1)  # (active_batch, 1, 1)
                final_output[active_indices] += weight * expert_output[active_indices]
        
        return final_output, binary_mask.sum(dim=0)



class FFNGlobalContextExpertFusion(nn.Module):


    def __init__(self,k=1):
        super().__init__()
        self.k = k
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def top_k_mask(self, routing_scores):
        values, indices = torch.topk(routing_scores, k=self.k, dim=-1)
        indices = indices.to(self.device)
        values = values.to(self.device)        
        mask = torch.zeros_like(routing_scores, dtype=torch.bool, device = self.device)
        #indices is a tensor of shape [batch_size, k] containing indices to be set to True
        #PyTorch automatically aligns the batch (b) and sequence (t) dimensions from indices with mask, and uses the values in indices[b, t, :] to set mask[b, t, indices[b, t, k]] = True.
        mask.scatter_(-1, indices, True) #scatter_ is an in-place version of scatter, tensor.scatter_(dim, index, value) dim: the dimension along which to index.
        scale = 1 / (values.sum(dim=-1, keepdim=True) + 1e-8)
        return mask, scale

    def forward(self, x, experts, routing_scores):
        """
        expert_outputs: (batch_size, num_experts, output_dim)
        routing_scores: (batch_size, num_experts)
        binary_mask: (batch_size, num_experts)
        """
        # default binary mask
        binary_mask, scale = self.top_k_mask(routing_scores)
        masked_scores = scale * routing_scores * binary_mask  # (batch_size, seq_len, num_experts)

        # Efficient hard routing: only compute active experts
        batch_size, seq_len, output_dim = x.size()
        num_experts = len(experts)
        final_output = torch.zeros(batch_size, seq_len, output_dim, device = self.device)#device=x.device)
        for i in range(num_experts):
            active_batch, active_token = (binary_mask[:, :, i] == 1).nonzero(as_tuple=True)
            if active_batch.numel() > 0:
                expert_output = experts[i](x)
                weight = masked_scores[active_batch, active_token, i].view(-1, 1)  # (active_batch, active_tokens, 1)
                final_output[active_batch,active_token] += weight * expert_output[active_batch,active_token]
        
        return final_output, binary_mask.sum(dim=0).sum(dim=0)





if __name__ == "__main__":
    B, S, F, num_expert = 2, 10, 8, 3
    x = torch.arange(B*S*F, dtype=torch.float32).reshape(B, S, F)
    x=torch.randn([2,10,8])    
    cls_ar_out = torch.randn([B,num_expert])
    ffnr_out = torch.randn([B,S,num_expert])

    g_fusion = AttentionGlobalContextExpertFusion(k=2)
    experts = [nn.modules.Linear(x.size(2),x.size(2)) for i in range(3)]
    g_fusion_out, expert_freq = g_fusion(x,experts,cls_ar_out)
    print("g_fusion_out ",g_fusion_out.size())
    print(g_fusion_out)
    print("expert_freq: ",expert_freq)

    print("Look at cls_ar_out and cls_ar_out They are diffferent shape since one for sequence level and the other for token level")
    print("cls_ar_out ",cls_ar_out.size())
    print("ffnr_out ",ffnr_out.size())


    g_fusion = FFNGlobalContextExpertFusion(k=2)
    experts = [nn.modules.Linear(x.size(2),x.size(2)) for i in range(3)]
    g_fusion_out, expert_freq = g_fusion(x,experts,ffnr_out)
    print("ffn_g_fusion_out ",g_fusion_out.size())
    print(g_fusion_out)
    print("expert_freq: ",expert_freq)
