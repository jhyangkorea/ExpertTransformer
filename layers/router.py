__all__ = ['RouterCompression','MeanRouter','CLSAttentionRouter','FFNRouter']
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class RouterCompression:
  def __init__(self, feat_dim, local_segment_length, local_stride, local_start, dist_interval, dist_start):
    self.feat_dim = feat_dim
    self.local_segment_length = local_segment_length
    self.local_stride = local_stride
    self.local_stride = local_stride
    self.local_start = local_start
    self.dist_interval = dist_interval
    self.dist_start = dist_start
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.even_positions = None
    self.even_indices = None
    self.odd_positions = None
    self.odd_indices = None
    self.comp_even_num_features = None
    self.comp_odd_num_features = None

  def build_even_indices(self, seq_len, feat_dim, local_segment_length, local_stride, local_start):
      """
      Returns:
        even_positions: (num_even,)
        even_indices:   (num_even, K_even) where K_even = E * local_segment_length
      """
      even_positions = torch.arange(0, seq_len, 2, device=self.device)  # (num_even,)
      E = max((feat_dim - local_segment_length) // local_stride, 0)
      if E == 0:
          # no local segments to take; return empty indices per even position
          return even_positions, torch.empty(len(even_positions), 0, dtype=torch.long, device=self.device)

      k = torch.arange(E, device=self.device)[:, None]                                # (E,1)
      m = torch.arange(local_segment_length, device=self.device)[None, :]             # (1,Ls)
      base = (local_start + even_positions[:, None, None] + k*local_stride + m)  # (num_even, E, Ls)
      even_indices = (base % feat_dim).reshape(len(even_positions), -1)          # (num_even, K_even)
      return even_positions, even_indices


  def build_odd_indices(self, seq_len, feat_dim, dist_interval, dist_start):
      """
      Returns:
        odd_positions: (num_odd,)
        odd_indices:   (num_odd, K_odd) where K_odd = O
      """
      odd_positions = torch.arange(1, seq_len, 2, device=self.device)                 # (num_odd,)
      if dist_interval <= 0:
          raise ValueError("dist_interval must be > 0")

      O = (feat_dim-1) // dist_interval + 1
      if O <= 0:
          return odd_positions, torch.empty(len(odd_positions), 0, dtype=torch.long, device=self.device)

      j = torch.arange(O, device=self.device)[None, :]                                # (1,O)
      base = dist_start + odd_positions[:, None] + j*dist_interval               # (num_odd, O)
      odd_indices = (base % feat_dim).to(torch.long)                             # (num_odd, O)
      return odd_positions, odd_indices


  # ---------------------------
  # Feature selectors (data gather)
  # ---------------------------

  def even_step_features(self, x, local_segment_length=2, local_stride=3, local_start=0, even_positions = None, even_indices = None):
      """
      x: (B, S, F)
      Returns:
        feats_even: (B, num_even, K_even)
        even_positions: (num_even,)
      """
      B, S, F = x.shape      
      if even_positions is None and even_indices is None:
          self.even_positions, self.even_indices = self.build_even_indices(S, F, local_segment_length, local_stride, local_start)
      if self.even_indices.numel() == 0:
          return x[:, :0, :0], self.even_positions

      type(even_indices)
      self.comp_even_num_features = self.even_indices.size(1)
      x_even = x[:, self.even_positions, :]                                           # (B, num_even, F)
      idx = self.even_indices.unsqueeze(0).expand(B, -1, -1)                          # (B, num_even, K_even)
      feats_even = torch.gather(x_even, 2, idx)                                   # (B, num_even, K_even)
      return feats_even, self.even_positions


  def odd_step_features(self, x, dist_interval=4, dist_start=0, odd_positions=None, odd_indices=None):
      """
      x: (B, S, F)
      Returns:
        feats_odd: (B, num_odd, K_odd)
        odd_positions: (num_odd,)
      """
      B, S, F = x.shape
      if odd_positions is None and odd_indices is None:
          self.odd_positions, self.odd_indices = self.build_odd_indices(S, F, dist_interval, dist_start)
      if self.odd_indices.numel() == 0:
          return x[:, :0, :0], self.odd_positions

      self.comp_odd_num_features = self.odd_indices.size(1)
      x_odd = x[:, self.odd_positions, :]                                             # (B, num_odd, F)
      idx = self.odd_indices.unsqueeze(0).expand(B, -1, -1)                           # (B, num_odd, K_odd)
      feats_odd = torch.gather(x_odd, 2, idx)                                     # (B, num_odd, K_odd)
      return feats_odd, self.odd_positions

#sequence average pooling
class MeanRouter(nn.Module):
    def __init__(self, feature_dim, num_expert,router_dropout=0.1):
        super().__init__()  #call the superclass constructor
        self.num_expert = num_expert
        self.mlp = nn.Linear(feature_dim, num_expert)
        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.dropout = nn.Dropout(router_dropout)
        self.logits = None

    def forward(self, xe, xo):
        pooled_xe = xe.mean(dim=1)  # shape: (batch_size, feature_dim)
        pooled_xo = xo.mean(dim=1)  # shape: (batch_size, feature_dim)
        pooled = torch.cat([pooled_xe, pooled_xo], dim=1)
        
        self.logits = self.dropout(self.mlp(pooled))
        return nn.functional.softmax(self.logits, dim=-1)  # shape: (batch_size, num_expert) #routing score

#feature average pooling
class MeanRouter(nn.Module):
    def __init__(self, num_patch, embed_dim, num_expert,router_dropout=0.1):
        super().__init__()  #call the superclass constructor
        self.num_expert = num_expert
        self.num_patch = num_patch
        self.embed_dim = embed_dim 
        self.mlp = nn.Linear(num_patch*embed_dim, num_expert)
        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.dropout = nn.Dropout(router_dropout)
        self.logits = None
        
    def forward(self, x):
        
        pooled = x.reshape([x.size(0),-1])        
        self.logits = self.dropout(self.mlp(pooled))
        return nn.functional.softmax(self.logits, dim=-1)  # shape: (batch_size, num_expert) #routing score






        
class CLSAttentionRouter(nn.Module):
    def __init__(self, feature_dim_even, feature_dim_odd, num_expert, num_heads=2,router_dropout=0.1):
        super().__init__()
        self.attn_temporal = nn.MultiheadAttention(embed_dim=feature_dim_even, num_heads=num_heads, batch_first=True)
        self.attn_contextual = nn.MultiheadAttention(embed_dim=feature_dim_odd, num_heads=num_heads, batch_first=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cls_token_temporal = nn.Parameter(torch.randn(1, 1, feature_dim_even, device = self.device))
        self.cls_token_contextual = nn.Parameter(torch.randn(1, 1, feature_dim_odd, device = self.device))
        self.mlp = nn.Linear(feature_dim_even + feature_dim_odd, num_expert) #3 experts
        self.dropout = nn.Dropout(router_dropout)
        self.logits = None
        self.to(self.device)

    def forward(self, xe, xo):
        batch_size = xe.size(0)
        
        # Expand CLS tokens for batch
        cls_t = self.cls_token_temporal.expand(batch_size, -1, -1)  # (B, 1, D)
        cls_c = self.cls_token_contextual.expand(batch_size, -1, -1)

        # Concatenate CLS tokens to input
        x_t = torch.cat([cls_t, xe], dim=1)  # (B, 1+T, D)
        x_c = torch.cat([cls_c, xo], dim=1)

        # Self-attention
        attn_out_t, _ = self.attn_temporal(x_t, x_t, x_t)
        attn_out_c, _ = self.attn_contextual(x_c, x_c, x_c)

        # Extract CLS outputs
        cls_temporal = attn_out_t[:, 0, :]  # (B, D)
        cls_contextual = attn_out_c[:, 0, :]  # (B, D)

        # Concatenate CLS tokens
        cls_concat = torch.cat([cls_temporal, cls_contextual], dim=-1)  # shape: (batch_size, feature_dim_even, feature_dim_odd)
        self.logits = self.dropout(self.mlp(cls_concat))
        # Two-layer MLP
        return nn.functional.softmax(self.logits, dim=-1) #routing score



class FFNRouter(nn.Module):
    def __init__(self, feature_dim, num_expert,router_dropout=0.1):
        super().__init__()  #call the superclass constructor
        self.num_expert = num_expert
        self.mlp = nn.Linear(feature_dim, num_expert)
        self.dropout = nn.Dropout(router_dropout)
        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.logits = None
    def forward(self, x):
        self.logits = self.dropout(self.mlp(x))
        return nn.functional.softmax(self.logits, dim=-1)  # shape: (batch_size, num_expert)


class RouterLossEMA:
    def __init__(self, num_experts, decay_constant=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ema_usage = torch.zeros(num_experts, device=self.device)
        self.decay_constant = decay_constant
        self.threshold = None
        self.cent_loss = nn.CrossEntropyLoss()        

    def update(self, routing_probs):
        # routing_probs: [batch_size, num_experts]
        batch_size = routing_probs.size(0)
        avg_usage = routing_probs.mean(dim=0)  # [num_experts]
        gamma = batch_size / (batch_size + self.decay_constant)
        self.ema_usage = (1 - gamma) * self.ema_usage + gamma * avg_usage
        self.ema_usage = self.ema_usage/self.ema_usage.sum() 

    def entropy_loss(self):
        usage = self.ema_usage + 1e-8  # avoid log(0)
        if len(usage.shape) == 1:
            return (usage * usage.log()).sum()
        else:
            return (usage * usage.log()).sum(axis=1).mean()

    def z_loss(self, routing_probs):
        # Convert probabilities to logits (up to a constant)
        log_p = torch.log(routing_probs + 1e-9)
        z = log_p - log_p.mean(dim=-1, keepdim=True)  # center logits    
        # Compute log-sum-exp
        lse = torch.logsumexp(z, dim=-1)
        return lse.pow(2).mean()

    def ce_loss(self,x,base_model,pred,y,logits):
        with torch.no_grad():  # Compute errors using current expert0
            pred0 = base_model(x)
            errors = ((pred0 - y) ** 2).mean()  # Per sample MSE
        if self.threshold == None:
            self.threshold = errors.median()  # Dynamic threshold
        else:
            self.threshold = 0.99*self.threshold + 0.01*errors.median()
        targets = (errors > self.threshold).long()  # 0 for low error (expert0), 1 for high (expert1)
        routing_loss = self.cent_loss(logits, targets)
        return routing_loss
        



if __name__ == "__main__":
    B, S, F = 2, 10, 8
    x = torch.arange(B*S*F, dtype=torch.float32).reshape(B, S, F)

    local_segment_length=2
    local_stride=3
    local_start=0
    dist_interval=4
    dist_start=0
    rc = RouterCompression(F, local_segment_length, local_stride, local_start, dist_interval, dist_start)
    # Build separately
    feats_even, pos_even = rc.even_step_features(x, local_segment_length=2, local_stride=3, local_start=0)
    feats_odd,  pos_odd  = rc.odd_step_features(x, dist_interval=4, dist_start=0)

    print(feats_even)
    print(feats_odd)

    comm_feats_dim = feats_even.size(2)+feats_odd.size(2)
    num_experts = 3
    mr = MeanRouter(comm_feats_dim, num_experts)
    mr_out = mr(feats_even,feats_odd)
    print(mr_out)
    ffnr = FFNRouter(x.size(2),num_experts)
    ffnr_out = ffnr(x) #routing score
    print(ffnr_out)    

    cls_ar = CLSAttentionRouter(feats_even.size(2), feats_odd.size(2), num_experts, num_heads=2)
    cls_ar_out = cls_ar(feats_even, feats_odd)
    print(cls_ar_out)

    print(rc.odd_indices.size(),rc.even_indices.size(),rc.comp_even_num_features,rc.comp_odd_num_features)


    num_experts = 4
    routing_probs = torch.arange(64,dtype=float).view(16,num_experts)
    router_loss_tracker = RouterLossEMA(num_experts)
    router_loss_tracker.update(routing_probs)
    router_loss = router_loss_tracker.entropy_loss()
    print(router_loss)
