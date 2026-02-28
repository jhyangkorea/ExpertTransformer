#whenever you add a new model
#1. add models at import
#2. add models at model_dict


from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, ExpertPatchTST, ExpertPatchTST_Step1, ExpertPatchTST_Step2, ExpertPatchTST_Step3, ExpertPatchTST_Step4, ExpertPatchTST_Step5, ExpertPatchTST_Step6, ExpertPatchTST_Step7
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric,Maskmetric
from layers.router import RouterLossEMA

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch.nn.init as init
import copy
import pickle

from datetime import datetime
time_string = datetime.now().strftime("%Y%m%d_%H%M%S")


warnings.filterwarnings('ignore')

load_model_path = '/home/jhyang00/expert_transformer/Expert_PatchTST/PatchTST_supervised_optimized/'

#This function is specifically designed for copying the model with 2 epxerts from a model with 1 except except the 2 nd expert
def model_copy(model1,model2):

    print("model1:", model1,flush=True,file=sys.stderr)    
    print("model2:", model2,flush=True,file=sys.stderr)    

    #model2.model.revin_layer = model1.model.revin_layer
    model2.model.padding_patch_layer =  model1.model.padding_patch_layer
    num_layer = len(model2.model.backbone.encoder.layers)
    for layer_idx in range(3):  # 3 layers
        # Copy attn_experts[0]
        model2.model.backbone.encoder.layers[layer_idx].attn_experts[0] = model1.model.backbone.encoder.layers[layer_idx].attn_experts[0]    
        # Copy ffn_experts[0]
        model2.model.backbone.encoder.layers[layer_idx].ffn_experts[0] =  model1.model.backbone.encoder.layers[layer_idx].ffn_experts[0]
        model2.model.backbone.encoder.layers[layer_idx].attn_experts[1] = model1.model.backbone.encoder.layers[layer_idx].attn_experts[0]    
        # Copy ffn_experts[0]
        model2.model.backbone.encoder.layers[layer_idx].ffn_experts[1] =  model1.model.backbone.encoder.layers[layer_idx].ffn_experts[0]        
        model2.model.backbone.encoder.layers[layer_idx].norm1 = model1.model.backbone.encoder.layers[layer_idx].norm1
        model2.model.backbone.encoder.layers[layer_idx].norm2 = model1.model.backbone.encoder.layers[layer_idx].norm2
    # Other shared parts (W_P, head, etc.) can be copied fully if desired:
    model2.model.backbone.W_P = model1.model.backbone.W_P
    model2.model.head = model1.model.head
    model2.model.revin_layer = copy.deepcopy(model1.model.revin_layer)
    return model2
  
# Selector using first transformer layer output
class ExpertNumberSelector(nn.Module):
        
    def __init__(self, args):
        super().__init__()
        #self.single_model = single_model
        #self.expert_model = expert_model           
        self.linear = nn.Linear(args.seq_len*args.enc_in, args.seq_len*args.enc_in//4)  # Predicts 1 or 2 experts
        self.linear2 = nn.Linear(args.seq_len*args.enc_in//4, 1)  # Predicts 1 or 2 experts
        init.xavier_uniform_(self.linear.weight)  # Uniform distribution        
        self.batch_size = args.batch_size
        self.dropout = nn.Dropout(0.2)

    def forward(self, batch_x):
        # x : [batch, seq_len, n_var]
        #batch_x = batch_x.permute(0,2,1)
        #print("Linear layer weights:", self.linear.weight,flush=True,file=sys.stderr)
        # Print biases
        #print("Linear layer biases:", self.linear.bias,flush=True,file=sys.stderr)     
        batch_x = batch_x.reshape([self.batch_size,-1])
        x = self.linear(batch_x) 
        logit = self.linear2(nn.functional.relu(self.dropout(x)))
        
        #logit = self.linear(batch_x)  # (batch, 1)
        prob_k2 = torch.sigmoid(logit)  # Probability of k=2
        k = (prob_k2 > 0.5).float() + 1  # (batch, 1), values 1 or 2
        #print("dimk: ",[k.size(),logit.size()],flush=True,file=sys.stderr)
        return k, logit #[batch_size,1]


# Define hook to capture first layer output
def hook(module, input, output):
    global first_layer_output
    first_layer_output = output

def create_subbatches(batch_input, batch_output, k):
    """
    Split batch into sub-batches based on number of experts (k=1 or k=2).
    
    Args:
        batch_input: Tensor of shape (seq_len, batch, embed_dim) or (batch, ...)
        k: Tensor of shape (batch, 1), values 1 or 2 indicating number of experts
    
    Returns:
        subbatch_1: Tensor of samples where k=1
        subbatch_2: Tensor of samples where k=2
        idx_1: Indices of samples where k=1
        idx_2: Indices of samples where k=2
    """
    # Ensure k is 1D integer tensor
    k = k.squeeze(-1).long()  # (batch,)
    
    # Get indices for k=1 and k=2
    idx_1 = torch.where(k == 1)[0]  # Indices where k=1
    idx_2 = torch.where(k == 2)[0]  # Indices where k=2
    
    # Split batch_input based on indices
    # Handle different input shapes
    if batch_input.dim() == 3:  # (batch, seq_len, embed_dim)
        subbatch_1 = batch_input[idx_1, :, :] if idx_1.numel() > 0 else None
        subbatch_2 = batch_input[idx_2, :, :] if idx_2.numel() > 0 else None
        ysubbatch_1 = batch_output[idx_1, :, :] if idx_1.numel() > 0 else None
        ysubbatch_2 = batch_output[idx_2, :, :] if idx_2.numel() > 0 else None
    elif batch_input.dim() == 2:  # (batch, features)
        subbatch_1 = batch_input[idx_1, :] if idx_1.numel() > 0 else None
        subbatch_2 = batch_input[idx_2, :] if idx_2.numel() > 0 else None
        ysubbatch_1 = batch_output[idx_1, :] if idx_1.numel() > 0 else None
        ysubbatch_2 = batch_output[idx_2, :] if idx_2.numel() > 0 else None
    else:
        raise ValueError("Unsupported batch_input shape")
    
    return subbatch_1, subbatch_2, ysubbatch_1, ysubbatch_2, idx_1, idx_2


# Load state_dict with filtering
def load_filtered_state_dict(model, state_dict_path):
    # Load saved state_dict
    saved_state_dict = torch.load(state_dict_path)
    
    # Get model's expected state_dict keys
    model_state_dict = model.state_dict()
    
    # Filter only matching keys
    filtered_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_state_dict}
    
    # Check for missing or unexpected keys
    missing = [k for k in model_state_dict if k not in saved_state_dict]
    unexpected = [k for k in saved_state_dict if k not in model_state_dict]
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Ignored unexpected keys: {unexpected}")
    
    # Load filtered state_dict
    model.load_state_dict(filtered_state_dict, strict=False)
    return model
    



# Load state_dict with filtering
def freeze_all_parameters(model):

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.requires_grad = False    
    #print(model,flush=True,file=sys.stderr)
    return model
    
def check_all_parameters_frozen(model):
    """
    Check if all parameters in the model have requires_grad=False.
    
    Args:
        model (nn.Module): The PyTorch model to check.
    
    Returns:
        bool: True if all parameters are non-trainable, False if any parameter is trainable.
    """
    all_frozen = True
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter {name}: requires_grad=True (trainable)", flush=True, file=sys.stderr)
            all_frozen = False
        else:
            param.requires_grad = False
            print(f"Parameter {name}: requires_grad=False (non-trainable)", flush=True, file=sys.stderr)
    if all_frozen:
        print("All parameters are non-trainable.", flush=True, file=sys.stderr)
    else:
        print("Some parameters are trainable.", flush=True, file=sys.stderr)
    return all_frozen


class MaskedMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(self, pred, target, mask=None):
        # Compute element-wise MSE
        mse = self.mse(pred, target, reduction='none')
        if mask is not None:
            # Mask out (e.g., where mask=0 or NaN; adjust logic as needed)
            mse = mse[mask]  # Or mse = mse * mask if mask is float weights
        else:
            mask = target != 0
            mse = mse[mask]  # Or mse = mse * mask if mask is float weights
        return mse.mean() if len(mse.shape) > 0 else mse



class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.expert_loss = None
        self.base_model = None
        #self.expert_num_selector = None
        self.threshold = None       
        self.bce_loss = nn.CrossEntropyLoss()
        self.expert_num_selector = ExpertNumberSelector(self.args).to(self.device)
        #self.expert_loss_weight = 0
        self.hook_handle = None 
        self.num_batch = None
        self.selector_loss = None
        #self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.3])).to(self.device)  # For selector logit
        self.bce_loss = nn.BCEWithLogitsLoss().to(self.device)  # For selector logit
        self.mse_loss = nn.MSELoss(reduction='none').to(self.device)            
        self.percentile_t = args.quantile
        self.percentile_avg_bool = args.percentile_avg_bool
        self.tem_var1 = None
        self.tem_var2 = None

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'ExpertPatchTST': ExpertPatchTST,
            'ExpertPatchTST_Step1': ExpertPatchTST_Step1, #for one expert training
            'ExpertPatchTST_Step2': ExpertPatchTST_Step2, #(2,2,1) router training with fixing experts 
            'ExpertPatchTST_Step3': ExpertPatchTST_Step3, #(2,2,1) experts training with fixed router   
            'ExpertPatchTST_Step4': ExpertPatchTST_Step4, #Step9 is required after Step3 and before Step4, #(2,2,2) training router with fixed experts
            'ExpertPatchTST_Step5': ExpertPatchTST_Step5, #(2,2,2) training all with selector    
            'ExpertPatchTST_Step6': ExpertPatchTST_Step6,                              
            'ExpertPatchTST_Step7': ExpertPatchTST_Step7,                                          
        }
        model = model_dict[self.args.model].Model(self.args).float()
            
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    """
            train   Fix
    Step1	base model		(1,1,1)
    Step2	Router	Expert	(2,2,1)
    Step3	Expert	Router	(2,2,1)
    Step4	Router	Expert	(2,2,2)
    Step5	Expert	Router	(2,2,2)
    
    """

    
    def gen_base_mode(self):
        if 'Step2' in self.args.model: 
            router_loss_obj = RouterLossEMA(2) #2 refers to 2 experts
            single_args = self.args
            single_args.num_attention_experts = 1
            single_args.num_ffn_experts = 1
            single_args.num_active_experts = 1
            self.base_model = ExpertPatchTST_Step1.Model(single_args).float()
            directory_name = f"ExpertPatchTST_Step1_{self.args.data}_{self.args.pred_len}_{self.args.random_seed}/"
            filename = directory_name+"checkpoint.pth"
            
            self.base_model = load_filtered_state_dict(self.base_model,load_model_path+filename)
            
            self.model = model_copy(self.base_model,self.model) #copy the base model to experot 0 in the model

            self.base_model = self.base_model.to(self.device)
            self.model = self.model.to(self.device)
        elif 'Step3' in self.args.model: #fine tuning with different routing loss
            router_loss_obj = RouterLossEMA(2) #2 refers to 2 experts
            directory_name = f"ExpertPatchTST_Step2_{self.args.data}_{self.args.pred_len}_{self.args.random_seed}/"
            step2_filename = directory_name+"checkpoint.pth"            
            self.model = load_filtered_state_dict(self.model,load_model_path+step2_filename)
            
            self.model = self.model.to(self.device)      
        elif 'Step4' in self.args.model: #fine tuning with different routing loss
            router_loss_obj = RouterLossEMA(2) #2 refers to 2 experts
            step3_filename = f"Step3_model_{self.args.data}_{self.args.pred_len}.pth"
            directory_name = f"ExpertPatchTST_Step3_{self.args.data}_{self.args.pred_len}_{self.args.random_seed}/"
            step3_filename = directory_name+"checkpoint.pth"                 
            self.model = load_filtered_state_dict(self.model,load_model_path+step3_filename)
            
            self.model = self.model.to(self.device)    
        elif 'Step5' in self.args.model: #fine tuning with different routing loss
            router_loss_obj = RouterLossEMA(2) #2 refers to 2 experts
            step4_filename = f"Step4_model_{self.args.data}_{self.args.pred_len}.pth"
            directory_name = f"ExpertPatchTST_Step4_{self.args.data}_{self.args.pred_len}_{self.args.random_seed}/"
            step4_filename = directory_name+"checkpoint.pth"                 
            self.model = load_filtered_state_dict(self.model,load_model_path+step4_filename)
            
            self.model = self.model.to(self.device)            
        elif 'Step6' in self.args.model: #fine tuning with different routing loss
            router_loss_obj = RouterLossEMA(2) #2 refers to 2 experts


            single_args = self.args
            single_args.num_attention_experts = 1
            single_args.num_ffn_experts = 1
            single_args.num_active_experts = 1
            self.base_model = ExpertPatchTST_Step1.Model(single_args).float()
            filename = f"base_model_{self.args.data}_{self.args.pred_len}.pth"
            directory_name = f"ExpertPatchTST_Step1_{self.args.data}_{self.args.pred_len}_{self.args.random_seed}/"
            filename = directory_name+"checkpoint.pth"                   
            self.base_model = load_filtered_state_dict(self.base_model,load_model_path+filename)
            self.base_model = self.base_model.to(self.device)

            step5_filename = f"Step5_model_{self.args.data}_{self.args.pred_len}.pth"
            directory_name = f"ExpertPatchTST_Step5_{self.args.data}_{self.args.pred_len}_{self.args.random_seed}/"
            step5_filename = directory_name+"checkpoint.pth"             
            self.model = load_filtered_state_dict(self.model,load_model_path+step5_filename)            
            self.model = self.model.to(self.device)               
            
            self.hook_handle = self.base_model.model.backbone.encoder.layers[0].register_forward_hook(self._hook_fn)
            
        
        elif 'Step7' in self.args.model: #fine tuning with different routing loss
            router_loss_obj = RouterLossEMA(2) #2 refers to 2 experts


            single_args = self.args
            single_args.num_attention_experts = 1
            single_args.num_ffn_experts = 1
            single_args.num_active_experts = 1
            self.base_model = ExpertPatchTST_Step1.Model(single_args).float()
            filename = f"base_model_{self.args.data}_{self.args.pred_len}.pth"
            directory_name = f"ExpertPatchTST_Step1_{self.args.data}_{self.args.pred_len}_{self.args.random_seed}/"
            filename = directory_name+"checkpoint.pth"            
            self.base_model = load_filtered_state_dict(self.base_model,load_model_path+filename)
            self.base_model = self.base_model.to(self.device)

            directory_name = f"ExpertPatchTST_Step5_{self.args.data}_{self.args.pred_len}_{self.args.random_seed}/" #It loads the model step 5 since step 6 optimizees selector only
            step6_filename = directory_name+"checkpoint.pth"            
            self.model = load_filtered_state_dict(self.model,load_model_path+step6_filename)            
            self.model = self.model.to(self.device)               
            self.expert_num_selector = load_filtered_state_dict(self.expert_num_selector,load_model_path+directory_name+'checkpoint.pth')
            
            self.hook_handle = self.base_model.model.backbone.encoder.layers[0].register_forward_hook(self._hook_fn)

    

    # Define hook to capture first layer output
    def _hook_fn(self, module, input, output):
        self.first_layer_output = output[0]  # Extract tensor, discard attn_weights

    def _select_criterion(self):
        return nn.MSELoss()
        

    def _get_data(self, flag):
        print(f"Hook captured output shape: {self.args}", flush=True, file=sys.stderr)

        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim


    #start from here
    def _selector_optimizer(self):
        opt_selector = optim.Adam(
            list(self.expert_num_selector.parameters()),
            lr=self.args.learning_rate
        )        
        return opt_selector
    def _single_optimizer(self):
        opt_selector = optim.Adam(
            list(self.base_model.parameters()),lr=self.args.learning_rate
        )
        return opt_selector
    def _double_optimizer(self):
        opt_selector = optim.Adam(
            self.model.parameters(),lr=self.args.learning_rate
        )
        return opt_selector
    def _select_criterion(self):
        if self.args.maskmetric == 1:
            criterion = MaskedMSELoss(reduction='mean')
        else:
            criterion = nn.MSELoss()        
        return criterion

    #Newly added for expert module
    def _compute_loss(self, outputs, batch_y, criterion):
        """
        Handles models that return (pred, aux_loss) instead of just pred.
        """
        if isinstance(outputs, tuple):  # ExpertPatchTST returns (pred, aux_loss)
            pred, aux_loss = outputs
            f_dim = -1 if self.args.features == 'MS' else 0
            pred = pred[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            main_loss = criterion(pred, batch_y)
            # weight aux loss by hyperparam (default 0.01 if not set)
            lambda_aux = getattr(self.args, "lambda_aux", 0.01)
            return main_loss + lambda_aux * aux_loss, pred, batch_y
        else:
            f_dim = -1 if self.args.features == 'MS' else 0
            pred = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            return criterion(pred, batch_y), pred, batch_y

    #This verion is for calcauting loss for each realization for batch is actully batch size times number of attributes
    def ce_loss(self, x, y):
        with torch.no_grad():  # Compute errors using current expert0\ 
            pred0, dummay = self.base_model(x)
            f_dim = -1 if self.args.features == 'MS' else 0            
            pred0 = pred0[:, -self.args.pred_len:, f_dim:]        
            y = y[:, -self.args.pred_len:, f_dim:].to(self.device)     
            
            errors = ((pred0 - y) ** 2).mean(axis=[1])  # Per sample MSE
            errors = errors.view(self.args.batch_size*self.args.enc_in)
        if self.threshold == None:
            self.threshold = torch.quantile(errors, q= self.percentile_t)
            
        else:
            self.threshold = 0.99*self.threshold + 0.01*torch.quantile(errors, q=0.75)
            if self.percentile_avg_bool == True:
                self.threshold = 0.99*self.threshold + 0.01*torch.quantile(errors,q= self.percentile_t)
            else:
                self.threshold = torch.quantile(errors,q= self.percentile_t)
        
        

        targets = (errors > self.threshold).long()  # 0 for low error (expert0), 1 for high (expert1)


        tot_batch_size, seq_len, num_experts, num_layer = self.model.ffn_router_logits.shape
        targets_ffn = targets.view(tot_batch_size, 1, 1).expand(-1, seq_len, num_layer)
        ffn_router_logits = self.model.ffn_router_logits.permute(0,2,1,3) #class shall be in axix=1

        # Compute BCE loss per position/variable
        ffn_router_logits = ffn_router_logits[:, 0, :, :]  # Select first expert, or adjust as needed   
        targets_ffn = targets_ffn.float()
        ffn_routing_loss = self.bce_loss(ffn_router_logits, targets_ffn)
        ffn_routing_loss = ffn_routing_loss.mean()  # Mean over all dimensions
        
        _, num_experts, num_layer = self.model.attention_router_logits.shape
        targets_attention = targets.view(tot_batch_size, 1).expand(-1, num_layer)
                
        attention_router_logits = self.model.attention_router_logits

        
        # Compute BCE loss per position
        attention_router_logits = attention_router_logits[:, 0, :]  # 
        targets_attention = targets_attention.float()
        attention_routing_loss = self.bce_loss(attention_router_logits, targets_attention)
        attention_routing_loss = attention_routing_loss.mean()  # Mean over all dimensions

        return attention_routing_loss

    def z_loss(self):
        # Convert probabilities to logits (up to a constant)
        routing_logits = self.model.ffn_router_logits        
        z = routing_logits - routing_logits.mean(dim=-1, keepdim=True)
        
        # Compute log-sum-exp
        lse = torch.logsumexp(z, dim=-1)
        ffn_z_loss = lse.pow(2).mean()


        routing_logits = self.model.attention_router_logits
        z = routing_logits - routing_logits.mean(dim=-1, keepdim=True)
        # Compute log-sum-exp
        lse = torch.logsumexp(z, dim=-1)
        attention_z_loss = lse.pow(2).mean()        
        
        
        return 0.5*ffn_z_loss + 0.5*attention_z_loss



    def expert_num_selector_loss(self, out_single, out_expert, y, expert_num_selctor_logits):
        out_single, out_expert, y = out_single.to(self.device), out_expert.to(self.device), y.to(self.device)        

        with torch.no_grad():
            # Top-1: assume expert0 as default single
            err_single = self.mse_loss(out_single, y).mean(dim=[1,2])
            err_combined = self.mse_loss(out_expert, y).mean(dim=[1,2])  # Current out is combined based on current gates/k
            sel_target = (err_combined < err_single).float().unsqueeze(1)  # (batch,1)
        selector_loss = self.bce_loss(expert_num_selctor_logits, sel_target)
        return selector_loss


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        if 'Step6' in self.args.model or  'Step7' in self.args.model:
            self.expert_num_selector.eval()
            self.base_model.eval()
        preds = []
        trues = []
        tot_iter = 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)  # Add this

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)    

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():                                                               
                        if 'Expert' in self.args.model:
                            outputs, expert_loss = self.model(batch_x)
                        elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                else:
                    if 'Expert' in self.args.model:
                        outputs, expert_loss = self.model(batch_x)
                    elif 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                #loss = criterion(pred, true)
                loss, pred, true = self._compute_loss(outputs, batch_y, criterion)

                if 'Step6' not in self.args.model and 'Step7' not in self.args.model:
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())                

                if 'Step2' in self.args.model:                    
                    expert_loss = self.ce_loss(batch_x,batch_y)
                    loss += self.model.expert_loss_weight*expert_loss
                    self.expert_loss = expert_loss.item()   
                elif 'Step3' in self.args.model or 'Step5' in self.args.model:
                    self.expert_loss = 0
                elif 'Step4' in self.args.model:
                    tot_iter += 1
                    expert_loss = self.z_loss() 
                    loss = expert_loss
                    self.expert_loss = expert_loss.item()    
                elif 'Step6' in self.args.model: #selection loss is not meaningful so it has the same loss
                    out_single, _ = self.base_model(batch_x)     
                    out_expert, _ = self.model(batch_x)

                    selected_k, expert_num_selctor_logits = self.expert_num_selector(batch_x)                    
                    self.selector_loss = self.expert_num_selector_loss(out_single, out_expert, batch_y, expert_num_selctor_logits)                    
                    subbatch_x_single, subbatch_x_double, subatch_y_single, subbatch_y_double, ind_single, ind_double = create_subbatches(batch_x, batch_y, selected_k)                    
                    
                    #This phase is to train each model for targeted training data and loss from each model will be used to train each model 
                    #The single expert model will be affected by the num_expert_sel_loss and z_loss while the expert model will be affected by z_loss only
                    if ind_single.size(0) > 0:
                        out_single, _ = self.base_model(subbatch_x_single)
                        #print("out_single",[out_single[1],self.base_model.model.z_loss],flush=True,file=sys.stderr)
                        single_loss, pred, true = self._compute_loss(out_single, subatch_y_single, criterion)
                        preds.append(pred.detach().cpu().numpy())
                        trues.append(true.detach().cpu().numpy())                          
                    if ind_double.size(0) > 0:    
                        out_double, expert_loss = self.model(subbatch_x_double)
                        double_loss, pred, true = self._compute_loss(out_double, subbatch_y_double, criterion)     
                        preds.append(pred.detach().cpu().numpy())
                        trues.append(true.detach().cpu().numpy())                      

                    if ind_single.size(0) > 0 and ind_double.size(0) > 0:   
                        loss = (single_loss+double_loss)/2
                    elif ind_single.size(0) > 0:
                        loss = single_loss
                    elif ind_double.size(0) > 0:
                        loss = double_loss
                    else:
                        print("There is something wrong with expert operation",flush=True,file=sys.stderr)
                    self.expert_loss = expert_loss.item()         
                    
                elif 'Step7' in self.args.model: #selection loss is not meaningful so it has the same loss
                    out_single, _ = self.base_model(batch_x)     
                    out_expert, _ = self.model(batch_x)
                    selected_k, expert_num_selctor_logits = self.expert_num_selector(batch_x)                    
                    self.model.expert_num_selector_loss(out_single, out_expert, batch_y, expert_num_selctor_logits)                    
                    subbatch_x_single, subbatch_x_double, subatch_y_single, subbatch_y_double, ind_single, ind_double = create_subbatches(batch_x, batch_y, selected_k)                    

                    #This phase is to train each model for targeted training data and loss from each model will be used to train each model 
                    #The single expert model will be affected by the num_expert_sel_loss and z_loss while the expert model will be affected by z_loss only
                    if ind_single.size(0) > 0:
                        out_single, _ = self.base_model(subbatch_x_single)
                        #print("out_single",[out_single[1],self.base_model.model.z_loss],flush=True,file=sys.stderr)
                        single_loss, pred, true = self._compute_loss(out_single, subatch_y_single, criterion)
                        preds.append(pred.detach().cpu().numpy())
                        trues.append(true.detach().cpu().numpy())                        
                    if ind_double.size(0) > 0:    
                        out_double, expert_loss = self.model(subbatch_x_double)
                        double_loss, pred, true = self._compute_loss(out_double, subbatch_y_double, criterion)     
                        preds.append(pred.detach().cpu().numpy())
                        trues.append(true.detach().cpu().numpy())                    

                    if ind_single.size(0) > 0 and ind_double.size(0) > 0:   
                        loss = (single_loss+double_loss)/2
                    elif ind_single.size(0) > 0:
                        loss = single_loss
                    elif ind_double.size(0) > 0:
                        loss = double_loss
                    else:
                        print("There is something wrong with expert operation",flush=True,file=sys.stderr)
                    self.expert_loss = expert_loss.item()   

        

                    
                elif 'Step1' in self.args.model:                        
                    self.expert_loss = expert_loss.item()                       
                elif 'Expert' in self.args.model:
                    loss += self.model.expert_loss_weight*expert_loss
                    self.expert_loss = expert_loss.item()

                    
                total_loss.append(loss.item())


        if 'Step7' in self.args.model or 'Step6' in self.args.model: 
            preds = np.concatenate(preds,axis=0)
            trues = np.concatenate(trues,axis=0)
        else:
            preds = np.array(preds)
            trues = np.array(trues)
        
                    
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])


        self.tem_var1 = preds
        self.tem_var2 = trues

        
        if self.args.maskmetric == 1:
            mae, mse, rmse, mape, mspe, rse, corr = Maskmetric(preds, trues)
        else:
            mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse),flush=True,file=sys.stderr)

        
        total_loss = np.average(total_loss)
        total_loss = mse
        self.model.train()
        return total_loss

    def train(self, setting):
        print("test_flop:",self.args.test_flop,flush=True,file=sys.stderr)
        #if 'Step6' in self.args.model:
        #    self.args.learning_rate = 0.00001
        self.gen_base_mode()
        torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        print("train_steps:",train_steps,len(vali_loader),len(test_loader),flush=True,file=sys.stderr)
        
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        if 'Step7' in self.args.model:    
            early_stopping_base_model = EarlyStopping(patience=self.args.patience, verbose=True)

        
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if 'Step2' in self.args.model or 'Step4' in self.args.model:        
            self.model = freeze_all_parameters(self.model)
            for layer in self.model.model.backbone.encoder.layers:
                for param in layer.ffn_router.parameters():
                    param.requires_grad = True        
                for param in layer.mean_router.parameters():
                    param.requires_grad = True
                for param in layer.norm1.parameters():
                    param.requires_grad = False
                for param in layer.norm2.parameters():
                    param.requires_grad = False                       
        elif 'Step3' in self.args.model or 'Step5' in self.args.model:        
            #self.model = freeze_all_parameters(self.model)
            for layer in self.model.model.backbone.encoder.layers:
                for param in layer.ffn_router.parameters():
                    param.requires_grad = False
                for param in layer.mean_router.parameters():
                    param.requires_grad = False     
                for param in layer.norm1.parameters():
                    param.requires_grad = False
                for param in layer.norm2.parameters():
                    param.requires_grad = False                     
        elif 'Step6' in self.args.model:
            selector_optim = self._selector_optimizer()
            self.model = freeze_all_parameters(self.model)
            self.base_model = freeze_all_parameters(self.base_model)
                   
        elif 'Step7' in self.args.model:
            #selector_optim = self._selector_optimizer()
            self.expert_num_selector = freeze_all_parameters(self.expert_num_selector)
            for layer in self.model.model.backbone.encoder.layers:
                for param in layer.ffn_router.parameters():
                    param.requires_grad = False
                for param in layer.mean_router.parameters():
                    param.requires_grad = False   
                for param in layer.norm1.parameters():
                    param.requires_grad = False
                for param in layer.norm2.parameters():
                    param.requires_grad = False                        
            
            single_optim = self._single_optimizer()
            double_optim = self._double_optimizer()      

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        
        num_batch = len(train_data)//self.args.batch_size
        self.num_batch = num_batch

        tot_iter = 0
        ffn_selection0 = []
        attention_selection0 = [] 
        ffn_selection1 = []
        attention_selection1 = [] 
        ffn_selection2 = []
        attention_selection2 = [] 
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            k_list = []

            
  

            self.model.expert_loss_weight = self.model.expert_loss_weight*(100/(100+epoch))
            if self.args.curricular_bool == False:
                self.model.expert_loss_weight = self.args.z_weight
            print("debug ",self.args.percentile_avg_bool,self.args.curricular_bool,self.model.expert_loss_weight,flush=True,file=sys.stderr)

            self.model.train()


            #if 'Step2' in self.args.model or 'Step3' in self.args.model:
            #    for module in self.model.modules():
            #        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            #            module.eval()  # Freeze BN stats
            #            print("BN is now freezed ",flush=True,file=sys.stderr)

            if 'Step6' in self.args.model:
                self.base_model.eval()
                self.model.eval() 
            
            epoch_time = time.time()
            loss_list = []
            e_loss_list = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                    
                iter_count += 1
                
                model_optim.zero_grad()
                if 'Step6' in self.args.model:
                    selector_optim.zero_grad()
                elif 'Step7' in self.args.model:
                    single_optim.zero_grad()
                    double_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Expert' in self.args.model:
                            #print("batch_x: ",batch_x.size(), flush=True, file = sys.stderr)
                            outputs, expert_loss = self.model(batch_x)
                        elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        if 'Expert' in self.args.model:
                            loss += expert_loss
                            self.expert_loss = expert_loss.item()
                        train_loss.append(loss.item())
                else:
                    if 'Expert' in self.args.model:
                        #print("batch_x: ",batch_x.size(), flush=True, file = sys.stderr)
                        outputs, expert_loss = self.model(batch_x)
                        #print("z_loss:exp1",self.model.z_loss,flush=True,file=sys.stderr)                        
                    elif 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)

                    if 'Step2' in self.args.model:
                                                
                        expert_loss = self.ce_loss(batch_x,batch_y)
                        temp_loss = loss.item()
                        loss = expert_loss

                        self.expert_loss = expert_loss.item() 
                        loss_list.append(temp_loss)
                        e_loss_list.append(self.expert_loss)
                    elif 'Step3' in self.args.model or 'Step5' in self.args.model:
                        tot_iter += 1
                        self.model.z_loss_weight = self.model.z_loss_weight*(num_batch/(num_batch+5*tot_iter))   
                        self.expert_loss = 0
                    elif 'Step4' in self.args.model:
                        tot_iter += 1
                        expert_loss = self.z_loss() 
                        loss = expert_loss

                        self.expert_loss = expert_loss.item()  
                        e_loss_list.append(self.expert_loss)
                    elif 'Step6' in self.args.model:
                        tot_iter += 1
                        out_single, _ = self.base_model(batch_x)     
                        out_expert, _ = self.model(batch_x)
                        
                        selected_k, expert_num_selctor_logits = self.expert_num_selector(batch_x)    
                        k_list.append(selected_k.sum().detach()-128)
                        self.selector_loss = self.expert_num_selector_loss(out_single, out_expert, batch_y, expert_num_selctor_logits)                    
                        
                        subbatch_x_single, subbatch_x_double, subatch_y_single, subbatch_y_double, ind_single, ind_double = create_subbatches(batch_x, batch_y, selected_k)                                            
                        #This phase is to train each model for targeted training data and loss from each model will be used to train each model 
                        #The single expert model will be affected by the num_expert_sel_loss and z_loss while the expert model will be affected by z_loss only
                        if ind_single.size(0) > 0:
                            out_single, _ = self.base_model(subbatch_x_single)
                            single_loss, pred, true = self._compute_loss(out_single, subatch_y_single, criterion)
                            
                        if ind_double.size(0) > 0:    
                            out_double, expert_loss = self.model(subbatch_x_double)
                            double_loss, pred, true = self._compute_loss(out_double, subbatch_y_double, criterion)     

                        self.expert_loss = self.selector_loss.item()    
                        loss = self.selector_loss
                        e_loss_list.append(self.expert_loss)         
                    elif 'Step7' in self.args.model:
                        tot_iter += 1
                        out_single, _ = self.base_model(batch_x)     
                        out_expert, _ = self.model(batch_x)
                        selected_k, expert_num_selctor_logits = self.expert_num_selector(batch_x)                    
                        self.model.expert_num_selector_loss(out_single, out_expert, batch_y, expert_num_selctor_logits)                    
                        subbatch_x_single, subbatch_x_double, subatch_y_single, subbatch_y_double, ind_single, ind_double = create_subbatches(batch_x, batch_y, selected_k)                                            
                        #This phase is to train each model for targeted training data and loss from each model will be used to train each model 
                        #The single expert model will be affected by the num_expert_sel_loss and z_loss while the expert model will be affected by z_loss only
                        if ind_single.size(0) > 0:
                            out_single, _ = self.base_model(subbatch_x_single)
                            single_loss, pred, true = self._compute_loss(out_single, subatch_y_single, criterion)
                            
                        if ind_double.size(0) > 0:    
                            out_double, expert_loss = self.model(subbatch_x_double)
                            double_loss, pred, true = self._compute_loss(out_double, subbatch_y_double, criterion)     
                        
                            expert_loss = self.model.num_expert_sel_loss + 0.1*self.model.z_loss.sum() #weight is not considered since it is for evaluati      
                            double_loss += self.model.expert_loss_weight*self.model.z_loss.sum()

                        if ind_single.size(0) > 0 and ind_double.size(0) > 0:   
                            loss = (single_loss+double_loss)/2
                        elif ind_single.size(0) > 0:
                            loss = single_loss
                        elif ind_double.size(0) > 0:
                            loss = double_loss
                        else:
                            print("There is something wrong with expert operation",flush=True,file=sys.stderr)
                        self.expert_loss = expert_loss.item()        

                    elif 'Step1' in self.args.model:                        
                        self.expert_loss = expert_loss.item()                    
                        if self.args.num_attention_experts > 1 or self.args.num_ffn_experts > 1:
                            print("Step1 expert_loss",loss,expert_loss,self.model.model.z_loss.sum(),flush=True,file=sys.stderr)
                            print("z_loss value:", self.model.model.z_loss.sum().item(),flush=True,file=sys.stderr)
                            print("z_loss requires_grad:", self.model.model.z_loss.requires_grad,flush=True,file=sys.stderr)
                            print("z_loss grad_fn:", self.model.model.z_loss.grad_fn,flush=True,file=sys.stderr)
                            print("After adding to loss — loss.requires_grad:", loss.requires_grad,flush=True,file=sys.stderr)                              
                            loss += self.model.expert_loss_weight*self.model.model.z_loss.sum()
                          
                    
                    elif 'Expert' in self.args.model:
                        print("z_loss:exp2",[loss,self.model.expert_loss_weight,],flush=True,file=sys.stderr)
                        loss += self.model.expert_loss_weight*expert_loss
                        self.expert_loss = expert_loss.item()                    
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                elif 'Step7' in self.args.model:

                    if ind_single.size(0) > 0:    
                        single_loss.backward()
                        single_optim.step()            
                    if ind_double.size(0) > 0:
                        double_loss.backward()                                        
                        double_optim.step()     
                elif 'Step6' in self.args.model:
                    self.selector_loss.backward()
                    selector_optim.step()                         
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
            print("train step2 ebug ",[self.model.expert_loss_weight, np.mean(loss_list), np.mean(e_loss_list)],flush=True,file=sys.stderr)               
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if 'Expert' in self.args.model:
                print("train debug ",epoch + 1,train_loss, train_loss - self.model.expert_loss_weight*self.expert_loss, self.model.expert_loss_weight,self.expert_loss,flush=True,file=sys.stderr)   
            elif 'TST' in self.args.model:
                print("train debug ",epoch + 1,train_loss,flush=True,file=sys.stderr)                 
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            if 'Expert' in self.args.model:
                print("val debug ",epoch + 1,vali_loss, vali_loss - self.model.expert_loss_weight*self.expert_loss, self.model.expert_loss_weight,self.expert_loss,flush=True,file=sys.stderr)     
            elif 'TST' in self.args.model:
                print("val debug ",epoch + 1,vali_loss,flush=True,file=sys.stderr)                       
            test_loss = self.vali(test_data, test_loader, criterion)
            if 'Expert' in self.args.model:
                print("test debug ",epoch + 1,test_loss,test_loss - self.model.expert_loss_weight*self.expert_loss, self.model.expert_loss_weight,self.expert_loss,flush=True,file=sys.stderr)
            elif 'TST' in self.args.model:
                print("test debug ",epoch + 1,test_loss,flush=True,file=sys.stderr)  
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print("check path",path,vali_loss,flush=True,file=sys.stderr)
            script_dir = '/home/jhyang00/expert_transformer/Expert_PatchTST/PatchTST_supervised_optimized/'
            filename = f"{self.args.model}_{self.args.data}_{self.args.pred_len}_{self.args.random_seed}"
            script_dir = script_dir+filename

            if not os.path.exists(script_dir):
                os.makedirs(script_dir)
            
            print("script_dir path",script_dir,flush=True,file=sys.stderr)             

            if 'Step6' in self.args.model:
                early_stopping(vali_loss, self.expert_num_selector, script_dir)
            elif 'Step7' in self.args.model:                
                early_stopping(vali_loss, self.model, script_dir)
                if not os.path.exists(script_dir + '/base_model/'):
                    os.makedirs(script_dir + '/base_model/')                  
                early_stopping_base_model(vali_loss, self.base_model, script_dir+'/base_model')
            else:
                early_stopping(vali_loss, self.model, script_dir)
                
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))


            ffn_selection0.append(self.model.model.backbone.encoder.layers[0].ffn_router_selection)
            attention_selection0.append(self.model.model.backbone.encoder.layers[0].attention_router_selection)
            ffn_selection1.append(self.model.model.backbone.encoder.layers[1].ffn_router_selection)
            attention_selection1.append(self.model.model.backbone.encoder.layers[1].attention_router_selection)
            ffn_selection2.append(self.model.model.backbone.encoder.layers[2].ffn_router_selection)
            attention_selection2.append(self.model.model.backbone.encoder.layers[2].attention_router_selection)                        
            
            self.model.model.backbone.encoder.layers[0].ffn_router_selection = []
            self.model.model.backbone.encoder.layers[0].attention_router_selection = []
            self.model.model.backbone.encoder.layers[1].ffn_router_selection = []
            self.model.model.backbone.encoder.layers[1].attention_router_selection = []
            self.model.model.backbone.encoder.layers[2].ffn_router_selection = []
            self.model.model.backbone.encoder.layers[2].attention_router_selection = []

            

        

        """
        with open("ffn_selection0_"+time_string+".pkl","wb") as f:
            pickle.dump(ffn_selection0,f)
        with open("ffn_selection1_"+time_string+".pkl","wb") as f:
            pickle.dump(ffn_selection1,f)
        with open("ffn_selection2_"+time_string+".pkl","wb") as f:
            pickle.dump(ffn_selection2,f)
        with open("attention_selection0_"+time_string+".pkl","wb") as f:
            pickle.dump(attention_selection0,f)
        with open("attention_selection1_"+time_string+".pkl","wb") as f:
            pickle.dump(attention_selection1,f)
        with open("attention_selection2_"+time_string+".pkl","wb") as f:
            pickle.dump(attention_selection2,f)            
        """    
        
        best_model_path = script_dir + '/' + 'checkpoint.pth'
        if 'Step6' in self.args.model:
            self.expert_num_selector.load_state_dict(torch.load(best_model_path))
        else:
            self.model.load_state_dict(torch.load(best_model_path))
        #self.model.load_state_dict(torch.load(custom_path))
        #if 'Step1' in self.args.model:        
        #    torch.save(self.model.state_dict(), best_model_path)        
        
        #if 'Step2' in self.args.model:        
        #    torch.save(self.model.state_dict(), best_model_path)        
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        test=1
        if test:
            print('loading model')
            script_dir = '/home/jhyang00/expert_transformer/Expert_PatchTST/PatchTST_supervised_optimized/'    
            filename = f"{self.args.model}_{self.args.data}_{self.args.pred_len}_{self.args.random_seed}"
            script_dir = script_dir+filename            
            best_model_path = script_dir + '/' + 'checkpoint.pth'
            if 'Step6' in self.args.model:
                self.expert_num_selector.load_state_dict(torch.load(best_model_path))
            elif 'Step7' in self.args.model:
                self.model.load_state_dict(torch.load(best_model_path))               
                best_base_model_path = script_dir + '/base_model/' + 'checkpoint.pth'
                self.base_model.load_state_dict(torch.load(best_base_model_path))     
            else:
                self.model.load_state_dict(torch.load(best_model_path))            

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        if 'Step6' in self.args.model or 'Step7' in self.args.model: 
            self.expert_num_selector.eval()
            self.base_model.eval()
        criterion = self._select_criterion()

        tot_iter = 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Expert' in self.args.model:
                            outputs, expert_loss = self.model(batch_x)
                        elif 'Linear' in self.args.model or 'TST' in self.args.model:
                                outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Expert' in self.args.model:
                        outputs, expert_loss = self.model(batch_x)
                    elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()


                loss, pred, true = self._compute_loss(outputs, batch_y, criterion)

                if 'Step6' not in self.args.model and 'Step7' not in self.args.model:      
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())
                    inputx.append(batch_x.detach().cpu().numpy())
                    print('Step6' not in self.args.model,flush=True,file=sys.stderr)                    
                    print("Step6out_expert",self.args.model,flush=True,file=sys.stderr)
                
                if 'Step2' in self.args.model:
                    expert_loss = self.ce_loss(batch_x,batch_y)
                    loss += self.model.expert_loss_weight*expert_loss
                    self.expert_loss = expert_loss.item()    
                elif 'Step3' in self.args.model or 'Step5' in self.args.model:
                    self.expert_loss = 0
                elif 'Step4' in self.args.model:
                    tot_iter += 1
                    self.model.z_loss_weight = self.model.z_loss_weight*(self.num_batch/(self.num_batch+5*tot_iter))      
                    expert_loss = self.z_loss()  
                    loss = expert_loss#self.model.expert_loss_weight*expert_loss
                    self.expert_loss = expert_loss.item()    
                elif 'Step6' in self.args.model:
                    tot_iter += 1
                    out_single, _ = self.base_model(batch_x)     
                    out_expert, _ = self.model(batch_x)

                    selected_k, expert_num_selctor_logits = self.expert_num_selector(batch_x)    
                    self.selector_loss = self.expert_num_selector_loss(out_single, out_expert, batch_y, expert_num_selctor_logits)                    
                    
                    subbatch_x_single, subbatch_x_double, subatch_y_single, subbatch_y_double, ind_single, ind_double = create_subbatches(batch_x, batch_y, selected_k)                    
                    
                    #This phase is to train each model for targeted training data and loss from each model will be used to train each model 
                    #The single expert model will be affected by the num_expert_sel_loss and z_loss while the expert model will be affected by z_loss only
                    if ind_single.size(0) > 0:
                        out_single, _ = self.base_model(subbatch_x_single)
                        single_loss, pred, true = self._compute_loss(out_single, subatch_y_single, criterion)
                        preds.append(pred.detach().cpu().numpy())
                        trues.append(true.detach().cpu().numpy())                        
                    if ind_double.size(0) > 0:    
                        out_double, expert_loss = self.model(subbatch_x_double)
                        double_loss, pred, true = self._compute_loss(out_double, subbatch_y_double, criterion)     
                        preds.append(pred.detach().cpu().numpy())
                        trues.append(true.detach().cpu().numpy())                    
                    self.expert_loss = self.selector_loss.item()    
                    loss = self.selector_loss
                elif 'Step7' in self.args.model:
                    tot_iter += 1
                    out_single, _ = self.base_model(batch_x)     
                    out_expert, _ = self.model(batch_x)
                    selected_k, expert_num_selctor_logits = self.expert_num_selector(batch_x)                    
                    self.model.expert_num_selector_loss(out_single, out_expert, batch_y, expert_num_selctor_logits)                    
                    subbatch_x_single, subbatch_x_double, subatch_y_single, subbatch_y_double, ind_single, ind_double = create_subbatches(batch_x, batch_y, selected_k)                    

                    
                    #This phase is to train each model for targeted training data and loss from each model will be used to train each model 
                    #The single expert model will be affected by the num_expert_sel_loss and z_loss while the expert model will be affected by z_loss only
                    if ind_single.size(0) > 0:
                        out_single, _ = self.base_model(subbatch_x_single)
                        single_loss, pred, true = self._compute_loss(out_single, subatch_y_single, criterion)
                        preds.append(pred.detach().cpu().numpy())
                        trues.append(true.detach().cpu().numpy())                        
                    if ind_double.size(0) > 0:    
                        out_double, expert_loss = self.model(subbatch_x_double)
                        double_loss, pred, true = self._compute_loss(out_double, subbatch_y_double, criterion)     
                        preds.append(pred.detach().cpu().numpy())
                        trues.append(true.detach().cpu().numpy())                    
                        expert_loss = self.model.num_expert_sel_loss + 0.1*self.model.z_loss.sum() #weight is not considered since it is for evaluati      
                        double_loss += self.model.expert_loss_weight*self.model.z_loss.sum()

                    if ind_single.size(0) > 0 and ind_double.size(0) > 0:   
                        loss = (single_loss+double_loss)/2
                    elif ind_single.size(0) > 0:
                        loss = single_loss
                    elif ind_double.size(0) > 0:
                        loss = double_loss
                    else:
                        print("There is something wrong with expert operation",flush=True,file=sys.stderr)
                    self.expert_loss = expert_loss.item()        


                    if ind_single.size(0) > 0 and ind_double.size(0) > 0:   
                        loss = single_loss+double_loss
                    elif ind_single.size(0) > 0:
                        loss = single_loss
                    elif ind_double.size(0) > 0:
                        loss = double_loss
                    else:
                        print("There is something wrong with expert operation",flush=True,file=sys.stderr)
                    self.expert_loss = expert_loss.item()      

                  
                    
                elif 'Step1' in self.args.model:                        
                    self.expert_loss = expert_loss.item()                       
                elif 'Expert' in self.args.model:                
                    loss += self.model.expert_loss_weight*expert_loss
                    self.expert_loss = expert_loss.item()                  
            
            
                if 'Step6' not in self.args.model and 'Step7' not in self.args.model:      

                    if i % 20 == 0:
                        input = batch_x.detach().cpu().numpy()    
                        true = batch_y.detach().cpu().numpy()  # Move true to CPU
                        pred = pred.detach().cpu().numpy()     # Move pred to CPU                    
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        #print("test_flop1:",self.args.test_flop,flush=True,file=sys.stderr)

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        #print("test_flop2:",self.args.test_flop,flush=True,file=sys.stderr)

        if 'Step6' in self.args.model or 'Step7' in self.args.model: 
            preds = np.concatenate(preds,axis=0)
            trues = np.concatenate(trues,axis=0)
            #inputx = np.concatenate(inputx,axis=0)        
        else:
            preds = np.array(preds)
            trues = np.array(trues)
            #inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        

        
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if self.args.maskmetric == 1:
            mae, mse, rmse, mape, mspe, rse, corr = Maskmetric(preds, trues)
        else:
            mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse),flush=True,file=sys.stderr)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        return mse

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Expert' in self.args.model:
                            outputs, expert_loss = self.model(batch_x)
                        elif 'Linear' in self.args.model or 'TST' in self.args.model:
                                outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Expert' in self.args.model:
                        outputs, expert_loss = self.model(batch_x)
                    elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if 'Step2' in self.args.model:
                    expert_loss = self.ce_loss(batch_x,batch_y)
                    loss += self.model.expert_loss_weight*expert_loss
                    self.expert_loss = expert_loss.item()  
                elif 'Step3' in self.args.model or 'Step5' in self.args.model:
                    expert_loss = self.model.z_loss.sum() #Since after training it will depend on z_loss
                    loss += self.model.expert_loss_weight*expert_loss
                    self.expert_loss = expert_loss.item()  
                elif 'Step6' in self.args.model or 'Step7' in self.args.model:
                    out_single = self.base_model(batch_x)     
                    selected_k, expert_num_selctor_logits = self.expert_num_selector(first_layer_output)                    
                    self.model.expert_num_selector_loss(self, out_single, out_expert, batch_y, expert_num_selctor_logits)                    
                    subbatch_x_single, subbatch_x_double, subatch_y_single, subbatch_y_double, ind_single, ind_double = create_subbatches(batch_x, batch_y, selected_k)                    
                    #This phase is to train each model for targeted training data and loss from each model will be used to train each model 
                    #The single expert model will be affected by the num_expert_sel_loss and z_loss while the expert model will be affected by z_loss only
                    if ind_single.size(0) > 0:
                        out_single = self.base_model(subbatch_x_single)
                        single_loss, pred, true = self._compute_loss(out_single, subatch_y_single, criterion)
                    if ind_double.size(0) > 0:    
                        out_double, expert_loss = self.model(subbatch_x_double)
                        double_loss, pred, true = self._compute_loss(out_double, subbatch_y_double, criterion)                     
                    #expert_loss = self.num_expert_sel_loss + 0.1*self.model.z_loss.sum() #weight is not considered since it is for evaluati                    
                    loss = single_loss+double_loss+self.model.expert_loss_weight*expert_loss
                    self.expert_loss = expert_loss.item()                        
                elif 'Expert' in self.args.model:
                    self.expert_loss = expert_loss.item()                              

                if  'Step6' in self.args.model or 'Step7' in self.args.model:
                    pred = out_single.detach().cpu().numpy()  # .squeeze()
                    preds.append(pred)
                    pred = out_double.detach().cpu().numpy()  # .squeeze()
                    preds.append(pred)                    
                else:
                    pred = outputs.detach().cpu().numpy()  # .squeeze()
                    preds.append(pred)
        
        
        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
