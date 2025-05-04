"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.weight_positive = nn.Parameter(torch.ones(ndim))
        self.weight_negative = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.bias_positive = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.bias_negative = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input, input_error=None, operator_noise=0, infos_GPU=None, infos_CPU=None, layer_counter=None):
        eps=1e-5
        if input_error is None:
            input_lower = input.clone()
            input_upper = input.clone()
        else:
            input_lower = input - input_error
            input_upper = input + input_error

        if layer_counter is not None:
            layer_info = {
                'layer_num': layer_counter[0],
                'layer_type': 'LayerNorm',
                'x': input,
                'x_lower': input_lower,
                'x_upper': input_upper,
                'y': None,
                'y_lower': None,
                'y_upper': None,
            }
        else:
            layer_info = {}
        
        
        def check_bounds(name, real_output, lower_bound, upper_bound):

            real_minus_lower = lower_bound - real_output
            real_minus_upper = real_output - upper_bound
            real_outside = torch.maximum(real_minus_lower, real_minus_upper)
            real_outside = torch.clamp(real_outside, min=0.0)

            bound_width = (upper_bound - lower_bound).max().item()

            print(f"[{name}] Output outside interval max: {real_outside.max().item()}")
            print(f"[{name}] Interval width max: {bound_width}")
            
            
        with torch.no_grad():
            # Step 1: 均值传播 (mean of input区间)
            input_mean = input.mean(dim=(-2, -1), keepdim=True)

            mean_lower = input_lower.mean(dim=(-2, -1), keepdim=True) - operator_noise
            mean_upper = input_upper.mean(dim=(-2, -1), keepdim=True) + operator_noise

            # check_bounds("Step_1 - Mean:", input_mean, mean_lower, mean_upper)

            # Step 2: (input - mean)区间传播
            input_centered_diff = input - input_mean

            x_lower_centered_min = input_lower - mean_upper 
            x_upper_centered_max = input_upper - mean_lower 

            centered_min = x_lower_centered_min - operator_noise
            centered_max = x_upper_centered_max + operator_noise

            # print(input_centered_diff.dtype, centered_max.dtype)
            # print(input_centered_diff.max() - centered_max.max())

            # check_bounds("Step_2 - Centered Input:", input_centered_diff, centered_min, centered_max)

            # Step 3: 方差传播
            input_square = input_centered_diff ** 2
            # Step 3.1: (x_i - mean)^2 区间传播                              
            centered_min_sq = centered_min ** 2
            centered_max_sq = centered_max ** 2

            squared_lower = torch.where(
                (centered_min <= 0) & (centered_max >= 0),
                torch.zeros_like(centered_min),  # 跨0，最小值是0
                torch.minimum(centered_min_sq, centered_max_sq)
            ) 
            squared_upper = torch.maximum(centered_min_sq, centered_max_sq) 

            ##### 平方错误复现！
#             error_mask = input_square > squared_upper
#             if error_mask.any():
#                 diff = (input_square - squared_upper)[error_mask]
#                 abs_diff_all = diff.abs()
#                 abs_diff_max = abs_diff_all.max().item()

#                 print("[验证] 有平方误差超出边界！最大超出值: ", abs_diff_max)

#                 # 获取 top-3 最大误差的索引（在 error_mask 压缩空间中的索引）
#                 topk_vals, topk_indices = torch.topk(abs_diff_all, k=min(3, abs_diff_all.numel()))
#                 offending_indices = torch.nonzero(error_mask, as_tuple=False)[topk_indices]

#                 for idx in offending_indices:
#                     idx = tuple(idx.tolist())
#                     v_input = input_centered_diff[idx].item()
#                     v_max = centered_max[idx].item()
#                     v_min = centered_min[idx].item()
#                     v_input_sq = input_square[idx].item()
#                     v_max_sq = squared_upper[idx].item()
#                     v_min_sq = squared_lower[idx].item()
#                     print(f"  - 位置 {idx}: input = {v_input:.8f}, min = {v_min:.8f}, max = {v_max:.8f}, input² = {v_input_sq:.8f}, min² = {v_min_sq:.8f}, max² = {v_max_sq:.8f}, Δmin = {v_min_sq - v_min_sq:.8e}, Δmax = {v_input_sq - v_max_sq:.8e}")

#                 if abs_diff_max > 1e-6:
#                     print("❌ 超出误差容忍范围！终止程序。")
#                     exit()
#             else:
#                 print("[验证] 所有平方值都在推导区间内 ✅")     
            ##### 平方错误复现！


            # check_bounds("Step_3.1 - Square:", input_square, squared_lower, squared_upper)

            # Step 3.2: 取mean
            input_variance = input_square.mean(dim=(-2, -1), keepdim=True)

            var_lower = squared_lower.mean(dim=(-2, -1), keepdim=True) - operator_noise
            var_upper = squared_upper.mean(dim=(-2, -1), keepdim=True) + operator_noise

            # check_bounds("Step_3.2 - Mean square:", input_variance, var_lower, var_upper)

            # Step 4: 归一化传播
            # Step 4.1: 开平方传播
            input_division = torch.sqrt(input_variance + eps)


            denom_lower = torch.sqrt(var_lower + eps)
            denom_upper = torch.sqrt(var_upper + eps)

            # check_bounds("Step_4.1 - Squared root:", input_division, denom_lower, denom_upper)

            # Step 4.2: 归一化传播       
            input_normalized = input_centered_diff / input_division

            input_candidates = torch.stack([
                (input_lower - mean_lower) / denom_lower,
                (input_lower - mean_lower) / denom_upper,
                (input_lower - mean_upper) / denom_lower,
                (input_lower - mean_upper) / denom_upper,
                (input_upper - mean_lower) / denom_lower,
                (input_upper - mean_lower) / denom_upper,
                (input_upper - mean_upper) / denom_lower,
                (input_upper - mean_upper) / denom_upper,
            ], dim=0)

            input_lower_norm = input_candidates.min(dim=0)[0] - operator_noise
            input_upper_norm = input_candidates.max(dim=0)[0] + operator_noise

            # check_bounds("Step_4.2 - Normalized:", input_normalized, input_lower_norm, input_upper_norm)

            # Step 5: 仿射变换 (weight和bias)
            weight = self.weight.to(input.device)
            bias = self.bias.to(input.device) if self.bias is not None else torch.zeros_like(self.weight).to(input.device)

            if bias is not None:
                output_normalized = input_normalized * weight + bias
            else:
                output_normalized = input_normalized * weight    

            # 必须区分正负weight                
            weight_r = weight.view(1, 1, -1)
            if bias is not None:
                bias_r = bias.view(1, 1, -1)

            mask_positive = (weight_r >= 0).float()
            mask_negative = (weight_r < 0).float()

            weight_positive = weight_r * mask_positive
            weight_negative = weight_r * mask_negative

            if bias is not None:
                output_lower = input_lower_norm * weight_positive + input_upper_norm * weight_negative + bias_r - operator_noise
                output_upper = input_upper_norm * weight_positive + input_lower_norm * weight_negative + bias_r + operator_noise
            else:
                output_lower = input_lower_norm * weight_positive + input_upper_norm * weight_negative - operator_noise
                output_upper = input_upper_norm * weight_positive + input_lower_norm * weight_negative + operator_noise

            # check_bounds("Step_5 - Affine:", output_normalized, output_lower, output_upper)

            # --- 正常forward ---
            x = F.layer_norm(input, self.weight.shape, weight, bias, eps)

            # layer_info['y'] = x
            # ！！！关键！使用内部F.layer_norm函数和我们的正则化实现不同，导致结果不一致，这里为了展示方法正确性，输出改为我们标准正则化输出
            layer_info['y'] = output_normalized

            layer_info['y_lower'] = output_lower
            layer_info['y_upper'] = output_upper

            if infos_GPU is not None and infos_CPU is None:
                infos_GPU.append(layer_info)
                print(f"Appending layer {layer_counter[0]} to infos_GPU, type = {layer_info['layer_type']}")
                real_minus_lower = output_lower - output_normalized
                real_minus_upper = output_normalized - output_upper
                real_outside = torch.maximum(real_minus_lower, real_minus_upper)
                real_outside = torch.clamp(real_outside, min=0.0)  # 只保留超出的部分
                print("Output range width: ", (output_upper - output_lower).max().item())
                print("Real GPU output range error: ", real_outside.max().item())

            if infos_CPU is not None:
                infos_CPU.append(layer_info)
                print(f"Appending layer {layer_counter[0]} to infos_CPU, type = {layer_info['layer_type']}")
                # # 确保都在CPU上做比较
                # output_normalized_cpu = output_normalized.detach().cpu()
                # x_cpu = x.detach().cpu()
                # diff = (output_normalized_cpu - x_cpu).abs().max().item()
                # print(f"Manual norm vs. torch norm diff: {diff}")


            if layer_counter is not None:
                layer_counter[0] += 1

            output_normalized_lower = output_lower
            output_normalized_upper = output_upper

        return x, output_normalized_lower, output_normalized_upper

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_attn_negative = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_attn_positive = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            

    def forward(self, x, x_error=None, operator_noise=0, infos_GPU=None, infos_CPU=None, layer_counter=None):
        if x_error is None:
            x_lower = x.clone()
            x_upper = x.clone()
        else:
            x_lower = x - x_error
            x_upper = x + x_error
        
        if layer_counter is not None:
            layer_info = {
                'layer_num': layer_counter[0],
                'layer_type': 'SelfAttention',
                'x': x,
                'x_lower': x_lower,
                'x_upper': x_upper,
                'y': None,
                'y_lower': None,
                'y_upper': None,
            }
        else:
            layer_info = {}
            
        def check_bounds(name, real_output, lower_bound, upper_bound):
            # 找出mask掉的位置（inf的位置）
            invalid_mask = torch.isinf(lower_bound) | torch.isinf(upper_bound)

            # 只在有效区域做比较
            real_output_valid = real_output.masked_fill(invalid_mask, 0.0)
            lower_bound_valid = lower_bound.masked_fill(invalid_mask, 0.0)
            upper_bound_valid = upper_bound.masked_fill(invalid_mask, 0.0)

            real_minus_lower = lower_bound_valid - real_output_valid
            real_minus_upper = real_output_valid - upper_bound_valid
            real_outside = torch.maximum(real_minus_lower, real_minus_upper)
            real_outside = torch.clamp(real_outside, min=0.0)

            bound_width = (upper_bound_valid - lower_bound_valid).max().item()

            print(f"[{name}] Output outside interval max: {real_outside.max().item()}")
            print(f"[{name}] Interval width max: {bound_width}")
        
        # CPU模式下检查输入是否对齐
        # if infos_CPU is not None:
        #     print(f"Attention input diff: {(x.to('cpu') - infos_GPU[layer_counter[0]]['x'].to('cpu')).abs().max()}")
            
        
        # 保证x_lower和x_upper和c_attn_positive的weight在同一个device
        device_c_attn = self.c_attn.weight.device
        if x.device != device_c_attn:
            x = x.to(device_c_attn)
        if x_lower.device != device_c_attn:
            x_lower = x_lower.to(device_c_attn)
        if x_upper.device != device_c_attn:
            x_upper = x_upper.to(device_c_attn)

        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
            

        # Step 1: Calculate Q K V and their interval bounds
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim           
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        with torch.no_grad():
            W = self.c_attn.weight
            W_pos = torch.clamp(W, min=0)
            W_neg = torch.clamp(W, max=0)
            
            out_lower = x_lower @ W_pos.T + x_upper @ W_neg.T
            out_upper = x_upper @ W_pos.T + x_lower @ W_neg.T

            if self.c_attn.bias is not None:
                out_lower += self.c_attn.bias
                out_upper += self.c_attn.bias

            q_lower, k_lower, v_lower = out_lower.split(self.n_embd, dim=2)
            q_upper, k_upper, v_upper = out_upper.split(self.n_embd, dim=2)

        # 正常 forward 输出 reshape
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 区间上下界 reshape
        with torch.no_grad():
            k_lower = k_lower.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) - operator_noise
            q_lower = q_lower.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) - operator_noise
            v_lower = v_lower.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) - operator_noise
 
            k_upper = k_upper.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) + operator_noise
            q_upper = q_upper.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) + operator_noise
            v_upper = v_upper.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) + operator_noise

            
        
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # 关闭self.flash保证GPU/CPU对齐
        self.flash = False
        
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)

            with torch.no_grad():
                y_q_lower_k_lower_v_lower = torch.nn.functional.scaled_dot_product_attention(q_lower, k_lower, v_lower, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
                y_q_lower_k_lower_v_upper = torch.nn.functional.scaled_dot_product_attention(q_lower, k_lower, v_upper, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
                y_q_lower_k_upper_v_lower = torch.nn.functional.scaled_dot_product_attention(q_lower, k_upper, v_lower, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
                y_q_lower_k_upper_v_upper = torch.nn.functional.scaled_dot_product_attention(q_lower, k_upper, v_upper, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
                y_q_upper_k_lower_v_lower = torch.nn.functional.scaled_dot_product_attention(q_upper, k_lower, v_lower, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
                y_q_upper_k_lower_v_upper = torch.nn.functional.scaled_dot_product_attention(q_upper, k_lower, v_upper, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
                y_q_upper_k_upper_v_lower = torch.nn.functional.scaled_dot_product_attention(q_upper, k_upper, v_lower, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
                y_q_upper_k_upper_v_upper = torch.nn.functional.scaled_dot_product_attention(q_upper, k_upper, v_upper, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            
                y_all = torch.concat([
                    y_q_lower_k_lower_v_lower.unsqueeze(0),
                    y_q_lower_k_lower_v_upper.unsqueeze(0),
                    y_q_lower_k_upper_v_lower.unsqueeze(0), 
                    y_q_lower_k_upper_v_upper.unsqueeze(0),
                    y_q_upper_k_lower_v_lower.unsqueeze(0), 
                    y_q_upper_k_lower_v_upper.unsqueeze(0), 
                    y_q_upper_k_upper_v_lower.unsqueeze(0),
                    y_q_upper_k_upper_v_upper.unsqueeze(0),
                ], dim=0)

                y_lower = y_all.min(dim=0)[0] - operator_noise
                y_upper = y_all.max(dim=0)[0] + operator_noise

        else:
            # Step 2:calculate of attention score and its interval bound
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            with torch.no_grad():
                # q_lower, q_upper: (B, nh, T, hs)
                # k_lower, k_upper: (B, nh, T, hs)

                chunk_size = 32  # 可以根据内存大小调整

                B, nh, T, hs = q_lower.shape
                att_lower = torch.zeros(B, nh, T, T, device=q_lower.device)
                att_upper = torch.zeros(B, nh, T, T, device=q_lower.device)

                # k方向一次性扩展，不分块
                k_lower_expand = k_lower.unsqueeze(2)  # (B, nh, 1, T, hs)
                k_upper_expand = k_upper.unsqueeze(2)

                for start in range(0, T, chunk_size):
                    end = min(start + chunk_size, T)

                    # 只在q方向分块
                    q_lower_chunk = q_lower[:, :, start:end, :].unsqueeze(3)  # (B, nh, chunk, 1, hs)
                    q_upper_chunk = q_upper[:, :, start:end, :].unsqueeze(3)

                    # 4种乘积组合 (注意这里k仍然是T，不切块)
                    prod_ll = q_lower_chunk * k_lower_expand
                    prod_lu = q_lower_chunk * k_upper_expand
                    prod_ul = q_upper_chunk * k_lower_expand
                    prod_uu = q_upper_chunk * k_upper_expand

                    # 每一项的局部最小最大
                    prod_min = torch.min(torch.min(prod_ll, prod_lu), torch.min(prod_ul, prod_uu))
                    prod_max = torch.max(torch.max(prod_ll, prod_lu), torch.max(prod_ul, prod_uu))
                    
                    # 防止爆炸
                    prod_min = torch.clamp(prod_min, min=-1e9, max=1e9)
                    prod_max = torch.clamp(prod_max, min=-1e9, max=1e9)

                    # sum over最后一个特征维度 hs
                    # test
                    # print(f"Before sum prod_min max: {prod_min.abs().max().item()}")
                    # print(f"Before sum prod_max max: {prod_max.abs().max().item()}")
                    
                    chunk_att_lower = prod_min.sum(dim=-1) / math.sqrt(hs)  # (B, nh, chunk, T)
                    chunk_att_upper = prod_max.sum(dim=-1) / math.sqrt(hs)
                    
                    # test
                    # print(f"After sum chunk_att_lower any inf: {torch.isinf(chunk_att_lower).any()}")
                    # print(f"After sum chunk_att_upper any inf: {torch.isinf(chunk_att_upper).any()}")

                    # 正确写回原始 att_lower和att_upper
                    att_lower[:, :, start:end, :] = chunk_att_lower - operator_noise
                    att_upper[:, :, start:end, :] = chunk_att_upper + operator_noise

            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            with torch.no_grad():
                att_lower = att_lower.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                att_upper = att_upper.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            
            # att logits中心值算完
            # check_bounds("Step2-AttentionLogits", att, att_lower, att_upper)
            
            # Step 3:calculate softmax and its interval bound
            
            with torch.no_grad():
                
                B, nh, T, _ = att_lower.shape  # (B: batch, nh: num_heads, T: seq_len)

                # === 1. 构造 logits x[i,j] 的上下界 ===
                x = att
                x_lower = att_lower
                x_upper = att_upper

                # check_bounds("Step3.1-Logits", x, x_lower, x_upper)

                # # === 2. 修复 (-inf) - (-inf) 导致的 NaN ===
                # invalid_mask = (x_lower == float('-inf')) & (x_upper == float('-inf'))
                # x_lower = x_lower.masked_fill(invalid_mask, 0.0)
                # x_upper = x_upper.masked_fill(invalid_mask, 0.0)

                # === 3. 计算 exp(x[i,j]) 的上下界 ===
                exp_x = torch.exp(x)

                is_pos_inf = (x_lower == float('inf')) & (x_upper == float('inf'))
                is_neg_inf = (x_lower == float('-inf')) & (x_upper == float('-inf'))
                is_safe = ~(is_pos_inf | is_neg_inf)

                exp_x_lower = torch.zeros_like(x_lower)
                exp_x_upper = torch.zeros_like(x_upper)

                exp_x_lower[is_safe] = torch.exp(x_lower[is_safe])
                exp_x_upper[is_safe] = torch.exp(x_upper[is_safe])

                exp_x_lower[is_pos_inf] = float('inf')
                exp_x_upper[is_pos_inf] = float('inf')

                exp_x_lower[is_neg_inf] = 0.0
                exp_x_upper[is_neg_inf] = 0.0

                # check_bounds("Step3.2-exp(x)", exp_x, exp_x_lower, exp_x_upper)

                # === 4. 构造 softmax 分母上下界 ===
                denom = exp_x.sum(dim=-1, keepdim=True)  # (B, nh, T, 1)
                denom_lower = exp_x_lower.sum(dim=-1, keepdim=True)
                denom_upper = exp_x_upper.sum(dim=-1, keepdim=True)

                # check_bounds("Step3.3-Denominator", denom, denom_lower, denom_upper)

                # === 5. 构造完整 softmax[i,j] 的上下界 ===
                softmax = exp_x / denom
                softmax_lower = exp_x_lower / denom_upper
                softmax_upper = exp_x_upper / denom_lower

                # check_bounds("Step3.4-Softmax", softmax, softmax_lower, softmax_upper)

                # === 6. 手动 softmax 验证 ===
                # softmax_manual = F.softmax(x, dim=-1)
                # max_diff = (softmax_manual - softmax).abs().max().item()
                # print("🧪 Max diff (manual vs. F.softmax):", max_diff)
                    

                # 最终得到 att_lower, att_upper 更新为 softmax后的上下界
                att_lower = softmax_lower - operator_noise
                att_upper = softmax_upper + operator_noise
                
            att = F.softmax(att, dim=-1)

            att = self.attn_dropout(att)
            with torch.no_grad():
                att_lower = self.attn_dropout(att_lower)
                att_upper = self.attn_dropout(att_upper)
                
            # check_bounds("Step3-AttentionWeights", att, att_lower, att_upper)

            # Step 3:calculate context and its inerval bound
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            with torch.no_grad():                
                y_lower_lower = att_lower @ v_lower # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
                y_lower_upper = att_lower @ v_upper # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
                y_upper_lower = att_upper @ v_lower # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
                y_upper_upper = att_upper @ v_upper # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
                y_all = torch.concat([
                    y_lower_lower.unsqueeze(0), 
                    y_lower_upper.unsqueeze(0), 
                    y_upper_lower.unsqueeze(0), 
                    y_upper_upper.unsqueeze(0), 
                ], dim=0)
                y_lower = y_all.min(dim=0)[0] - operator_noise
                y_upper = y_all.max(dim=0)[0] + operator_noise

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        with torch.no_grad():
            y_lower = y_lower.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
            y_upper = y_upper.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # check_bounds("Step4-Context", y, y_lower, y_upper)
        
        # Step 5: output projection and its interval bound
        y = self.resid_dropout(self.c_proj(y))
        # eval模式下dropout为恒等映射，无须考虑误差
        # 现在也要计算 y_lower, y_upper的变换
        with torch.no_grad():
            weight = self.c_proj.weight
            bias = self.c_proj.bias

            mask_positive = (weight >= 0).float()
            mask_negative = (weight < 0).float()

            weight_positive = weight * mask_positive
            weight_negative = weight * mask_negative

            y_lower_proj = (y_lower @ weight_positive.t()) + (y_upper @ weight_negative.t())
            y_upper_proj = (y_upper @ weight_positive.t()) + (y_lower @ weight_negative.t())

            if bias is not None:
                y_lower_proj += bias
                y_upper_proj += bias

            y_lower = y_lower_proj - operator_noise
            y_upper = y_upper_proj + operator_noise
            
        # check_bounds("Step5-Project", y, y_lower, y_upper)
            
        # GPU/NPU模式下，只有输出值x有价值
        x = y
        x_lower = y_lower
        x_upper = y_upper
        layer_info['y'] = x
        layer_info['y_lower'] = x_lower
        layer_info['y_upper'] = x_upper

        # GPU/NPU模式下，只有输出值y有价值
        if infos_GPU is not None and infos_CPU is None:              
            infos_GPU.append(layer_info)
            print(f"Appending layer {layer_counter[0]} to infos_GPU, type = {layer_info['layer_type']}")
            # ### test
            real_minus_lower = y_lower - y
            real_minus_upper = y - y_upper
            real_outside = torch.maximum(real_minus_lower, real_minus_upper)
            real_outside = torch.clamp(real_outside, min=0.0)  # 只保留超出的部分
            print("Output range width: ", (y_upper - y_lower).max().item())
            print("Real GPU output range error: ", real_outside.max().item())

        # CPU模式下，只有输出上下界有价值
        if infos_CPU is not None:
            infos_CPU.append(layer_info)
            print(f"Appending layer {layer_counter[0]} to infos_CPU, type = {layer_info['layer_type']}")
            # 确保都在CPU上做比较
                    # output_normalized_cpu = output_normalized.detach().cpu()
                    # x_cpu = x.detach().cpu()
                    # diff = (output_normalized_cpu - x_cpu).abs().max().item()
                    # print(f"Manual norm vs. torch norm diff: {diff}")
        
        if layer_counter is not None:
            layer_counter[0] = layer_counter[0] + 1
        
        return y, y_lower, y_upper

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_fc_negative    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_fc_positive    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj_negative  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj_positive  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, x_error=None, operator_noise=0, infos_GPU=None, infos_CPU=None, layer_counter=None): 
        if x_error is None:
            x_lower = x.clone()
            x_upper = x.clone()
        else:
            x_lower = x - x_error
            x_upper = x + x_error
        
        if layer_counter is not None:
            layer_info = {
                'layer_num': layer_counter[0],
                'layer_type': 'MLP',
                'x': x,
                'x_lower': x_lower,
                'x_upper': x_upper,
                'y': None,
                'y_lower': None,
                'y_upper': None,
            }
        else:
            layer_info = {}
            
        if x.device != self.c_fc.weight.device:
            x = x.to(self.c_fc.weight.device)
        if x_lower.device != self.c_fc.weight.device:
            x_lower = x_lower.to(self.c_fc.weight.device)
        if x_upper.device != self.c_fc.weight.device:
            x_upper = x_upper.to(self.c_fc.weight.device)
       
        
        # 第一步
        x_fc = self.c_fc(x)
        
        W = self.c_fc.weight
        b = self.c_fc.bias
        x_forward_fc = x @ W.T
        if b is not None:
            x_forward_fc += b
        
        if infos_GPU is not None and infos_CPU is None:
            layer_info['GPU_inter_outputs'] = [x_fc.detach().clone()]
            layer_info['x_forward_fc'] = x_forward_fc.detach().clone()
            
        if infos_CPU is not None:
            x_fc_cpu = x_fc.detach().clone()
            x_fc2_cpu = x_forward_fc.detach().clone()
            layer = next((item for item in infos_GPU if item["layer_num"] == layer_counter[0]), None)
            x_fc_gpu = layer['GPU_inter_outputs'][0].to('cpu')
            print("---------MLP TEST---------- ")
            print("CPU/NPU C_FC error: ", (x_fc_gpu - x_fc_cpu).abs().max().item())
            x_fc_manual_gpu = layer['x_forward_fc'].to('cpu')
            print("CPU/NPU FC_forward_manual error: ", (x_fc_manual_gpu - x_fc2_cpu).abs().max().item())
        
        # --- IBP for first Linear (c_fc) ---
        with torch.no_grad():           
            W = self.c_fc.weight
            b = self.c_fc.bias

            W_pos = torch.clamp(W, min=0)
            W_neg = torch.clamp(W, max=0)

            x_lower_fc = x_lower @ W_pos.T + x_upper @ W_neg.T - operator_noise
            x_upper_fc = x_upper @ W_pos.T + x_lower @ W_neg.T + operator_noise

            if b is not None:
                x_lower_fc += b
                x_upper_fc += b
        
        # Old version
#         with torch.no_grad():
#             mask_positive = self.c_fc.weight.data >= 0
#             mask_negative = self.c_fc.weight.data < 0
#             c_fc_negative = self.c_fc.weight.data.clone()
#             c_fc_negative[mask_positive] = 0
#             self.c_fc_negative.weight = nn.Parameter(c_fc_negative)
#             c_fc_positive = self.c_fc.weight.data.clone()
#             c_fc_positive[mask_negative] = 0
#             self.c_fc_positive.weight = nn.Parameter(c_fc_positive)
#         with torch.no_grad():
#             x_lower_negative = self.c_fc_negative(x_lower)
#             x_lower_positive = self.c_fc_positive(x_lower)
#             x_upper_negative = self.c_fc_negative(x_upper)
#             x_upper_positive = self.c_fc_positive(x_upper)

#             x_lower = x_lower_positive + x_upper_negative - operator_noise
#             x_upper = x_upper_positive + x_lower_negative + operator_noise

        # 第二步：激活函数
        x_gelu = self.gelu(x_fc)
        
        if infos_GPU is not None and infos_CPU is None:
            layer_info['GPU_inter_outputs'].append(x_gelu.detach().clone())
            
        if infos_CPU is not None:
            x_act_cpu = x_gelu.detach().clone()
            layer = next((item for item in infos_GPU if item["layer_num"] == layer_counter[0]), None)
            x_act_gpu = layer['GPU_inter_outputs'][1].to('cpu')
            x_fc_gpu = layer['GPU_inter_outputs'][0].to('cpu')
            x_gelu_2_cpu = self.gelu(x_fc_gpu)
            print("CPU/NPU GELU ONLY error: ", (x_act_gpu - x_gelu_2_cpu).abs().max().item())
            print("CPU/NPU GELU error: ", (x_act_gpu - x_act_cpu).abs().max().item())
        
        with torch.no_grad():
            # gelu分段单调临界点为-0.75246142，精确到小数点后8位
            x_crit = -0.75246142
            gelu_lower = self.gelu(x_lower_fc)
            gelu_upper = self.gelu(x_upper_fc)

            # 构造掩码，判断哪些位置包含临界点
            cross_crit_mask = (x_lower_fc < x_crit) & (x_upper_fc > x_crit)

            # 默认情况：端点 min/max
            min_val = torch.minimum(gelu_lower, gelu_upper)
            max_val = torch.maximum(gelu_lower, gelu_upper)

            # 临界点处的 gelu 值（是常数）
            crit_val = self.gelu(torch.tensor(x_crit, device=x_lower_fc.device))

            # 用 mask 替换下界中跨越临界点的那些位置
            x_lower_act = torch.where(cross_crit_mask, crit_val, min_val)
            x_upper_act = max_val

            # 加入 operator_noise
            x_lower_act = x_lower_act - operator_noise
            x_upper_act = x_upper_act + operator_noise

            # # IBP through GELU。单调，所以直接激活上下界。后续：GELU不是单调的，有问题，弃用
            # x_lower_act = self.gelu(x_lower_fc) - operator_noise
            # x_upper_act = self.gelu(x_upper_fc) + operator_noise
            
            

        # old version
#         with torch.no_grad():
#             x_lower = self.gelu(x_lower) - operator_noise
#             x_upper = self.gelu(x_upper) + operator_noise
            
#             mask_negative = self.c_proj.weight.data >= 0
#             mask_positive = self.c_proj.weight.data < 0
#             c_proj_negative = self.c_proj.weight.data.clone()
#             c_proj_negative[mask_negative] = 0
#             self.c_proj_negative.weight = nn.Parameter(c_proj_negative)
#             c_proj_positive = self.c_proj.weight.data.clone()
#             c_proj_positive[mask_positive] = 0
#             self.c_proj_positive.weight = nn.Parameter(c_proj_positive)

        # 第二步：线性变换
        x_proj = self.c_proj(x_gelu)
        
        if infos_GPU is not None and infos_CPU is None:
            layer_info['GPU_inter_outputs'].append(x_proj.detach().clone())
            
        if infos_CPU is not None:
            x_proj_cpu = x_proj.detach().clone()
            layer = next((item for item in infos_GPU if item["layer_num"] == layer_counter[0]), None)
            x_proj_gpu = layer['GPU_inter_outputs'][2].to('cpu')
            x_gelu_gpu = layer['GPU_inter_outputs'][1].to('cpu')
            x_proj_2_cpu = self.c_proj(x_gelu_gpu)
            print("CPU/NPU proj ONLY error: ", (x_proj_gpu - x_proj_2_cpu).abs().max().item())
            print("CPU/NPU proj error: ", (x_proj_gpu - x_proj_cpu).abs().max().item())
        
        with torch.no_grad():
            # --- IBP for second Linear (c_proj) ---
            W = self.c_proj.weight
            b = self.c_proj.bias

            W_pos = torch.clamp(W, min=0)
            W_neg = torch.clamp(W, max=0)

            x_lower_proj = x_lower_act @ W_pos.T + x_upper_act @ W_neg.T - operator_noise
            x_upper_proj = x_upper_act @ W_pos.T + x_lower_act @ W_neg.T + operator_noise

            if b is not None:
                x_lower_proj += b
                x_upper_proj += b

        
#         with torch.no_grad():
#             x_lower_negative = self.c_proj_negative(x_lower)
#             x_lower_positive = self.c_proj_positive(x_lower)
#             x_upper_negative = self.c_proj_negative(x_upper)
#             x_upper_positive = self.c_proj_positive(x_upper)

#             x_lower = x_lower_positive + x_upper_negative - operator_noise
#             x_upper = x_upper_positive + x_lower_negative + operator_noise

        x = self.dropout(x_proj)
        # eval推理模式下dropout为恒等变换，不考虑误差
        # with torch.no_grad():
        #     x_lower = self.dropout(x_lower) - operator_noise
        #     x_upper = self.dropout(x_upper) + operator_noise
        
        # GPU/NPU模式下，只有输出值x有价值 old
        # layer_info['y'] = x
        # layer_info['y_lower'] = x_lower
        # layer_info['y_upper'] = x_upper
        
        layer_info['y'] = x
        layer_info['y_lower'] = x_lower_proj
        layer_info['y_upper'] = x_upper_proj
            
        # GPU/NPU模式下，只有输出值y有价值
        if infos_GPU is not None and infos_CPU is None:              
            infos_GPU.append(layer_info)
            print(f"Appending layer {layer_counter[0]} to infos_GPU, type = {layer_info['layer_type']}")
            ### test
            real_minus_lower = x_lower_proj - x
            real_minus_upper = x - x_upper_proj
            real_outside = torch.maximum(real_minus_lower, real_minus_upper)
            real_outside = torch.clamp(real_outside, min=0.0)  # 只保留超出的部分
            print("Output range width: ", (x_upper_proj - x_lower_proj).max().item())
            print("Real GPU output range error: ", real_outside.max().item())
            # exit()

        # CPU模式下，只有输出上下界有价值
        if infos_CPU is not None:
            infos_CPU.append(layer_info)
            print("----------- MLP finishes. --------")
            print(f"Appending layer {layer_counter[0]} to infos_CPU, type = {layer_info['layer_type']}")
            # layer = next((item for item in infos_GPU if item["layer_num"] == layer_counter[0]), None)
            # x_gpu = layer['x'].detach().to('cpu').clone()
            # y_gpu = layer['y'].detach().to('cpu').clone()
            # print("CPU/NPU input allignment error: ", (x_gpu - layer_info['x'].detach().to('cpu').clone()).abs().max().item())
            # print("CPU/NPU output error: ", (y_gpu - layer_info['y']).abs().max().item())
            # real_minus_lower = x_lower_proj - y_gpu
            # real_minus_upper = y_gpu - x_upper_proj
            # real_outside = torch.maximum(real_minus_lower, real_minus_upper)
            # real_outside = torch.clamp(real_outside, min=0.0)  # 只保留超出的部分
            # print("Output range width: ", (x_upper_proj - x_lower_proj).max().item())
            # print("NPU/GPU outside CPU range error: ", real_outside.max().item())
            # print("-----------------")
        
        if layer_counter is not None:
            layer_counter[0] = layer_counter[0] + 1

        return x, x_lower, x_upper

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, x_error=None, operator_noise=0, infos_GPU=None, infos_CPU=None, layer_counter=None):
        if x_error is None:
            x_lower = x.clone()
            x_upper = x.clone()
        else:
            x_lower = x - x_error
            x_upper = x + x_error  
            
        # layer_info = {
        #     'layer_num': None,
        #     'layer_type': 'Block',
        #     'x': x,
        #     'x_lower': x_lower,
        #     'x_upper': x_upper,
        #     'y': None,
        #     'y_lower': None,
        #     'y_upper': None,
        # }

        if infos_CPU is not None:
            # we need to first search the correct layer in GPU
            layer = next((item for item in infos_GPU if item["layer_num"] == layer_counter[0]), None)
            x, x_lower, x_upper = self.ln_1(layer['x'], x_error, operator_noise, infos_GPU, infos_CPU, layer_counter)
            ## test
            # y_real = layer['y'].detach().to('cpu').clone()
            # y_pred = infos_CPU[-1]['y'].detach().to('cpu').clone()
            # y_cpu_lower = infos_CPU[-1]['y_lower'].detach().to('cpu').clone()
            # y_cpu_upper = infos_CPU[-1]['y_upper'].detach().to('cpu').clone()
            # print("CPU/NPU output error: ", (y_real - y_pred).abs().max().item())
            # real_minus_lower = y_cpu_lower - y_real
            # real_minus_upper = y_real - y_cpu_upper
            # real_outside = torch.maximum(real_minus_lower, real_minus_upper)
            # real_outside = torch.clamp(real_outside, min=0.0)  # 只保留超出的部分
            # print("NPU outside CPU range error: ", real_outside.max().item())
            # bound_width = (y_cpu_upper - y_cpu_lower).max().item()
            # print("CPU Interval width max: ", bound_width)
        else:
            x, x_lower, x_upper = self.ln_1(x, x_error, operator_noise, infos_GPU, infos_CPU=None, layer_counter=layer_counter)
        
        if infos_CPU is not None:
            # we need to first search the correct layer in GPU
            layer = next((item for item in infos_GPU if item["layer_num"] == layer_counter[0]), None)
            attn_x, attn_x_lower, attn_x_upper = self.attn(layer['x'], x_error, operator_noise, infos_GPU, infos_CPU, layer_counter)
            ### test
            # y_real = layer['y'].detach().to('cpu').clone()
            # y_pred = infos_CPU[-1]['y'].detach().to('cpu').clone()
            # y_cpu_lower = infos_CPU[-1]['y_lower'].detach().to('cpu').clone()
            # y_cpu_upper = infos_CPU[-1]['y_upper'].detach().to('cpu').clone()
            # print("CPU/NPU output error: ", (y_real - y_pred).abs().max().item())
            # real_minus_lower = y_cpu_lower - y_real
            # real_minus_upper = y_real - y_cpu_upper
            # real_outside = torch.maximum(real_minus_lower, real_minus_upper)
            # real_outside = torch.clamp(real_outside, min=0.0)  # 只保留超出的部分
            # print("NPU outside CPU range error: ", real_outside.max().item())
            # bound_width = (y_cpu_upper - y_cpu_lower).max().item()
            # print("CPU Interval width max: ", bound_width)
        else:
            attn_x, attn_x_lower, attn_x_upper = self.attn(x, x_error, operator_noise, infos_GPU, infos_CPU=None, layer_counter=layer_counter)
        
        if x.device != attn_x.device:
            x = x.to(attn_x.device)
        if x_lower.device != attn_x_lower.device:
            x_lower = x_lower.to(attn_x_lower.device)
        if x_upper.device != attn_x_upper.device:
            x_upper = x_upper.to(attn_x_lower.device)
        x = x + attn_x
        
        with torch.no_grad():
            x_lower = x_lower + attn_x_lower
            x_upper = x_upper + attn_x_upper

        if infos_CPU is not None:
            # we need to first search the correct layer in GPU
            layer = next((item for item in infos_GPU if item["layer_num"] == layer_counter[0]), None)
            x, x_lower, x_upper = self.ln_2(layer['x'], x_error, operator_noise, infos_GPU, infos_CPU, layer_counter)
            ## test
            # y_real = layer['y'].detach().to('cpu').clone()
            # y_pred = infos_CPU[-1]['y'].detach().to('cpu').clone()
            # y_cpu_lower = infos_CPU[-1]['y_lower'].detach().to('cpu').clone()
            # y_cpu_upper = infos_CPU[-1]['y_upper'].detach().to('cpu').clone()
            # print("CPU/NPU output error: ", (y_real - y_pred).abs().max().item())
            # real_minus_lower = y_cpu_lower - y_real
            # real_minus_upper = y_real - y_cpu_upper
            # real_outside = torch.maximum(real_minus_lower, real_minus_upper)
            # real_outside = torch.clamp(real_outside, min=0.0)  # 只保留超出的部分
            # print("NPU outside range error: ", real_outside.max().item())
            # bound_width = (y_cpu_upper - y_cpu_lower).max().item()
            # print("CPU Interval width max: ", bound_width)
        else:    
            x, x_lower, x_upper = self.ln_2(x, x_error, operator_noise, infos_GPU, infos_CPU=None, layer_counter=layer_counter)
            
        if infos_CPU is not None:
            # we need to first search the correct layer in GPU
            layer = next((item for item in infos_GPU if item["layer_num"] == layer_counter[0]), None)
            mlp_x, mlp_x_lower, mlp_x_upper = self.mlp(layer['x'], x_error, operator_noise, infos_GPU, infos_CPU, layer_counter)
            # ## test
            # y_real = layer['y'].detach().to('cpu').clone()
            # y_pred = infos_CPU[-1]['y'].detach().to('cpu').clone()
            # y_cpu_lower = infos_CPU[-1]['y_lower'].detach().to('cpu').clone()
            # y_cpu_upper = infos_CPU[-1]['y_upper'].detach().to('cpu').clone()
            # print("In Block-CPU/NPU output error: ", (y_real - y_pred).abs().max().item())
            # real_minus_lower = y_cpu_lower - y_real
            # real_minus_upper = y_real - y_cpu_upper
            # real_outside = torch.maximum(real_minus_lower, real_minus_upper)
            # real_outside = torch.clamp(real_outside, min=0.0)  # 只保留超出的部分
            # print("In Block-NPU outside range error: ", real_outside.max().item())
            # bound_width = (y_cpu_upper - y_cpu_lower).max().item()
            # print("In Block-CPU Interval width max: ", bound_width)
            # exit()
        else:
            mlp_x, mlp_x_lower, mlp_x_upper = self.mlp(x, x_error, operator_noise, infos_GPU, infos_CPU=None, layer_counter=layer_counter)
        
        if x.device != mlp_x.device:
            x = x.to(mlp_x.device)
        x = x + mlp_x

        if x_lower.device != mlp_x_lower.device:
            x_lower = x_lower.to(mlp_x_lower.device)
        if x_upper.device != mlp_x_upper.device:
            x_upper = x_upper.to(mlp_x_upper.device)
        with torch.no_grad():
            x_lower = x_lower + mlp_x_lower
            x_upper = x_upper + mlp_x_upper
        
#         layer_info['layer_num'] = layer_counter[0]
#         layer_info['y'] = x
#         layer_info['y_lower'] = x_lower
#         layer_info['y_upper'] = x_upper
            
#         # GPU/NPU模式下，只有输出值y有价值
#         if infos_GPU is not None and infos_CPU is None:     
#             infos_GPU.append(layer_info)
#             print(f"Appending layer {layer_counter[0]} to infos_GPU, type = {layer_info['layer_type']}")
#             ### test
#             # real_minus_lower = x_lower - x
#             # real_minus_upper = x - x_upper
#             # real_outside = torch.maximum(real_minus_lower, real_minus_upper)
#             # real_outside = torch.clamp(real_outside, min=0.0)  # 只保留超出的部分
#             # print("Output range width: ", (x_upper - x_lower).max().item())
#             # print("Real GPU output range error: ", real_outside.max().item())

#         # CPU模式下，只有输出上下界有价值
#         if infos_CPU is not None:
#             infos_CPU.append(layer_info)
#             print(f"Appending layer {layer_counter[0]} to infos_CPU, type = {layer_info['layer_type']}")
            
#         layer_counter[0] = layer_counter[0] + 1
            
        return x, x_lower, x_upper

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, robust_veri=None, input_error=1e-5, infos_GPU=None, noising_input=False, operator_noise=0):
        infos_CPU = []
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        if noising_input:
            x = x + (torch.rand_like(x).to(device) * (input_error_upper - input_error_lower) + input_error_lower)
        with torch.no_grad():
            x_lower = x - input_error
            x_upper = x + input_error
        # record input
        ######### test
#         if infos_GPU is not None:
#             diff = (x - infos_GPU['x'][0].cpu()).abs()
#             print(f"Max diff after .to('cpu'): {diff.max().item():.2e}")
#             print(f"Mean diff after .to('cpu'): {diff.mean().item():.2e}")

#             # 额外：确认是否所有元素都完全一致
#             all_close = torch.allclose(x, infos_GPU['x'][0].cpu(), atol=1e-7)
#             print(f"All elements close within 1e-7? {'✅' if all_close else '❌'}")
#             exit()
        #########
        

        x_GPU = []
        # robust_veri can be Off, CPU, GPU
        if robust_veri == 'CPU' and infos_GPU is not None:
            # get GPU x info
            x_GPU = [layer['x'].detach().to("cpu") for layer in infos_GPU]
        layer_counter = [0]
        if infos_GPU is not None:
            for block in self.transformer.h:
                if robust_veri == 'CPU':
                    if layer_counter[0] >= len(x_GPU):
                        raise ValueError(f"layer_counter {layer_counter[0]} out of x_GPU range {len(x_GPU)}")
                    x_GPU_input = x_GPU[layer_counter[0]]
                    x, x_lower, x_upper = block(x_GPU_input, input_error, operator_noise, infos_GPU, infos_CPU, layer_counter)
                else:
                    x, x_lower, x_upper = block(x, input_error, operator_noise, infos_GPU, infos_CPU=None, layer_counter=layer_counter)

        if infos_GPU is not None:
            if robust_veri == 'CPU':
                x_GPU_input = x_GPU[layer_counter[0]]
                y_real = infos_GPU[layer_counter[0]]['y'].detach().to('cpu').clone()
                
                x, x_lower, x_upper = self.transformer.ln_f(x_GPU_input, input_error, operator_noise, infos_GPU, infos_CPU, layer_counter)
                
                y_pred = infos_CPU[-1]['y'].detach().to('cpu').clone()
                y_cpu_lower = infos_CPU[-1]['y_lower'].detach().to('cpu').clone()
                y_cpu_upper = infos_CPU[-1]['y_upper'].detach().to('cpu').clone()
                print("CPU/NPU output error: ", (y_real - y_pred).abs().max().item())
                real_minus_lower = y_cpu_lower - y_real
                real_minus_upper = y_real - y_cpu_upper
                real_outside = torch.maximum(real_minus_lower, real_minus_upper)
                real_outside = torch.clamp(real_outside, min=0.0)  # 只保留超出的部分
                print("NPU outside range error: ", real_outside.max().item())
                bound_width = (y_cpu_upper - y_cpu_lower).max().item()
                print("CPU Interval width max: ", bound_width)
            else:              
                x, x_lower, x_upper = self.transformer.ln_f(x, input_error, operator_noise, infos_GPU, infos_CPU=None, layer_counter=layer_counter)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits += (torch.rand_like(logits).to(device) * 2 * input_error - input_error)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits += (torch.rand_like(logits).to(device) * 2 * input_error - input_error)
            loss = None

        return logits, loss, infos_CPU

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        # use_fused = fused_available and device_type == 'cuda'
        use_fused = False
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
