import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        assert self.d_k * num_heads == d_model, "d_model must be divisible by num_heads"
        
        # 定义线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None, use_cache=False, cache=None):
        batch_size = q.size(0)
        
        # 线性变换并分割多头
        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 处理KV缓存
        if use_cache:
            if cache is not None:
                K = torch.cat([cache['k'], K], dim=2)
                V = torch.cat([cache['v'], V], dim=2)
                cache['k'] = K
                cache['v'] = V
            else:
                cache = {'k': K, 'v': V}
        
        # 计算注意力得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重到V
        output = torch.matmul(attn_weights, V)
        
        # 合并多头并线性变换
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        return output, attn_weights, cache

# 示例：计算随机矩阵的注意力权重
d_model = 512
num_heads = 8
batch_size = 2
seq_len = 5

# 初始化多头注意力
attn = MultiHeadAttention(d_model, num_heads)

# 生成随机输入（自注意力）
q = k = v = torch.randn(batch_size, seq_len, d_model)

# 前向传播
output, attn_weights, _ = attn(q, k, v)

print("输出张量形状:", output.shape)        # (2, 5, 512)
print("注意力权重形状:", attn_weights.shape) # (2, 8, 5, 5)

# KV缓存示例
cache = None
q_step = torch.randn(batch_size, 1, d_model)
output_step, attn_weights_step, cache = attn(q_step, q_step, q_step, use_cache=True, cache=cache)
print("\n带缓存的K形状:", cache['k'].shape) # (2, 8, 1, 64)