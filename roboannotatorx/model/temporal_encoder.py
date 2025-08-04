# import torch
# import torch.nn as nn
# import math

# class RotaryEmbedding(nn.Module):
#     """Rotary Position Embedding"""
#     def __init__(self, dim):
#         super().__init__()
#         inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
#         self.register_buffer('inv_freq', inv_freq)
        
#     def forward(self, seq_len: int, device: torch.device):
#         t = torch.arange(seq_len, device=device)
#         freqs = torch.einsum('i,j->ij', t, self.inv_freq)
#         emb = torch.cat((freqs, freqs), dim=-1)
#         cos = emb.cos()
#         sin = emb.sin()
#         return cos, sin

# def rotate_half(x):
#     x1, x2 = x.chunk(2, dim=-1)
#     return torch.cat((-x2, x1), dim=-1)

# def apply_rotary_pos_emb(q, k, cos, sin):
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed

# class MotionEncoder(nn.Module):
#     def __init__(self, hidden_size, nhead=8, num_layers=2, dropout=0.1):
#         super().__init__()
#         assert hidden_size % 2 == 0, "hidden_size must be even for RoPE"
        
#         self.hidden_size = hidden_size
#         self.nhead = nhead
#         self.dropout = nn.Dropout(p=dropout)
        
#         # RoPE embedding
#         self.rope = RotaryEmbedding(hidden_size // nhead)  # head_dim
        
#         # 投影矩阵
#         self.q_proj = nn.Linear(hidden_size, hidden_size)
#         self.k_proj = nn.Linear(hidden_size, hidden_size)
#         self.v_proj = nn.Linear(hidden_size, hidden_size)
#         self.o_proj = nn.Linear(hidden_size, hidden_size)
        
#         # Transformer encoder layer
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_size,
#             nhead=nhead,
#             dim_feedforward=4*hidden_size,
#             dropout=dropout,
#             batch_first=False
#         )
        
#         self.transformer_encoder = nn.TransformerEncoder(
#             encoder_layer=encoder_layer,
#             num_layers=num_layers
#         )
    
#     def _reshape_for_attention(self, x):
#         seq_len, batch_size, hidden_size = x.shape
#         head_dim = hidden_size // self.nhead
#         return x.view(seq_len, batch_size * self.nhead, head_dim)
    
#     def forward(self, x, src_mask=None, src_padding_mask=None):
#         seq_len, batch_size, _ = x.shape
#         device = x.device
        
#         # Generate RoPE encodings
#         cos, sin = self.rope(seq_len, device)
        
#         # Project to Q, K, V
#         q = self.q_proj(x)
#         k = self.k_proj(x)
#         v = self.v_proj(x)
        
#         # Reshape for multi-head attention
#         q = self._reshape_for_attention(q)
#         k = self._reshape_for_attention(k)
#         v = self._reshape_for_attention(v)
        
#         # Apply RoPE
#         cos = cos.unsqueeze(1).expand(-1, batch_size * self.nhead, -1)
#         sin = sin.unsqueeze(1).expand(-1, batch_size * self.nhead, -1)
#         q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
#         # Reshape back
#         q = q.view(seq_len, batch_size, self.hidden_size)
#         k = k.view(seq_len, batch_size, self.hidden_size)
#         v = v.view(seq_len, batch_size, self.hidden_size)
        
#         # Apply transformer encoder
#         output = self.transformer_encoder(q, mask=src_mask, src_key_padding_mask=src_padding_mask)
        
#         return output

# def get_motion_encoder_state(model_params, keys_to_match=None):
#     """获取Motion Encoder相关的参数"""
#     if keys_to_match is None:
#         # 由于使用RoPE，需要包含更多参数
#         keys_to_match = [
#             'motion_encoder.transformer_encoder',  # transformer层参数
#             'motion_encoder.rope',                # RoPE相关参数和buffer
#             'motion_encoder.q_proj',             # Q投影层
#             'motion_encoder.k_proj',             # K投影层
#             'motion_encoder.v_proj',             # V投影层
#             'motion_encoder.o_proj',             # 输出投影层
#         ]
    
#     motion_state_dict = {}
    
#     # 遍历命名参数
#     for name, param in model_params:
#         if any(key in name for key in keys_to_match):
#             motion_state_dict[name] = param
    
#     return motion_state_dict

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding"""
    def __init__(self, hidden_size, max_seq_length=5000):
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_length, hidden_size)
        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [seq_len, 1, hidden_size]
        
        # 注册为buffer（不作为模型参数）
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: 输入张量 [seq_len, batch_size, hidden_size]
        """
        return x + self.pe[:x.size(0)]

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.pos_encoder = PositionalEncoding(hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建标准的Transformer Encoder层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=4*hidden_size,
            dropout=dropout,
            batch_first=False  # 保持 [seq_len, batch, hidden] 的输入格式
        )
        
        # 创建完整的Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
    
    def forward(self, x, src_mask=None, src_padding_mask=None):
        """
        Args:
            x: 输入序列 [seq_len, batch_size, hidden_size]
            src_mask: 源序列的attention mask
            src_padding_mask: padding mask
        Returns:
            output: 输出序列 [seq_len, batch_size, hidden_size]
        """
        # 添加位置编码
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # 通过transformer encoder
        output = self.transformer_encoder(x, mask=src_mask, 
                                       src_key_padding_mask=src_padding_mask)
        
        return output

def get_transformer_state(model_params, keys_to_match=None):
    """获取Transformer相关的参数"""
    if keys_to_match is None:
        keys_to_match = [
            'transformer_encoder',  # transformer层参数
            'pos_encoder',         # 位置编码参数
        ]
    
    transformer_state_dict = {}
    
    # 遍历命名参数
    for name, param in model_params:
        if any(key in name for key in keys_to_match):
            transformer_state_dict[name] = param
    
    return transformer_state_dict