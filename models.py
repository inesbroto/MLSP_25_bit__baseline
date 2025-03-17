import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple1DCNN(nn.Module):
    def __init__(self, num_sensors=312, num_classes=3, dropout_prob=0):
        super(Simple1DCNN, self).__init__()
        
        self.features = nn.Sequential(
            self._conv_block(num_sensors, 64, 3),
            nn.MaxPool1d(2),
            nn.Dropout1d(p=dropout_prob),
            self._conv_block(64, 128, 5),
            nn.MaxPool1d(2),
            nn.Dropout1d(p=dropout_prob),
            self._conv_block(128, 256, 7),
            nn.MaxPool1d(2),
            nn.Dropout1d(p=dropout_prob),
            self._conv_block(256, 512, 9),
            nn.AdaptiveMaxPool1d(1),  # Global max pooling
            nn.Dropout1d(p=dropout_prob)
        )
        
        self.classifier = nn.Sequential(
            nn.Conv1d(512, 128, 1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Conv1d(128, num_classes, 1)
        )

    def _conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, xc):

        # the control chunk its not used by this simple model
        # (you can try to implement some models that take advantage of it)

        # Input shape: [batch, 1, time, sensors]
        x = x.squeeze(1).permute(0, 2, 1)  # → [batch, sensors, time]
        
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(-1)  # Remove the last dimension of size 1




if __name__ == "__main__":
 
    X = torch.randn(32, 1, 300, 312)  # [batch_size, channels, t_steps, sensors]
    Xc = torch.randn(32, 1, 300, 312)

    model = Simple1DCNN(num_sensors=X.shape[3], num_classes=3)
    output = model(X, Xc)

    print(f"Input shape 1: {X.shape}, Output shape 1: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")


























#     class PositionalEncoding(nn.Module):

#     def __init__(self, d_model: int, max_len: int = 300):
#         super().__init__()

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, d_model)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: Tensor, shape [batch_size, seq_len, d_model]
#         """
#         x = x + self.pe[:x.size(1)]
#         return x
    
# def scaled_dot_product_attention(q, k, v, mask=None):
#     d_k = q.size(-1)
#     attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
#     if mask is not None:
#         attn_logits = attn_logits.masked_fill(mask == 0, -1e9)
#     attn_weights = F.softmax(attn_logits, dim=-1)
#     output = torch.matmul(attn_weights, v)
#     return output, attn_weights

# class MultiHeadAttention(nn.Module):

#     def __init__(self, d_model, num_heads):
#         super().__init__()
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.head_dim = d_model // num_heads
        
#         self.q_linear = nn.Linear(d_model, d_model)
#         self.k_linear = nn.Linear(d_model, d_model)
#         self.v_linear = nn.Linear(d_model, d_model)
#         self.out_linear = nn.Linear(d_model, d_model)
        
#     def forward(self, q, k, v, mask=None):
#         batch_size = q.size(0)
        
#         # Linear projections and split into heads
#         q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
#         # Compute attention
#         scores, attn_weights = scaled_dot_product_attention(q, k, v, mask)
        
#         # Concatenate heads and apply final linear
#         scores = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
#         output = self.out_linear(scores)
#         return output
    

# class TransformerEncoderLayer(nn.Module):
#     def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
#         super().__init__()
#         self.self_attn = MultiHeadAttention(d_model, num_heads)
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
        
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
        
#         self.activation = F.relu
        
#     def forward(self, src):
#         # Self attention
#         src2 = self.self_attn(src, src, src)
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
        
#         # Feed forward
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src
    
# class SensorTransformer(nn.Module):

#     def __init__(self, num_sensors=306, d_model=512, num_heads=8, num_layers=6, num_classes=3):
#         super().__init__()
        
#         # Input embedding
#         self.input_proj = nn.Linear(num_sensors, d_model)
#         self.pos_encoder = PositionalEncoding(d_model)
        
#         # Encoder layers
#         self.layers = nn.ModuleList([
#             TransformerEncoderLayer(d_model, num_heads)
#             for _ in range(num_layers)
#         ])
        
#         # Classifier
#         self.classifier = nn.Sequential(
#             nn.Linear(d_model, d_model//2),
#             nn.ReLU(),
#             nn.Linear(d_model//2, num_classes)
#         )
        
#         # Initialize weights
#         self._init_weights()
        
#     def _init_weights(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
                
#     def forward(self, x,xc):
#         # Input shape: [batch_size, 1, 300, 306]
#         x = x.squeeze(1)  # Remove channel dim -> [batch_size, 300, 306]
        
#         # Project input
#         x = self.input_proj(x)  # [batch_size, 300, d_model]
        
#         # Add positional encoding
#         x = self.pos_encoder(x)
        
#         # Transformer layers
#         for layer in self.layers:
#             x = layer(x)
            
#         # Aggregate temporal dimension
#         x = x.mean(dim=1)  # [batch_size, d_model]
        
#         # Classify
#         logits = self.classifier(x)
#         return logits