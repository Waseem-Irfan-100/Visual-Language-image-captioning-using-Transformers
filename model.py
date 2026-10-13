import torch
import torch.nn as nn
from transformers import ViTModel
import math
from config import Config

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class ViTEncoder(nn.Module):
    def __init__(self, encoder_name=Config.ENCODER_NAME):
        super().__init__()
        self.vit = ViTModel.from_pretrained(encoder_name)
        for param in self.vit.parameters():
            param.requires_grad = False
            
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.last_hidden_state

# ---------------------------------------------------------
# CUSTOM DECODER LAYER (No PyTorch hacks needed!)
# ---------------------------------------------------------
class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Self-Attention (Text-to-Text)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Cross-Attention (Text-to-Image)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        # 1. Masked Self-Attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # 2. Cross-Attention
        # By setting average_attn_weights=False, we safely extract the attention map for visualization!
        tgt2, cross_attn_weights = self.multihead_attn(
            tgt, memory, memory, 
            key_padding_mask=None,
            average_attn_weights=False
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # 3. Feed Forward
        tgt2 = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt, cross_attn_weights

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, dropout, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        
        self.layers = nn.ModuleList([
            CustomTransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout) 
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        tgt = self.embedding(tgt)
        tgt = self.pos_encoder(tgt)
        
        all_cross_attentions = []
        
        for layer in self.layers:
            tgt, attn_weights = layer(
                tgt=tgt, memory=memory, 
                tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask
            )
            all_cross_attentions.append(attn_weights)
            
        output = self.fc_out(tgt)
        
        # We return the output and the attention weights from the final layer
        return output, all_cross_attentions[-1]

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = ViTEncoder()
        
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=Config.EMBED_DIM,
            num_layers=Config.DECODER_LAYERS,
            num_heads=Config.DECODER_HEADS,
            ff_dim=Config.DECODER_FF_DIM,
            dropout=Config.DROPOUT,
            max_len=Config.MAX_CAPTION_LEN
        )
        
    def forward(self, pixel_values, input_ids, tgt_mask=None, tgt_padding_mask=None):
        # Used during training (returns just outputs)
        memory = self.encoder(pixel_values)
        outputs, _ = self.decoder(
            tgt=input_ids, 
            memory=memory, 
            tgt_mask=tgt_mask, 
            tgt_key_padding_mask=tgt_padding_mask
        )
        return outputs
        
    def forward_with_attention(self, pixel_values, input_ids, tgt_mask=None):
        # Used during Inference for Heatmaps
        memory = self.encoder(pixel_values)
        outputs, attn_weights = self.decoder(
            tgt=input_ids, memory=memory, tgt_mask=tgt_mask
        )
        return outputs, attn_weights
