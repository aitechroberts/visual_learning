import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class StandardSelfAttention(nn.Module):
    """
    Standard single-head self-attention layer for Vision Transformer
    Uses scaled dot-product attention as shown in the formula
    """
    def __init__(self, embed_dim, dropout=0.1):
        super(StandardSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim  # Single head, so head_dim = embed_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
               For ViT: (batch_size, num_patches + 1, embed_dim)
        Returns:
            output: Attention output of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        Q = self.W_q(x)  # (batch_size, seq_len, embed_dim)
        K = self.W_k(x)  # (batch_size, seq_len, embed_dim)
        V = self.W_v(x)  # (batch_size, seq_len, embed_dim)
        
        # Compute attention scores using dot product
        # scores = Q @ K^T / sqrt(d_h)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        
        return output


class L2SelfAttention(nn.Module):
    """
    L2 distance-based single-head self-attention layer
    Uses L2 distance instead of dot product for computing attention scores
    """
    def __init__(self, embed_dim, dropout=0.1, temperature=1.0):
        super(L2SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim
        self.temperature = temperature  # Temperature parameter for scaling
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            output: Attention output of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        Q = self.W_q(x)  # (batch_size, seq_len, embed_dim)
        K = self.W_k(x)  # (batch_size, seq_len, embed_dim)
        V = self.W_v(x)  # (batch_size, seq_len, embed_dim)
        
        # Compute L2 distance-based attention scores
        # For each query, compute L2 distance to all keys
        # ||Q_i - K_j||^2 = ||Q_i||^2 + ||K_j||^2 - 2 * Q_i · K_j
        
        # Compute squared norms
        Q_norm_sq = torch.sum(Q**2, dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        K_norm_sq = torch.sum(K**2, dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        
        # Compute dot products
        QK_dot = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, seq_len, seq_len)
        
        # Compute L2 distances squared: ||Q_i - K_j||^2
        # Broadcasting: Q_norm_sq is (batch, seq_len, 1), K_norm_sq.transpose is (batch, 1, seq_len)
        l2_distances_sq = Q_norm_sq + K_norm_sq.transpose(-2, -1) - 2 * QK_dot
        
        # Convert distances to similarities (negative distance scaled by temperature)
        # We use negative distance so that smaller distances give higher attention
        scores = -l2_distances_sq / self.temperature
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        
        return output


class VisionTransformerBlock(nn.Module):
    """
    Complete Vision Transformer block with either standard or L2 attention
    """
    def __init__(self, embed_dim, attention_type='standard', dropout=0.1, mlp_ratio=4):
        super(VisionTransformerBlock, self).__init__()
        
        # Choose attention mechanism
        if attention_type == 'standard':
            self.attention = StandardSelfAttention(embed_dim, dropout)
        elif attention_type == 'l2':
            self.attention = L2SelfAttention(embed_dim, dropout)
        else:
            raise ValueError("attention_type must be 'standard' or 'l2'")
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP block
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Attention with residual connection and layer norm
        x = x + self.attention(self.norm1(x))
        
        # MLP with residual connection and layer norm
        x = x + self.mlp(self.norm2(x))
        
        return x
"""
Standard Self-Attention

Uses the formula from your image: Attention(X) = softmax(QK^T/√d_h)V
Computes dot products between queries and keys
Scaled by 1/√d_h for numerical stability

L2 Self-Attention

Uses L2 distance instead of dot product: ||Q_i - K_j||²
Converts distances to similarities using negative distance
Includes a temperature parameter for scaling

Key Implementation Details:

L2 Distance Computation: I use the mathematical identity ||a-b||² = ||a||² + ||b||² - 2a·b for efficient computation
Similarity Conversion: Negative L2 distances are used so smaller distances yield higher attention weights
Complete ViT Blocks: Both attention mechanisms are wrapped in full transformer blocks with layer normalization and MLP

Usage:
python# Standard attention
std_attention = StandardSelfAttention(embed_dim=768)

# L2 attention
l2_attention = L2SelfAttention(embed_dim=768, temperature=1.0)

# Complete ViT blocks
std_block = VisionTransformerBlock(embed_dim=768, attention_type='standard')
l2_block = VisionTransformerBlock(embed_dim=768, attention_type='l2')
The L2 attention mechanism can potentially capture different similarity patterns compared to dot-product attention,
as it's based on Euclidean distance rather than cosine similarity (which dot-product approximates when normalized).
"""

# Example usage and comparison
if __name__ == "__main__":
    # Parameters
    batch_size = 2
    seq_len = 197  # 196 patches + 1 CLS token for ViT-Base
    embed_dim = 768
    
    # Create dummy input (representing image patches + CLS token)
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    print("Input shape:", x.shape)
    print()
    
    # Test Standard Attention
    print("=== Standard Dot-Product Attention ===")
    std_attention = StandardSelfAttention(embed_dim)
    std_output = std_attention(x)
    print("Output shape:", std_output.shape)
    print("Output mean:", std_output.mean().item())
    print("Output std:", std_output.std().item())
    print()
    
    # Test L2 Attention
    print("=== L2 Distance-Based Attention ===")
    l2_attention = L2SelfAttention(embed_dim)
    l2_output = l2_attention(x)
    print("Output shape:", l2_output.shape)
    print("Output mean:", l2_output.mean().item())
    print("Output std:", l2_output.std().item())
    print()
    
    # Test complete blocks
    print("=== Complete ViT Blocks ===")
    std_block = VisionTransformerBlock(embed_dim, 'standard')
    l2_block = VisionTransformerBlock(embed_dim, 'l2')
    
    std_block_output = std_block(x)
    l2_block_output = l2_block(x)
    
    print("Standard ViT block output shape:", std_block_output.shape)
    print("L2 ViT block output shape:", l2_block_output.shape)

