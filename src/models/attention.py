import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionLayer(nn.Module):
    """
    Attention mechanism for multi-agent coordination.
    Implements scaled dot-product attention as used in ATT-MADDPG.
    """
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Linear transformations for Q, K, V
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        """
        Forward pass of attention layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations
        Q = self.query(x)  # (batch_size, seq_len, hidden_dim)
        K = self.key(x)    # (batch_size, seq_len, hidden_dim)
        V = self.value(x)  # (batch_size, seq_len, hidden_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Reshape back to original dimensions
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        
        # Output projection
        output = self.out_proj(attended)
        
        # Residual connection and layer normalization
        output = self.layer_norm(output + x[:, :, :self.hidden_dim])
        
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module for processing multi-agent observations.
    """
    def __init__(self, obs_dim, hidden_dim, num_heads=4, num_layers=2):
        super(MultiHeadAttention, self).__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        
        # Stack of attention layers
        self.attention_layers = nn.ModuleList([
            AttentionLayer(hidden_dim, hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, obs_dim)
        
    def forward(self, observations, mask=None):
        """
        Process multi-agent observations through attention layers.
        
        Args:
            observations: Tensor of shape (batch_size, num_agents, obs_dim)
            mask: Optional mask tensor
        
        Returns:
            Processed observations and attention weights
        """
        batch_size, num_agents, obs_dim = observations.size()
        
        # Input projection
        x = self.input_proj(observations)
        
        # Apply attention layers
        attention_weights = []
        for layer in self.attention_layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        
        # Output projection
        output = self.output_proj(x)
        
        return output, attention_weights

class AttentionMADDPGEncoder(nn.Module):
    """
    Encoder that processes multi-agent observations using attention mechanism.
    """
    def __init__(self, obs_dim, hidden_dim=128, num_heads=4, num_layers=2):
        super(AttentionMADDPGEncoder, self).__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        
        # Multi-head attention
        self.attention = MultiHeadAttention(obs_dim, hidden_dim, num_heads, num_layers)
        
        # Additional processing layers
        self.processor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, observations, mask=None):
        """
        Encode multi-agent observations with attention.
        
        Args:
            observations: Tensor of shape (batch_size, num_agents, obs_dim)
            mask: Optional mask tensor
        
        Returns:
            Encoded observations and attention weights
        """
        # Apply attention
        attended_obs, attention_weights = self.attention(observations, mask)
        
        # Process attended observations
        processed_obs = self.processor(attended_obs)
        
        return processed_obs, attention_weights
