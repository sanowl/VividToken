import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PaligemmaModel
from typing import Dict, List, Tuple, Optional
import math

class LoRALinear(nn.Module):
    """
    Corrected implementation of LoRA for linear layers
    """
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.original_layer = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(p=dropout)
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.original_layer.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original layer
        original_output = self.original_layer(x)
        
        # LoRA path
        lora_output = (
            self.dropout(x) @ 
            self.lora_A @ 
            self.lora_B
        ) * self.scaling
        
        return original_output + lora_output

class TransformerLoRAWrapper(nn.Module):
    """
    Wrapper to apply LoRA to transformer layers
    """
    def __init__(self, transformer_layer, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base_layer = transformer_layer
        
        # Apply LoRA to attention layers
        if hasattr(transformer_layer, 'self_attn'):
            self.q_lora = LoRALinear(
                transformer_layer.self_attn.q_proj.in_features,
                transformer_layer.self_attn.q_proj.out_features,
                rank, alpha, dropout
            )
            self.k_lora = LoRALinear(
                transformer_layer.self_attn.k_proj.in_features,
                transformer_layer.self_attn.k_proj.out_features,
                rank, alpha, dropout
            )
            self.v_lora = LoRALinear(
                transformer_layer.self_attn.v_proj.in_features,
                transformer_layer.self_attn.v_proj.out_features,
                rank, alpha, dropout
            )
            
            # Replace attention projections with LoRA versions
            self.base_layer.self_attn.q_proj = self.q_lora
            self.base_layer.self_attn.k_proj = self.k_lora
            self.base_layer.self_attn.v_proj = self.v_lora

    def forward(self, *args, **kwargs):
        return self.base_layer(*args, **kwargs)

class CorrectedTLDRModel(nn.Module):
    """
    Corrected Token-Level Detective Reward Model implementation
    """
    def __init__(self, config: Dict = None):
        super().__init__()
        if config is None:
            config = {
                'hidden_dim': 2048,
                'lora_rank': 512,
                'lora_alpha_train': 128,
                'lora_dropout': 0.1,
                'image_resolution': 448,
                'num_image_tokens': 1024
            }
        
        # Initialize PaliGemma backbone
        self.backbone = PaligemmaModel.from_pretrained('google/paligemma-3b-mix-448')
        
        # Freeze vision encoder
        for param in self.backbone.vision_encoder.parameters():
            param.requires_grad = False
            
        # Apply LoRA to projection layer
        self.proj_lora = LoRALinear(
            self.backbone.proj_layer.in_features,
            self.backbone.proj_layer.out_features,
            config['lora_rank'],
            config['lora_alpha_train'],
            config['lora_dropout']
        )
        
        # Apply LoRA to decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerLoRAWrapper(
                layer,
                config['lora_rank'],
                config['lora_alpha_train'],
                config['lora_dropout']
            ) for layer in self.backbone.decoder.layers
        ])
        
        # Reward head
        self.reward_head = nn.Linear(config['hidden_dim'], 1)

    def forward(
        self,
        image: torch.Tensor,
        prompt: torch.Tensor,
        response: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Vision features
        vision_features = self.backbone.vision_encoder(image)
        
        # Project with LoRA
        projected_features = self.proj_lora(vision_features)
        
        # Prepare inputs
        inputs_embeds = torch.cat([
            projected_features,
            self.backbone.decoder.embed_tokens(prompt),
            self.backbone.decoder.embed_tokens(response)
        ], dim=1)
        
        # Process through decoder layers
        hidden_states = inputs_embeds
        for layer in self.decoder_layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask
            )
        
        # Get response token rewards
        response_length = response.size(1)
        response_hidden = hidden_states[:, -response_length:]
        token_rewards = torch.sigmoid(self.reward_head(response_hidden))
        
        return token_rewards

def compute_flexible_sentence_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    sentence_breaks: List[List[int]],
    threshold: float = 0.5,
    error_tolerance: float = 0.2
) -> float:
    """
    Improved sentence-level accuracy with error tolerance
    """
    binary_preds = (predictions > threshold).float()
    sentence_accuracies = []
    
    for i, breaks in enumerate(sentence_breaks):
        for start, end in zip(breaks[:-1], breaks[1:]):
            sent_pred = binary_preds[i, start:end]
            sent_label = labels[i, start:end]
            
            # Calculate accuracy with tolerance
            correct_ratio = (sent_pred == sent_label).float().mean()
            sent_acc = (correct_ratio >= (1 - error_tolerance)).float()
            sentence_accuracies.append(sent_acc)
            
    return torch.tensor(sentence_accuracies).mean().item()