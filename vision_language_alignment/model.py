#!/usr/bin/env python3
"""
Vision-Language Model for LLaMA 4 alignment training.
Combines pretrained ViT encoder with pretrained LLM.
"""

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

# Project paths (resolve to absolute paths)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEXT_PRETRAIN_DIR = PROJECT_ROOT / "text_pretraining"
DEFAULT_LLM_CKPT = TEXT_PRETRAIN_DIR / "checkpoints" / "best.pt"
DEFAULT_CONFIG = PROJECT_ROOT / "config.toml"

# Import Llama from text_pretraining/model.py (avoid name conflict)
spec = importlib.util.spec_from_file_location("llama_model", TEXT_PRETRAIN_DIR / "model.py")
llama_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llama_module)
Llama = llama_module.Llama


class VisionEncoder(nn.Module):
    """
    Vision encoder using pretrained ViT-B/16 (frozen) with trainable MLP projector.

    Flow: Image [B,3,224,224] -> ViT -> MLP -> [B, 196, llm_embed_dim]
    """

    def __init__(self, llm_embed_dim=768):
        super().__init__()

        # Load pretrained ViT-B/16
        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        # Extract components
        self.patch_embed = vit.conv_proj
        self.pos_embed = vit.encoder.pos_embedding
        self.encoder = vit.encoder.layers
        self.norm = vit.encoder.ln

        # Freeze vision encoder
        for param in self.parameters():
            param.requires_grad = False

        # Trainable 2-layer MLP projector (like LLaVA)
        self.proj = nn.Sequential(
            nn.Linear(768, 768 * 4),
            nn.GELU(),
            nn.Linear(768 * 4, llm_embed_dim)
        )

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # [B, 196, 768]

        # Add positional embeddings (skip CLS position)
        x = x + self.pos_embed[:, 1:, :]

        # Transformer encoder
        for layer in self.encoder:
            x = layer(x)
        x = self.norm(x)

        # Project to LLM space
        x = self.proj(x)
        return x


class VisionLanguageModel(nn.Module):
    """
    Vision-Language Model combining ViT encoder with pretrained LLM.

    Only the MLP projector is trainable (~4.7M params).
    """

    def __init__(self, vocab_size, llm_embed_dim, llm_config, llm_ckpt_path):
        """
        Args:
            vocab_size: Original vocab size (32000)
            llm_embed_dim: LLM embedding dimension (768)
            llm_config: Config dict for Llama model
            llm_ckpt_path: Path to pretrained LLM checkpoint
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.llm_embed_dim = llm_embed_dim
        self.num_vision_tokens = 196

        # Vision encoder
        self.vision_encoder = VisionEncoder(llm_embed_dim=llm_embed_dim)

        # LLM
        self.llm = Llama(
            vocab_size=vocab_size,
            n_layers=llm_config["n_layers"],
            d_model=llm_config["d_model"],
            d_head=llm_config["d_head"],
            n_heads=llm_config["n_heads"],
            n_kv_heads=llm_config["n_kv_heads"],
            kv_d_head=llm_config["kv_d_head"],
            d_ff_standard=llm_config["d_ff_standard"],
            num_experts=llm_config["num_experts"],
            num_experts_per_tok=llm_config["num_experts_per_tok"],
            d_expert=llm_config["d_expert"],
            rope_layers_ratio=llm_config["rope_layers_ratio"],
            chunk_size=llm_config["chunk_size"],
            rope_theta=llm_config["rope_theta"],
        )

        # Load pretrained LLM weights
        self._load_llm_checkpoint(llm_ckpt_path)

        # Expand embedding layer (32000 -> 32001 for <image> token)
        self._expand_embeddings()

        # Freeze LLM (after expanding)
        for param in self.llm.parameters():
            param.requires_grad = False

        # Special token IDs
        self.image_token_id = vocab_size  # 32000
        self.pad_token_id = 3

    def _load_llm_checkpoint(self, ckpt_path):
        """Load pretrained LLM weights."""
        if Path(ckpt_path).exists():
            print(f"Loading LLM checkpoint from: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            self.llm.load_state_dict(checkpoint["model_state_dict"])
            print("LLM checkpoint loaded successfully")
        else:
            raise FileNotFoundError(f"LLM checkpoint not found: {ckpt_path}")

    def _expand_embeddings(self):
        """Expand embedding and projection layers for <image> token."""
        # Expand input embedding
        old_emb = self.llm.emb.emb
        new_emb = nn.Embedding(self.vocab_size + 1, self.llm_embed_dim)
        new_emb.weight.data[:self.vocab_size, :] = old_emb.weight.data
        new_emb.weight.data[self.vocab_size, :] = torch.randn(self.llm_embed_dim) * (self.llm_embed_dim ** -0.5)
        self.llm.emb.emb = new_emb

        # Expand output projection
        old_proj = self.llm.proj_vocab
        new_proj = nn.Linear(self.llm_embed_dim, self.vocab_size + 1, bias=False)
        new_proj.weight.data[:self.vocab_size, :] = old_proj.weight.data
        new_proj.weight.data[self.vocab_size, :] = torch.randn(self.llm_embed_dim) * 0.02
        self.llm.proj_vocab = new_proj

    def forward(self, image, input_ids):
        """
        Forward pass for vision-language model.

        Args:
            image: [B, 3, 224, 224]
            input_ids: [B, seq_len] with <image> token at position 0

        Returns:
            logits: [B, 196 + seq_len - 1, vocab_size + 1]
        """
        # Vision encoding
        vision_embeds = self.vision_encoder(image)  # [B, 196, d_model]

        # Text embedding (skip <image> token at position 0)
        text_embeds = self.llm.emb(input_ids)  # [B, seq_len, d_model]
        text_embeds = text_embeds[:, 1:, :]  # [B, seq_len-1, d_model]

        # Combine vision + text
        combined = torch.cat([vision_embeds, text_embeds], dim=1)  # [B, 196+seq_len-1, d_model]

        # Pass through LLM decoder
        for i, decoder in enumerate(self.llm.decoder_layers):
            combined = decoder(i, combined)

        combined = self.llm.rms_norm(combined)
        logits = self.llm.proj_vocab(combined)

        return logits


def create_vlm(config, llm_ckpt_path):
    """
    Factory function to create VisionLanguageModel.

    Args:
        config: Config dict with model parameters
        llm_ckpt_path: Path to pretrained LLM checkpoint

    Returns:
        VisionLanguageModel instance
    """
    # Compute derived config values
    config["d_head"] = config["d_model"] // config["n_heads"]
    config["kv_d_head"] = config["d_model"] // config["n_kv_heads"]

    model = VisionLanguageModel(
        vocab_size=config["vocab_size"],
        llm_embed_dim=config["d_model"],
        llm_config=config,
        llm_ckpt_path=llm_ckpt_path,
    )

    return model


if __name__ == "__main__":
    import tomllib

    # Load config
    with open(DEFAULT_CONFIG, "rb") as f:
        cfg = tomllib.load(f)

    # Create model
    model = create_vlm(cfg, str(DEFAULT_LLM_CKPT))

    # Check trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Test forward pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    image = torch.randn(2, 3, 224, 224, device=device)
    input_ids = torch.randint(0, 32000, (2, 128), device=device)
    input_ids[:, 0] = 32000  # <image> token

    logits = model(image, input_ids)
    print(f"Input image: {image.shape}")
    print(f"Input ids: {input_ids.shape}")
    print(f"Output logits: {logits.shape}")
