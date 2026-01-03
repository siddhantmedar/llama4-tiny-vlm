#!/usr/bin/env python3
"""
Instruction-Tuned Vision-Language Model for LLaMA 4.

Loads Phase 2 VLM checkpoint and unfreezes projector + LLM for instruction tuning.
ViT encoder remains frozen.

Trainable: ~385M params (81.7%)
  - MLP Projector: 4.7M
  - LLM Decoder: 380M
Frozen: ~90M params (ViT encoder)
"""

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEXT_PRETRAIN_DIR = PROJECT_ROOT / "text_pretraining"
VL_ALIGNMENT_DIR = PROJECT_ROOT / "vision_language_alignment"

DEFAULT_CONFIG = PROJECT_ROOT / "config.toml"
DEFAULT_LLM_CKPT = TEXT_PRETRAIN_DIR / "checkpoints" / "best.pt"
DEFAULT_VLM_CKPT = VL_ALIGNMENT_DIR / "checkpoints" / "best_vlm.pt"

# Import VisionLanguageModel from Phase 2
spec = importlib.util.spec_from_file_location("vlm_model", VL_ALIGNMENT_DIR / "model.py")
vlm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vlm_module)
VisionLanguageModel = vlm_module.VisionLanguageModel


class InstructionTunedVLM(nn.Module):
    """
    Instruction-tuned VLM wrapper.

    Wraps Phase 2 VisionLanguageModel with modified freezing:
    - ViT encoder: frozen (vision_encoder.patch_embed, encoder, norm, pos_embed)
    - MLP Projector: trainable (vision_encoder.proj)
    - LLM: trainable (llm.*)
    """

    def __init__(self, config, vocab_size, llm_embed_dim, llm_ckpt_path, vlm_ckpt_path):
        super().__init__()

        # Create VLM (loads LLM checkpoint internally)
        self.vlm = VisionLanguageModel(
            vocab_size=vocab_size,
            llm_embed_dim=llm_embed_dim,
            llm_config=config,
            llm_ckpt_path=llm_ckpt_path
        )

        # Load Phase 2 VLM checkpoint (trained projector)
        self._load_vlm_checkpoint(vlm_ckpt_path)

        # Set freezing: ViT frozen, projector + LLM trainable
        self._configure_freezing()

    def _load_vlm_checkpoint(self, ckpt_path):
        """Load Phase 2 VLM checkpoint with trained projector."""
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"VLM checkpoint not found: {ckpt_path}")

        print(f"Loading VLM checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.vlm.load_state_dict(checkpoint["model_state_dict"])
        print(f"VLM checkpoint loaded (val_loss: {checkpoint.get('val_loss', 'N/A'):.4f})")

    def _configure_freezing(self):
        """
        Configure parameter freezing for instruction tuning.

        Frozen (ViT encoder):
          - vision_encoder.patch_embed
          - vision_encoder.pos_embed
          - vision_encoder.encoder
          - vision_encoder.norm

        Trainable:
          - vision_encoder.proj (MLP projector)
          - llm.* (all LLM parameters)
        """
        for name, param in self.vlm.named_parameters():
            if 'vision_encoder.proj' in name or 'llm' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, image, input_ids):
        """
        Forward pass.

        Args:
            image: [B, 3, 224, 224] - input images
            input_ids: [B, seq_len] - token ids with <image> at position 0

        Returns:
            logits: [B, 196 + seq_len - 1, vocab_size + 1]
        """
        return self.vlm(image, input_ids)


def create_instruct_vlm(config, llm_ckpt_path, vlm_ckpt_path):
    """
    Factory function to create InstructionTunedVLM.

    Args:
        config: Config dict with model parameters
        llm_ckpt_path: Path to Phase 1 LLM checkpoint
        vlm_ckpt_path: Path to Phase 2 VLM checkpoint

    Returns:
        InstructionTunedVLM instance
    """
    # Compute derived config values
    config["d_head"] = config["d_model"] // config["n_heads"]
    config["kv_d_head"] = config["d_model"] // config["n_kv_heads"]

    model = InstructionTunedVLM(
        config=config,
        vocab_size=config["vocab_size"],
        llm_embed_dim=config["d_model"],
        llm_ckpt_path=llm_ckpt_path,
        vlm_ckpt_path=vlm_ckpt_path
    )

    return model


if __name__ == "__main__":
    import tomllib

    # Load config
    with open(DEFAULT_CONFIG, "rb") as f:
        cfg = tomllib.load(f)

    # Create model
    model = create_instruct_vlm(cfg, str(DEFAULT_LLM_CKPT), str(DEFAULT_VLM_CKPT))

    # Check trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable
    print(f"\nParameter Summary:")
    print(f"  Trainable: {trainable:,} ({100*trainable/total:.1f}%)")
    print(f"  Frozen:    {frozen:,} ({100*frozen/total:.1f}%)")
    print(f"  Total:     {total:,}")

    # Test forward pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    image = torch.randn(2, 3, 224, 224, device=device)
    input_ids = torch.randint(0, 32000, (2, 128), device=device)
    input_ids[:, 0] = 32000  # <image> token

    with torch.no_grad():
        logits = model(image, input_ids)

    print(f"\nForward Pass Test:")
    print(f"  Input image:   {image.shape}")
    print(f"  Input ids:     {input_ids.shape}")
    print(f"  Output logits: {logits.shape}")
    print(f"\nModel ready for instruction tuning!")
