#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import os
from pathlib import Path
import tomllib
import numpy as np


def set_seed(seed=313):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()

DEVICE = "cuda"
CONFIG_PATH = Path(__file__).parent.parent / "config.toml"

with open(CONFIG_PATH, "rb") as f:
    cfg = tomllib.load(f)
    assert (
        divmod(cfg["d_model"], cfg["n_heads"])[1] == 0
    ), "d_model should be divisble by n_heads"
    assert (
        divmod(cfg["n_heads"], cfg["n_kv_heads"])[1] == 0
    ), "n_heads should be divisble by n_kv_heads"

    cfg["d_head"] = cfg["d_model"] // cfg["n_heads"]
    cfg["kv_d_head"] = cfg["d_model"] // cfg["n_kv_heads"]


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.emb.weight, mean=0, std=(d_model) ** -0.5)

    def forward(self, x):
        return self.emb(x)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        den = ((x**2).sum(dim=-1, keepdim=True) / x.size(-1) + 1e-6) ** 0.5
        return (x / den) * self.gamma


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        d_head,
        n_heads,
        n_kv_heads,
        kv_d_head,
        rope_layers_ratio,
        chunk_size,
        rope_theta,
    ):
        super().__init__()

        self.rope_period = int(1 / (1 - rope_layers_ratio))
        self.rope_theta = rope_theta
        self.chunk_size = chunk_size

        self.d_model = d_model
        self.d_head = d_head
        self.kv_d_head = kv_d_head

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads

        self.q_norm = RMSNorm(d_head)
        self.k_norm = RMSNorm(d_head)

        self.w_q = nn.Linear(self.d_model, self.n_heads * self.d_head, bias=False)
        self.w_k = nn.Linear(self.d_model, self.n_kv_heads * self.d_head, bias=False)
        self.w_v = nn.Linear(self.d_model, self.n_kv_heads * self.d_head, bias=False)

        self.temp_scale = nn.Parameter(torch.tensor(1.0))  # learnable, starts at 1.0
        self.proj_out = nn.Linear(self.d_model, self.d_model, bias=False)

    @staticmethod
    def _apply_rope(x, base=10_000):
        """
        For each pair of dimensions (2i, 2i+1):
        q_rotated[2i]   = q[2i] * cos(m*θᵢ) - q[2i+1] * sin(m*θᵢ)
        q_rotated[2i+1] = q[2i] * sin(m*θᵢ) + q[2i+1] * cos(m*θᵢ)

        Where theta_i = 10000^{-2i/d}
        """
        # x: [b,s,n_heads,d_head]
        s, _, dim = x.shape[1:]
        x_rotated = torch.zeros_like(x)

        m = torch.arange(s, device=x.device).unsqueeze(-1)  # pos
        i = torch.arange(0, dim, 2, device=x.device)  # dim idx

        theta = base ** (-i / dim)

        x1, x2 = x[:, :, :, 0::2], x[:, :, :, 1::2]

        angle = (m * theta).unsqueeze(-2)
        x_rotated[:, :, :, 0::2] = x1 * torch.cos(angle) - x2 * torch.sin(angle)
        x_rotated[:, :, :, 1::2] = x1 * torch.sin(angle) + x2 * torch.cos(angle)

        return x_rotated

    @staticmethod
    def _create_chunk_mask(seq_length, chunk_size, device):
        """RoPE layers: attend within chunks only"""
        mask = torch.zeros(seq_length, seq_length, device=device)

        for i in range(0, seq_length, chunk_size):
            end = min(seq_length, i + chunk_size)
            mask[i:end, i:end] = 1

        mask = mask * torch.tril(torch.ones_like(mask))

        return ~mask.bool()

    def forward(self, idx, x):
        b, s, _ = x.shape

        q = self.w_q(x).view(b, s, self.n_heads, self.d_head)  # [b,s,n_heads,d_head]
        k = self.w_k(x).view(
            b, s, self.n_kv_heads, self.d_head
        )  # [b,s,n_kv_heads,d_head]
        v = self.w_v(x).view(
            b, s, self.n_kv_heads, self.d_head
        )  # [b,s,n_kv_heads,d_head]

        q = self.q_norm(q)
        k = self.k_norm(k)

        is_rope_layer = (idx + 1) % self.rope_period != 0

        if is_rope_layer:
            q = MultiHeadAttention._apply_rope(q, self.rope_theta)  # rotated
            k = MultiHeadAttention._apply_rope(k, self.rope_theta)  # rotated

        # repeat KV heads to match Q heads
        n_rep = self.n_heads // self.n_kv_heads
        k = torch.repeat_interleave(k, repeats=n_rep, dim=2)  # [b,s,n_heads,d_head]
        v = torch.repeat_interleave(v, repeats=n_rep, dim=2)  # [b,s,n_heads,d_head]

        # Transpose for attention
        q = q.transpose(1, 2)  # [b,n_heads,s,d_head]
        k = k.transpose(1, 2)  # [b,n_heads,s,d_head]
        v = v.transpose(1, 2)  # [b,n_heads,s,d_head]

        scores = q @ k.transpose(-1, -2) / (self.d_head**0.5)  # [b,n_heads,s,s]

        # Temperature scaling for NoPE layers - apply BEFORE masking to avoid -inf/temp gradient issues
        if not is_rope_layer:
            temp = F.softplus(self.temp_scale) + 0.5  # min ~0.5
            scores = scores / temp

        # causal mask
        if is_rope_layer:
            mask = MultiHeadAttention._create_chunk_mask(s, self.chunk_size, x.device)
        else:
            # NoPE: full causal attention
            mask = torch.triu(torch.ones(s, s, device=x.device), diagonal=1).bool()

        scores = torch.masked_fill(scores, mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)

        output = attn_weights @ v  # [b,n_heads,s,d_head]
        output = output.contiguous().transpose(1, 2).reshape(b, s, self.d_model)

        return self.proj_out(output)  # [b,s,d_model]


class DenseFFNBlock(nn.Module):
    def __init__(self, d_model, d_ff_standard):
        super().__init__()

        self.w_gate = nn.Linear(d_model, d_ff_standard, bias=False)
        self.w_up = nn.Linear(d_model, d_ff_standard, bias=False)
        self.w_down = nn.Linear(d_ff_standard, d_model, bias=False)

    def forward(self, x):
        # x: [b,s,d_model]
        out1 = F.silu(self.w_gate(x))  # [b,s,dff]
        out2 = self.w_up(x)  # [b,s,dff]
        return self.w_down((out1 * out2))  # [b,s,d_model]


class Expert(nn.Module):
    def __init__(self, d_model, d_expert):
        super().__init__()

        self.w_gate = nn.Linear(d_model, d_expert, bias=False)
        self.w_up = nn.Linear(d_model, d_expert, bias=False)
        self.w_down = nn.Linear(d_expert, d_model, bias=False)

    def forward(self, x):
        # x: [b*s,d_model]
        out1 = F.silu(self.w_gate(x))  # [b*s,dexp]
        out2 = self.w_up(x)  # [b*s,dexp]
        return self.w_down((out1 * out2))  # [b*s,d_model]


class GatingNetwork(nn.Module):
    def __init__(self, num_experts, num_experts_per_tok, d_model):
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.w_router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        logits = self.w_router(x)  # [b,s,num_experts]
        router_probs = F.softmax(logits, dim=-1)  # [b,s,num_experts]
        top_k_weights, top_k_indices = torch.topk(
            router_probs, self.num_experts_per_tok, dim=-1
        )
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        return top_k_weights, top_k_indices, router_probs


class MoEFFNBlock(nn.Module):
    def __init__(self, num_experts, num_experts_per_tok, d_expert, d_model):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.gate = GatingNetwork(num_experts, num_experts_per_tok, self.d_model)

        self.shared_expert = Expert(d_model, d_expert)
        self.experts = nn.ModuleList(
            [Expert(d_model, d_expert) for _ in range(self.num_experts)]
        )

        self.aux_loss = 0.0

    def _load_balancing_loss(self, router_probs, indices):
        """
        encourages uniform expert usage. without this, router collapses to always picking same experts.
        note: the minimum loss is always 1.0 regardless of number of experts!
        """
        # router_probs: [b, s, num_experts] - pre-topk softmax probs
        # indices: [b, s, topk] - selected expert indices

        router_probs_flat = router_probs.view(
            -1, self.num_experts
        )  # [b*s, num_experts]
        indices_flat = indices.view(-1)  # [b*s*topk]

        # f_i = fraction of tokens routed to expert i
        counts = torch.bincount(indices_flat, minlength=self.num_experts).float()
        f = counts / indices_flat.numel()

        # P_i = mean routing probability for expert i
        P = router_probs_flat.mean(dim=0)

        # Loss = N * Σ(f_i * P_i)
        # Minimized when both f and P are uniform (1/N each)
        return self.num_experts * (f * P).sum()

    def forward(self, x):
        b, s, d_model = x.shape
        weights, indices, router_probs = self.gate(x)

        x_flatten = x.contiguous().view(-1, self.d_model)  # [b*s,d_model]

        output = torch.zeros_like(x_flatten)  # [b*s,d]

        shared_output = self.shared_expert(x_flatten)  # [b*s,d_model]

        # for each expert, gather tokens assigned to it
        for expert_idx in range(self.num_experts):
            expert_mask = indices == expert_idx  # [b,s,topk]
            token_mask = expert_mask.any(dim=-1).flatten(start_dim=0)  # [b*s]
            if token_mask.sum() == 0:
                continue

            token_weights = weights[expert_mask]  # num_assigned

            expert_input = x_flatten[token_mask]  # [num_assigned,d_model]
            expert_output = self.experts[expert_idx](
                expert_input
            )  # [num_assigned,d_model]
            output[token_mask] += (
                token_weights.unsqueeze(-1) * expert_output
            )  # [b*s,d_model]

        output = output + shared_output

        self.aux_loss = self._load_balancing_loss(router_probs, indices)

        return output.contiguous().view(b, s, d_model)  # [b,s,d_model]


class FFN(nn.Module):
    def __init__(
        self, d_model, d_ff_standard, num_experts, num_experts_per_tok, d_expert
    ):
        super().__init__()
        self.dense_ffn_block = DenseFFNBlock(d_model, d_ff_standard)
        self.moe_ffn_block = MoEFFNBlock(
            num_experts, num_experts_per_tok, d_expert, d_model
        )

    def forward(self, idx, x):
        return self.dense_ffn_block(x) if idx % 2 == 0 else self.moe_ffn_block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        d_model,
        d_head,
        n_heads,
        n_kv_heads,
        kv_d_head,
        d_ff_standard,
        num_experts,
        num_experts_per_tok,
        d_expert,
        rope_layers_ratio,
        chunk_size,
        rope_theta,
    ):
        super().__init__()
        self.mha = MultiHeadAttention(
            d_model,
            d_head,
            n_heads,
            n_kv_heads,
            kv_d_head,
            rope_layers_ratio,
            chunk_size,
            rope_theta,
        )
        self.ffn = FFN(
            d_model, d_ff_standard, num_experts, num_experts_per_tok, d_expert
        )
        self.rms_norm1 = RMSNorm(d_model)
        self.rms_norm2 = RMSNorm(d_model)

    def forward(self, idx, x):
        x_original = x
        x = self.rms_norm1(x)
        x = self.mha(idx, x) + x_original

        x_original = x
        x = self.rms_norm2(x)
        x = self.ffn(idx, x)

        return x + x_original


class Llama(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_layers,
        d_model,
        d_head,
        n_heads,
        n_kv_heads,
        kv_d_head,
        d_ff_standard,
        num_experts,
        num_experts_per_tok,
        d_expert,
        rope_layers_ratio,
        chunk_size,
        rope_theta,
    ):
        super().__init__()

        self.emb = Embedding(vocab_size, d_model)
        self.decoder_layers = nn.ModuleList(
            [
                Decoder(
                    d_model,
                    d_head,
                    n_heads,
                    n_kv_heads,
                    kv_d_head,
                    d_ff_standard,
                    num_experts,
                    num_experts_per_tok,
                    d_expert,
                    rope_layers_ratio,
                    chunk_size,
                    rope_theta,
                )
                for _ in range(n_layers)
            ]
        )
        self.rms_norm = RMSNorm(d_model)
        self.proj_vocab = nn.Linear(d_model, vocab_size, bias=False)
        self.aux_loss = 0.0

    def forward(self, x):
        out = self.emb(x)

        aux_loss = 0.0

        for i, decoder in enumerate(self.decoder_layers):
            out = decoder(i, out)

            # Only add aux_loss for MoE layers (odd indices use MoE)
            if i % 2 == 1:  # MoE layer
                aux_loss += decoder.ffn.moe_ffn_block.aux_loss

        out = self.rms_norm(out)
        logits = self.proj_vocab(out)

        self.aux_loss = aux_loss

        return logits


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on: {device}")

    model = Llama(
        vocab_size=cfg["vocab_size"],
        n_layers=cfg["n_layers"],
        d_model=cfg["d_model"],
        d_head=cfg["d_head"],
        n_heads=cfg["n_heads"],
        n_kv_heads=cfg["n_kv_heads"],
        kv_d_head=cfg["kv_d_head"],
        d_ff_standard=cfg["d_ff_standard"],
        num_experts=cfg["num_experts"],
        num_experts_per_tok=cfg["num_experts_per_tok"],
        d_expert=cfg["d_expert"],
        rope_layers_ratio=cfg["rope_layers_ratio"],
        chunk_size=cfg["chunk_size"],
        rope_theta=cfg["rope_theta"],
    ).to(device)

    x = torch.randint(0, 900, (2, 512), device=device)  # Test with seq_len=512

    print(f"input shape: {x.shape}")
    out = model(x)
    print(f"output shape: {out.shape}")
    print(f"output has NaN: {torch.isnan(out).any()}")
    print(f"output min/max: {out.min():.4f} / {out.max():.4f}")
