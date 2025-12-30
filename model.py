import torch
import torch.nn as nn
import tomllib

with open("config.toml", "rb") as f:
    cfg = tomllib.load(f)
    cfg["d_head"] = cfg["d_model"] // cfg["num_head"]



class Dropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0:
            mask = (torch.rand_like(x) > self.p).float()
            x = mask * x
            x /= 1 - self.p
        return x


class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.eps = 1e-6
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mu, var = x.mean(dim=-1, keepdim=True), x.var(dim=-1, keepdim=True)
        x = (x - mu) / (var + self.eps) ** 0.5
        return x * self.alpha + self.beta


class Embedding(nn.Module):
    def __init__(self, patch_size, img_size, in_channels=3):
        super().__init__()
        
        emb = nn.Embedding(cfg['vocab_size'], cfg['d_model'])
        torch.nn.init.normal_(emb.weight, mean=0.0, std=(cfg['d_model'])**-0.5)

    def forward(self, x):
        token_emb = self.proj(x)  # [b, p, patch_dim] -> [b,p,d_model]
        x = token_emb + self.pos_emb  # [b, p, d_model]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_q = nn.Linear(cfg["d_model"], cfg["d_model"])
        self.w_k = nn.Linear(cfg["d_model"], cfg["d_model"])
        self.w_v = nn.Linear(cfg["d_model"], cfg["d_model"])

        self.proj_out = nn.Linear(cfg["d_model"], cfg["d_model"])

    def forward(self, x):
        b, p, _ = x.shape

        q = self.w_q(x)  # [b,p,d_model]
        k = self.w_k(x)  # [b,p,d_model]
        v = self.w_v(x)  # [b,p,d_model]

        q = q.view(b, p, cfg["num_head"], cfg["d_head"]).transpose(
            1, 2
        )  # [b,h,p,d_head]
        k = k.view(b, p, cfg["num_head"], cfg["d_head"]).transpose(
            1, 2
        )  # [b,h,p,d_head]
        v = v.view(b, p, cfg["num_head"], cfg["d_head"]).transpose(
            1, 2
        )  # [b,h,p,d_head]

        scores = q @ k.transpose(-2, -1)  # [b,h,p,p]
        attn_weights = torch.softmax(
            scores / (cfg["d_head"] ** 0.5), dim=-1
        )  # [b,h,p,p]
        output = attn_weights @ v  # [b,h,p,p] @ [b,h,p,d_head] = [b,h,p,d_head]
        output = output.transpose(1, 2).contiguous().view(b, p, -1)

        output = self.proj_out(output)

        return output  # [b,p,d_model]


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_block = nn.Sequential(
            nn.Linear(cfg["d_model"], 4 * cfg["d_model"]), nn.GELU(), Dropout(0.1)
        )

        self.fc2_block = nn.Sequential(
            nn.Linear(4 * cfg["d_model"], cfg["d_model"]), Dropout(0.1)
        )

    def forward(self, x):
        x = self.fc1_block(x)  # [b,p,4*d_model]
        x = self.fc2_block(x)  # [b,p,d_model]
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = LayerNorm(cfg["d_model"])
        self.layer_norm2 = LayerNorm(cfg["d_model"])

        self.dropout = Dropout
        self.mha = MultiHeadAttention()
        self.mlp = MLP()

    def forward(self, x):
        # Attention block with residual
        x_residual = x
        x = self.layer_norm1(x)
        x = self.mha(x)
        x = x + x_residual

        # MLP block with residual
        x_residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + x_residual

        return x


class VisionTransformer(nn.Module):
    def __init__(self, patch_size=4, img_size=32, in_channels=3, num_classes=100):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.embedding = Embedding(patch_size, img_size, in_channels)
        self.encoder_layers = nn.ModuleList(
            [Encoder() for _ in range(cfg["num_encoder_layer"])]
        )
        self.norm = LayerNorm(cfg["d_model"])
        self.head = nn.Linear(cfg["d_model"], num_classes)

    def _extract_patches(self, x):
        # x: [b, c, h, w]
        b, c = x.shape[:2]
        p = self.patch_size

        # Unfold to patches
        x = x.unfold(2, p, p).unfold(3, p, p)  # [b, c, n_h, n_w, p, p]
        x = x.contiguous().view(b, c, -1, p, p)  # [b, c, num_patches, p, p]
        x = x.permute(0, 2, 1, 3, 4)  # [b, num_patches, c, p, p]
        x = x.contiguous().view(b, -1, c * p * p)  # [b, num_patches, c*p*p]

        return x

    def forward(self, x):
        # Extract patches
        x = self._extract_patches(x)  # [b,c,h,w] -> [b,num_patches,c*p*p]

        # Embedding
        x = self.embedding(x)  # [b,num_patches,d_model]

        # Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)

        # Classification head (use [CLS] token or global average pooling)
        x = self.norm(x)
        x = x.mean(
            dim=1
        )  # Global average pooling [b,num_patches,d_model] -> [b,d_model]
        out = self.head(x)  # [b,num_classes]

        return out


if __name__ == "__main__":
    # Test the model
    model = VisionTransformer(patch_size=4, img_size=32, in_channels=3, num_classes=100)

    # Test forward pass
    x = torch.randn(1, 3, 32, 32)  # [batch, channels, height, width]
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # Should be [1, 100]
