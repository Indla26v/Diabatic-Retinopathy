import torch
import torch.nn as nn
import torchvision.models as tv_models

class CNNTransformerDR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        eff = tv_models.efficientnet_b3(weights=tv_models.EfficientNet_B3_Weights.DEFAULT)
        self.cnn_encoder = eff.features

        self.proj = nn.Conv2d(1536, cfg['token_dim'], kernel_size=1, bias=False)

        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg['token_dim']))
        self.pos_embed = nn.Parameter(torch.randn(1, 101, cfg['token_dim']) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg['token_dim'],
            nhead=cfg['num_heads'],
            dim_feedforward=cfg['ff_dim'],
            dropout=cfg['tf_dropout'],
            activation='gelu',
            norm_first=True,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.classifier = nn.Sequential(
            nn.Linear(cfg['token_dim'], 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, cfg['num_classes']),
        )

    def forward(self, x):
        feat = self.cnn_encoder(x)
        feat = self.proj(feat)
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).permute(0, 2, 1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]
        tokens = self.transformer(tokens)
        pooled = tokens[:, 0]
        return self.classifier(pooled)

cfg = {'token_dim': 512, 'num_heads': 8, 'ff_dim': 1024, 'tf_dropout': 0.15, 'num_classes': 5}
model = CNNTransformerDR(cfg)
ckpt = torch.load('best_hybrid_model (1).pth', map_location='cpu', weights_only=True)
model.load_state_dict(ckpt)
print('Loaded successfully')
