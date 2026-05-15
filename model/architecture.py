import torch
import torch.nn as nn


# model
class ChessTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, d_ff, dropout):
        super(ChessTransformer, self).__init__()

        self.input_embedding = nn.Linear(19, d_model)

        self.rank_embedding = nn.Parameter(torch.randn(1, 8, d_model // 2))
        self.file_embedding = nn.Parameter(torch.randn(1, 8, d_model // 2))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.evaluation_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.move_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, 64)
        )

    def forward(self, x, legal_move_mask=None):
        batch_size = x.shape[0]
        x = self.input_embedding(x)

        x = x + torch.cat((self.rank_embedding.repeat_interleave(8, dim=1), self.file_embedding.repeat(1, 8, 1)), 2)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        logits = self.encoder(x)

        evaluation_values = logits[:, 0, :]
        evaluation_logits = self.evaluation_head(evaluation_values)

        move_values = logits[:, 1:, :]
        move_logits = self.move_head(move_values)

        if legal_move_mask is not None:
            move_logits = move_logits.view(batch_size, -1)

            move_logits += legal_move_mask

        return evaluation_logits, move_logits


class ChessTransformer_conv(nn.Module):
    def __init__(self, d_model, nhead, num_layers, d_ff, dropout):
        super(ChessTransformer_conv, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=19, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU()
        )

        self.rank_embedding = nn.Parameter(torch.randn(1, 8, d_model // 2))
        self.file_embedding = nn.Parameter(torch.randn(1, 8, d_model // 2))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.evaluation_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.move_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, 64)
        )

    def forward(self, x, legal_move_mask=None):
        batch_size = x.shape[0]

        x = torch.transpose(x, 1, 2)
        x = x.view(batch_size, 19, 8, 8)

        x = self.conv_layers(x)

        x = x.view(batch_size, 512, -1)
        x = torch.transpose(x, 1, 2)

        x = x + torch.cat((self.rank_embedding.repeat_interleave(8, dim=1), self.file_embedding.repeat(1, 8, 1)), 2)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        logits = self.encoder(x)

        evaluation_values = logits[:, 0, :]
        evaluation_logits = self.evaluation_head(evaluation_values)

        move_values = logits[:, 1:, :]
        move_logits = self.move_head(move_values)

        if legal_move_mask is not None:
            move_logits = move_logits.view(batch_size, -1)

            move_logits += legal_move_mask

        return evaluation_logits, move_logits


"""model = ChessTransformer(256, 16, 8, 1024, 0.1)
print(model)
num_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)
print(f"Number of parameters: {num_params}")"""