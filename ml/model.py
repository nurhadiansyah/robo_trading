"""
Deep Learning Architecture: Financial Time-Series Models
Features:
1. MarketTransformer (Primary): Multi-Head Self-Attention, Positional Encoding, Pre-LayerNorm, Temporal Readout.
2. MarketAttentionLSTM (Legacy Compatible): Bi-LSTM with Temporal Attention.
3. MarketAttentionLSTM_M1 (Legacy Compatible): 1-Minute Bi-LSTM checkpoint loader.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for time-series sequences.
    Preserves chronological order of market candles in lookback window.
    """
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TemporalAttentionPooling(nn.Module):
    """
    Attention pooling layer to extract sequence-level representations
    from transformer token representations.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, d_model)
        scores = self.query(x)               # (batch_size, seq_len, 1)
        weights = F.softmax(scores, dim=1)   # (batch_size, seq_len, 1)
        context = torch.sum(weights * x, dim=1)  # (batch_size, d_model)
        return context, weights

class MarketTransformer(nn.Module):
    """
    Financial Market Time-Series Transformer
    - Input Feature LayerNorm & Linear Projection
    - Sinusoidal Positional Encoding
    - Transformer Encoder Layers with Multi-Head Self-Attention & GELU FeedForward
    - Pre-LayerNorm (norm_first=True) for stable training dynamics
    - Temporal Attention Readout
    - Multi-Task Dual Heads:
      1) Direction Classification (Logits for Class 0: Bearish/Down, Class 1: Bullish/Up)
      2) Expected Return Regression (Scalar price return percentage)
    """
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.2,
                 max_len: int = 500, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # Input Projection & Normalization
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        # Transformer Encoder Stack with Pre-LayerNorm for gradient stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Sequence Readout
        self.readout_norm = nn.LayerNorm(d_model)
        self.attention_pooling = TemporalAttentionPooling(d_model)

        # Shared Representation
        self.shared_fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Head 1: Direction Classification (0: Down, 1: Up)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 2)
        )

        # Head 2: Expected Return Regression
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x_norm = self.input_norm(x)
        x_proj = self.input_proj(x_norm)             # (batch_size, seq_len, d_model)
        x_pe = self.pos_encoder(x_proj)

        encoded = self.transformer_encoder(x_pe)     # (batch_size, seq_len, d_model)
        encoded_norm = self.readout_norm(encoded)

        context, attn_weights = self.attention_pooling(encoded_norm)  # (b, d_model)
        feat = self.shared_fc(context)

        dir_logits = self.classifier(feat)
        exp_return = self.regressor(feat).squeeze(-1)
        return dir_logits, exp_return, attn_weights

    def predict_signal(self, x, threshold: float = 0.55):
        self.eval()
        with torch.no_grad():
            dir_logits, exp_ret, _ = self.forward(x)
            probs = F.softmax(dir_logits, dim=-1)
            prob_up = probs[:, 1].item()
            prob_down = probs[:, 0].item()

            if prob_up >= threshold:
                signal = "BUY"
                confidence = prob_up
            elif prob_down >= threshold:
                signal = "SELL"
                confidence = prob_down
            else:
                signal = "HOLD"
                confidence = max(prob_up, prob_down)

            return {
                "signal": signal,
                "confidence": float(confidence),
                "prob_up": float(prob_up),
                "prob_down": float(prob_down),
                "expected_return": float(exp_ret.item())
            }


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_outputs):
        scores = self.projection(lstm_outputs)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * lstm_outputs, dim=1)
        return context, weights


class TemporalAttentionM1(nn.Module):
    """Attention layer matching Kaggle weights (model_m1.pth) keys exactly."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )

    def forward(self, lstm_outputs):
        scores = self.attn(lstm_outputs)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * lstm_outputs, dim=1)
        return context, weights


class MarketAttentionLSTM(nn.Module):
    """Legacy Bi-LSTM with Temporal Attention Architecture."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.25, **kwargs):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        bi_hidden = hidden_dim * 2
        self.attention = TemporalAttention(bi_hidden)
        self.shared_fc = nn.Sequential(
            nn.Linear(bi_hidden, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 2)
        )
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x_norm = self.input_norm(x)
        lstm_out, _ = self.lstm(x_norm)
        context, attn_weights = self.attention(lstm_out)
        feat = self.shared_fc(context)
        dir_logits = self.classifier(feat)
        exp_return = self.regressor(feat).squeeze(-1)
        return dir_logits, exp_return, attn_weights


class MarketAttentionLSTM_M1(nn.Module):
    """Legacy 1-Minute Scalping LSTM architecture for Kaggle weights."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.25, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        rnn_out_dim = hidden_dim * 2
        self.attention = TemporalAttentionM1(rnn_out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_out_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        self.regressor = nn.Sequential(
            nn.Linear(rnn_out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        logits = self.classifier(context)
        exp_return = self.regressor(context).squeeze(-1)
        return logits, exp_return, attn_weights


# Aliases
MarketTransformerM1 = MarketTransformer


if __name__ == "__main__":
    # Smoke test for Transformer model shape
    model = MarketTransformer(input_dim=22, d_model=64, nhead=4, num_layers=2)
    dummy_input = torch.randn(4, 30, 22)
    logits, ret, attn = model(dummy_input)
    print("MarketTransformer Smoke Test:")
    print("Logits shape:", logits.shape)
    print("Expected return shape:", ret.shape)
    print("Attention weights shape:", attn.shape)
    sig = model.predict_signal(dummy_input[0:1])
    print("Sample Signal:", sig)
    print("Transformer test PASSED successfully!")
