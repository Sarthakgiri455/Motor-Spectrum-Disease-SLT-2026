import torch
from torch import nn
from transformers import WhisperModel


class SharedWhisperMultiTaskModel(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        freeze_encoder: bool = True,
        unfreeze_last_n_layers: int = 0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        whisper = WhisperModel.from_pretrained(model_name)
        self.encoder = whisper.encoder
        hidden_size = whisper.config.d_model

        self.ddk_head = self._build_head(hidden_size, dropout)
        self.vowel_head = self._build_head(hidden_size, dropout)
        self.read_head = self._build_head(hidden_size, dropout)

        if freeze_encoder:
            self.freeze_encoder()
            if unfreeze_last_n_layers > 0:
                self.unfreeze_last_layers(unfreeze_last_n_layers)

    @staticmethod
    def _build_head(hidden_size: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_last_layers(self, n_layers: int) -> None:
        layers = getattr(self.encoder, "layers", None)
        if layers is None:
            return
        for layer in layers[-n_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    def forward(self, waveform: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        # Whisper's encoder consumes log-Mel input features; the collator converts
        # raw waveforms before calling the model.
        encoder_outputs = self.encoder(input_features=waveform)
        pooled = encoder_outputs.last_hidden_state.mean(dim=1)

        logits = pooled.new_zeros((pooled.size(0),), dtype=pooled.dtype)
        heads = {
            0: self.ddk_head,
            1: self.vowel_head,
            2: self.read_head,
        }

        for task_value, head in heads.items():
            mask = task_id == task_value
            if mask.any():
                logits[mask] = head(pooled[mask]).squeeze(-1)

        return logits
