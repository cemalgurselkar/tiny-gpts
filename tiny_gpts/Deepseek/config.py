from dataclasses import dataclass
@dataclass
class DeepSeekConfig:
    vocab_size: int = 10000
    hidden_size: int = 512
    intermediate_size: int = 1376
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    tie_word_embeddings: bool = True
    learning_rate:float = 1e-4
    batch_size: int =4
    epochs: int = 5
    
    @classmethod
    def tiny(cls):
        return cls()

    @classmethod
    def micro(cls):
        return cls(
            hidden_size=256,
            intermediate_size=688,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )