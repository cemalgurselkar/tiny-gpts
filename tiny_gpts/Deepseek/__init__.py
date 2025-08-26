from .config import DeepSeekConfig
from .model import DeepSeekCasualLM
from .utils import *
from .train import DeepSeek

def train(**kwargs):
    """One-line training function"""
    trainer = DeepSeek.from_config(**kwargs)
    trainer.train(
        data_path=kwargs.get('data_path', 'data.txt'),
        model_path=kwargs.get('model_path', 'deepseek_model.pth')
    )

__all__ = [
    "DeepSeekConfig",
    "DeepSeekCasualLM", 
    "DeepSeek",
    "train"  # train fonksiyonunu da export et!
]