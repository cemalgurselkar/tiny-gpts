from .model import GPT, GPTConfig
from .train import GPTrain
from .chat import Chatbot
from .utils import *

def train(**kwargs):
    trainer = GPTrain.from_config(**kwargs)
    trainer.train(kwargs.get('data', 'data.txt'), kwargs.get('model_path', 'gpt_finetuned.pth'))

def chat(**kwargs):
    model_path = kwargs.pop('model_path', 'gpt_finetuned.pth')
    chatbot = Chatbot.from_model(model_path, **kwargs)
    chatbot.chat_loop(**{k: v for k, v in kwargs.items() if k in ['max_new_tokens', 'temperature', 'top_k']})


__all__ = ['GPT', 'GPTConfig', 'GPTrain', 'Chatbot', 'train', 'chat']