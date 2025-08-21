import torch
from transformers import GPT2Tokenizer

class Chatbot:

    def __init__(self, model_path, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        from .model import GPT
        self.model = GPT(config)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.to(self.device)
        self.model.eval()
    
    def generate(self, prompt, max_new_token = 100, temperature=0.8, top_k = 50):
        encoded = self.tokenizer.encode(prompt, return_tensor='.pt').to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                encoded,
                max_new_token=max_new_token,
                temperature=temperature,
                top_k=top_k
            )
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
    
    def chat_loop(self, max_new_token=100, temperature=0.8, top_k=50):
        print("Chatbot is starting..... If you wish, you can exit by typing 'quit' !!")
        while True:
            user_input = input("Question: ").strip()
            if user_input.lower() in ["quit", "exit"]:
                break
            prompt = f"Question: {user_input} Answer:"
            response = self.generate(prompt, max_new_token, temperature, top_k)
            answer = response[len(prompt):].strip()
            print(f"Answer: {answer}")
    
    @classmethod
    def from_model(cls, model_path, **config_kwargs):
        from .config import GPTConfig
        config = GPTConfig(**config_kwargs)
        return cls(model_path, config)