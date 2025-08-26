import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os
from transformers import GPT2Tokenizer
from .model import DeepSeekCasualLM
from .config import DeepSeekConfig
from .utils import *

class DeepSeek:

    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using Device: {self.device}")

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = DeepSeekCasualLM(config)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.1
        )

        self.model.to(self.device)
        self.scaler = torch.cuda.amp.GradScaler() if self.device == 'cuda' else None

    def _load_data(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        tokens = self.tokenizer.encode(text)
        tokens = [min(t, self.config.vocab_size - 1) for t in tokens]
        seq_len = min(self.config.max_position_embeddings, 256)
        xs, ys = [], []

        for i in range(0, len(tokens) - seq_len, seq_len//2):
            chunk = tokens[i:i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                xs.append(chunk[:-1])
                ys.append(chunk[1:])
        print(f"Generated {len(xs)} training samples")
        return xs, ys
    
 
    def train(self, data_path, model_path="deepseek_model.pth"):
        print(f"Loading data from {data_path}")

        try:
            run_dir = create_run_dir()
            print(f"Run directory created: {run_dir}")
        except Exception as e:
            print(f"Run directory creation failed: {e}")
            run_dir = "."
        
        xs, ys = self._load_data(data_path)

        xs_tensor = torch.tensor(xs, dtype=torch.long)
        ys_tensor = torch.tensor(ys, dtype=torch.long)

        dataset = TensorDataset(xs_tensor, ys_tensor)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        print(f"Starting training for {self.config.epochs} epochs...")
        self.model.train()
        train_losses = []

        for epoch in range(self.config.epochs):
            total_loss = 0
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")

            for batch_idx, (x,y) in enumerate(pbar):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                if self.device == 'cuda':
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        outputs = self.model(input_ids=x, labels=y)
                        loss = outputs['loss']
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(input_ids=x, labels=y)
                    loss = outputs['loss']
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)

                pbar.set_postfix({"Loss": f"{loss.item():.5f}", "Avg Loss": f"{avg_loss:5f}"})
                train_losses.append(loss.item())

                if self.device == 'cuda' and batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
            print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    
        save_loss_plot(train_losses, run_dir)
        save_training_log(self.config, train_losses, run_dir, 
                         model_path=os.path.join(run_dir, "deepseek_finetuned.pth"))

        model_save_path = os.path.join(run_dir, model_path)
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Training complete. Model saved to {model_save_path}")

    @classmethod
    def from_config(cls, **kwargs):
        config = DeepSeekConfig(**kwargs)
        return cls(config)