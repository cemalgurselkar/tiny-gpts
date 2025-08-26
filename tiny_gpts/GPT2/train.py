# GPT2/train.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import GPT2Tokenizer
from .model import GPT, GPTConfig
from .utils import create_run_dir, save_loss_plot, save_training_log
import os


class GPTrain:
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT(config)
        self.optimizer = self.model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=config.learning_rate,
            betas=(0.9, 0.999),
            device=self.device
        )
        self.model.to(self.device)
        self.autocast_dtype = torch.float16 if self.device == 'cuda' else torch.float32

    def load_data(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = self.tokenizer.encode(text)
        seq_len = self.config.block_size
        xs, ys = [], []
        for i in range(0, len(tokens) - seq_len, seq_len):
            chunk = tokens[i:i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                xs.append(chunk[:-1])
                ys.append(chunk[1:])
        return xs, ys

    def train(self, data_path, model_path="model.pth"):
        print(f"Loading data from {data_path}...")
        try:
            run_dir = create_run_dir()
            print(f"Run file is created: {run_dir}")
        except Exception as e:
            print("Run dosyası oluşturulamadı!!")
            

        print(f"Data is loading: {data_path}")

        xs, ys = self.load_data(data_path)
        dataset = TensorDataset(torch.tensor(xs), torch.tensor(ys))
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        print(f"Starting training for {self.config.epochs} epochs...")
        self.model.train()
        train_losses = []
        
        for epoch in range(self.config.epochs):
            pbar = tqdm(loader)
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                with torch.autocast(device_type=self.device, dtype=self.autocast_dtype):
                    _, loss = self.model(x, targets=y)

                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())
                pbar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        save_loss_plot(train_losses, run_dir)
        save_training_log(self.config, train_losses, run_dir, model_path=os.path.join(run_dir,"gpt_finetuned.pth"))

        print(f"Training complete. Saving model to {model_path}")
        model_save_path = os.path.join(run_dir, "model.pth")
        torch.save(self.model.state_dict(), model_path)


    @classmethod
    def from_config(cls, **kwargs):
        config = GPTConfig(**kwargs)
        return cls(config)