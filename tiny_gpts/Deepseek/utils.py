# GPT2/utils.py
import os
import matplotlib.pyplot as plt
from datetime import datetime
import re
import torch
def create_run_dir(base_dir="runs", name="train"):

    os.makedirs(base_dir, exist_ok=True)

    try:
        existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    except Exception as e:
        print(f"❌ Klasör listelenemedi: {e}")
        raise

    numbers = []
    for d in existing_dirs:
        if d == name:
            numbers.append(1)  # train → 1
        elif re.fullmatch(f"{name}\\d+", d):  # train2, train3, ...
            try:
                num = int(re.findall(f"{name}(\\d+)", d)[0])
                numbers.append(num)
            except:
                continue

    if not numbers:
        new_number = 1
    else:
        new_number = max(numbers) + 1

    if new_number == 1:
        run_dir = os.path.join(base_dir, name)
    else:
        run_dir = os.path.join(base_dir, f"{name}{new_number}")

    try:
        os.makedirs(run_dir, exist_ok=False)
        print(f"✅ Yeni klasör oluşturuldu: {run_dir}")
    except FileExistsError:
        print(f"❌ Hata: '{run_dir}' zaten var ve exist_ok=False")
        return create_run_dir(base_dir, name)
    except Exception as e:
        print(f"❌ Klasör oluşturulamadı: {e}")
        raise

    return run_dir

def save_loss_plot(train_losses, run_dir):
    print(f"📉 Grafik kaydetmeye çalışıyor: {run_dir}")
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Training Loss", color="blue")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(run_dir, "loss_curve.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"✅ Loss grafiği kaydedildi: {plot_path}")
    except Exception as e:
        print(f"❌ Grafik kaydedilemedi: {e}")


def save_training_log(config, train_losses, run_dir, model_path):
    print(f"📄 Log kaydetmeye çalışıyor: {run_dir}")
    try:
        log_content = f"""
        # DeepSeek Training Log
        Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        ## Model Architecture
        vocab_size: {config.vocab_size}
        hidden_size: {config.hidden_size}
        intermediate_size: {config.intermediate_size}
        num_hidden_layers: {config.num_hidden_layers}
        num_attention_heads: {config.num_attention_heads}
        num_key_value_heads: {config.num_key_value_heads}
        max_position_embeddings: {config.max_position_embeddings}
        rms_norm_eps: {config.rms_norm_eps}
        rope_theta: {config.rope_theta}
        attention_bias: {config.attention_bias}
        tie_word_embeddings: {config.tie_word_embeddings}

        ## Training Configuration
        epochs: {config.epochs}
        batch_size: {config.batch_size}
        learning_rate: {config.learning_rate}
        device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}

        ## Training Results
        Final Loss: {train_losses[-1]:.6f}
        Total Steps: {len(train_losses)}
        Average Loss: {sum(train_losses) / len(train_losses):.6f}
        Minimum Loss: {min(train_losses):.6f}
        Model Saved: {model_path}

        ## Hardware Info
        CUDA Available: {torch.cuda.is_available()}
        GPU Name: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}
        """.strip()

        log_path = os.path.join(run_dir, "training_log.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(log_content)
        print(f"✅ Log kaydedildi: {log_path}")
    except Exception as e:
        print(f"❌ Log kaydedilemedi: {e}")