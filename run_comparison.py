import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
import math

# Import models
from dpnn_transformer_model import DPNNTransformerModel
from standard_transformer_model import StandardTransformerModel
from dpnn.core.config import Preset # Import Preset enum

# Dummy data and tokenizer (for demonstration purposes)
class DummyTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.id_to_token = {i: str(i) for i in range(vocab_size)}
        self.token_to_id = {str(i): i for i in range(vocab_size)}

    def encode(self, text):
        return [self.token_to_id.get(token, 0) for token in text.split()]

    def decode(self, token_ids):
        return " ".join([self.id_to_token.get(token_id, '<unk>') for token_id in token_ids])

def get_dummy_data(vocab_size, batch_size, seq_len, device):
    # Generate random token IDs
    data = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # For language modeling, target is usually the next token
    target = torch.roll(data, shifts=-1, dims=-1)
    return data, target

# Hyperparameters
vocab_size = 1000
d_model = 128 # Reduced for faster testing
num_heads = 4
d_ff = 256
num_blocks = 2
batch_size = 32
seq_len = 50
epochs = 5
lr = 0.001
log_dir = "runs" # TensorBoard log directory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_evaluate(model_name, model, optimizer, criterion, writer, log_file):
    print(f"\n--- Training and Evaluating {model_name} ---")
    log_file.write(f"\n--- Training and Evaluating {model_name} ---\n")

    model.train()
    total_loss = 0
    start_time = time.time()
    max_memory_allocated = 0

    for epoch in range(1, epochs + 1):
        data, target = get_dummy_data(vocab_size, batch_size, seq_len, device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # For language modeling, reshape output and target for CrossEntropyLoss
        # output: (batch_size, seq_len, vocab_size)
        # target: (batch_size, seq_len)
        loss = criterion(output.reshape(-1, vocab_size), target.reshape(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        if device.type == 'cuda':
            current_memory = torch.cuda.memory_allocated(device) / (1024**2) # MB
            max_memory_allocated = max(max_memory_allocated, torch.cuda.max_memory_allocated(device) / (1024**2))
            writer.add_scalar(f'Memory/Allocated_MB/{model_name}', current_memory, epoch)
            writer.add_scalar(f'Memory/Max_Allocated_MB/{model_name}', max_memory_allocated, epoch)

        # Log to TensorBoard
        writer.add_scalar(f'Loss/{model_name}', loss.item(), epoch)
        
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}", end='')
        if device.type == 'cuda':
            print(f", Mem: {current_memory:.2f} MB")
        else:
            print()
        log_file.write(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}\n")

    avg_loss = total_loss / epochs
    end_time = time.time()
    time_taken = end_time - start_time

    # Calculate Perplexity (for language models)
    # PPL = exp(average negative log-likelihood)
    # For CrossEntropyLoss, it's exp(average loss)
    perplexity = math.exp(avg_loss)

    print(f"--- {model_name} Training Summary ---")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity (PPL): {perplexity:.2f}")
    print(f"Total Training Time: {time_taken:.2f} seconds")
    if device.type == 'cuda':
        print(f"Max GPU Memory Allocated: {max_memory_allocated:.2f} MB")

    log_file.write(f"--- {model_name} Training Summary ---\n")
    log_file.write(f"Average Loss: {avg_loss:.4f}\n")
    log_file.write(f"Perplexity (PPL): {perplexity:.2f}\n")
    log_file.write(f"Total Training Time: {time_taken:.2f} seconds\n")
    if device.type == 'cuda':
        log_file.write(f"Max GPU Memory Allocated: {max_memory_allocated:.2f} MB\n")

    writer.add_scalar(f'Metrics/{model_name}_Average_Loss', avg_loss, 0)
    writer.add_scalar(f'Metrics/{model_name}_Perplexity', perplexity, 0)
    writer.add_scalar(f'Metrics/{model_name}_Training_Time', time_taken, 0)
    if device.type == 'cuda':
        writer.add_scalar(f'Metrics/{model_name}_Max_GPU_Memory_MB', max_memory_allocated, 0)

# Main comparison script
if __name__ == "__main__":
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Setup TensorBoard writers
    standard_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'standard_transformer'))
    dpnn_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'dpnn_transformer'))

    # Setup log files
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    standard_run_name = f"standard_transformer_comparison_{timestamp}"
    dpnn_run_name = f"dpnn_transformer_comparison_{timestamp}"
    standard_log_path = os.path.join("results", f"{standard_run_name}.log")
    dpnn_log_path = os.path.join("results", f"{dpnn_run_name}.log")

    os.makedirs("results", exist_ok=True) # Ensure results directory exists

    with open(standard_log_path, "w") as standard_log_file, \
         open(dpnn_log_path, "w") as dpnn_log_file:

        # Standard Transformer
        standard_model = StandardTransformerModel(vocab_size, d_model, num_heads, d_ff, num_blocks).to(device)
        standard_optimizer = optim.Adam(standard_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss() # Standard loss for language modeling

        train_and_evaluate("Standard_Transformer", standard_model, standard_optimizer, criterion, standard_writer, standard_log_file)

        # DPNN Transformer
        dpnn_model = DPNNTransformerModel(vocab_size, d_model, num_heads, d_ff, num_blocks,
                                        preset=Preset.BALANCED, k_top=min(8, seq_len)).to(device)
        dpnn_optimizer = optim.Adam(dpnn_model.parameters(), lr=lr)
        
        train_and_evaluate("DPNN_Transformer", dpnn_model, dpnn_optimizer, criterion, dpnn_writer, dpnn_log_file)

    standard_writer.close()
    dpnn_writer.close()

    print("\nComparison complete. Run 'tensorboard --logdir runs' to view results.")
    print(f"Detailed logs saved to: {standard_log_path} and {dpnn_log_path}")