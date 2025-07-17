import torch
from torch import nn, Tensor
import tenseal as ts
from .crypto import HEManager
from .watermark import WatermarkManager

class SecureModel(nn.Module):
    """
    A conceptual secure neural network model that integrates HE and watermarking
    at the input and output boundaries.
    
    Internal layers operate on plaintext for simplicity in this PoC.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, watermark_id: str, secret_key: str):
        super().__init__()
        self.he = HEManager()
        self.wm = WatermarkManager(watermark_id, secret_key=secret_key)
        self.watermark_id = watermark_id
        
        # Define internal plaintext model
        self.internal_model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Store input and output shapes for watermarking/decryption
        self.input_shape = (1, input_dim) # Assuming batch size of 1 for simplicity in shape management
        self.output_shape = (1, output_dim)

    def forward(self, x_plain_input: Tensor) -> ts.CKKSVector:
        # If in training mode, bypass HE for gradient flow
        if self.training:
            # In a real scenario, the training entity has the keys and thus can
            # work with the plaintext model directly.
            return self.internal_model(x_plain_input)

        # In evaluation mode, enforce security measures
        # 1. Encrypt input to simulate secure channel
        x_encrypted = self.he.encrypt(x_plain_input)
        
        # 2. Decrypt for internal plaintext computation
        # This step is a simplification for the PoC.
        x_decrypted_flat = self.he.decrypt(x_encrypted)
        x_decrypted = x_decrypted_flat.reshape(x_plain_input.shape)
        
        # 3. Internal model computation
        y_plain_output = self.internal_model(x_decrypted)
        
        # 4. Embed watermark into the output
        y_marked_plain = self.wm.embed(y_plain_output)
        
        # 5. Encrypt the final watermarked output
        y_encrypted_output = self.he.encrypt(y_marked_plain)
        
        return y_encrypted_output