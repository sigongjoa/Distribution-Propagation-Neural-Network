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
        # 1. Watermark input (optional, but good for demonstrating input integrity)
        # For this PoC, we'll watermark the output of the internal model.
        
        # 2. Encrypt input
        # The input to the SecureModel is assumed to be plaintext.
        # We encrypt it here to simulate a secure input channel.
        x_encrypted = self.he.encrypt(x_plain_input)
        
        # 3. Decrypt for internal plaintext computation
        # In a fully HE-compatible model, this step would be avoided.
        x_decrypted_flat = self.he.decrypt(x_encrypted)
        x_decrypted = x_decrypted_flat.reshape(self.input_shape)
        
        # 4. Internal model computation (on plaintext)
        y_plain_output = self.internal_model(x_decrypted)
        
        # 5. Embed watermark into the plaintext output
        y_marked_plain = self.wm.embed(y_plain_output)
        
        # 6. Encrypt the watermarked output
        y_encrypted_output = self.he.encrypt(y_marked_plain)
        
        # 7. Verification is now done externally in the demo script
        # assert self.wm.verify(y_decrypted_for_wm_reshaped, self.output_shape, self.watermark_id), \
        #     "Invalid watermark detected in SecureModel output!"
            
        return y_encrypted_output