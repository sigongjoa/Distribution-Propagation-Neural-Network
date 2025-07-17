import hashlib
import torch

class WatermarkManager:
    def __init__(self, watermark_id: str):
        self.watermark_id = watermark_id
        self.hash_length = 32  # SHA256 produces a 32-byte hash

    def _generate_watermark(self, tensor: torch.Tensor) -> torch.Tensor:
        """Generates a watermark based on the tensor content and watermark_id."""
        # For simplicity, we'll hash the tensor's flattened content and the watermark_id.
        # In a real scenario, this would be more sophisticated.
        hasher = hashlib.sha256()
        # Detach the tensor from the computation graph before converting to numpy
        hasher.update(tensor.detach().contiguous().cpu().numpy().tobytes())
        hasher.update(self.watermark_id.encode('utf-8'))
        # Convert hash digest to a torch tensor of appropriate dtype
        return torch.tensor(list(hasher.digest()), dtype=torch.float32) # Use float32 to match common tensor dtypes

    def embed(self, tensor: torch.Tensor) -> torch.Tensor:
        """Embeds a watermark into the tensor by overwriting a portion of it."""
        if tensor.numel() < self.hash_length:
            raise ValueError(f"Tensor too small to embed watermark. Requires at least {self.hash_length} elements.")

        original_shape = tensor.shape
        flat_tensor = tensor.flatten()
        watermark = self._generate_watermark(flat_tensor[self.hash_length:]) # Generate watermark from the data part
        
        # Overwrite the first `hash_length` elements with the watermark
        # Ensure watermark dtype matches tensor dtype
        marked_flat_tensor = flat_tensor.clone()
        marked_flat_tensor[:self.hash_length] = watermark.to(marked_flat_tensor.dtype)
        
        return marked_flat_tensor.reshape(original_shape)

    def verify(self, marked_tensor: torch.Tensor) -> bool:
        """Verifies the watermark in the tensor."""
        if marked_tensor.numel() < self.hash_length:
            return False # Cannot verify if tensor is too small

        flat_marked_tensor = marked_tensor.flatten()
        
        # Extract the embedded watermark (first `hash_length` elements)
        extracted_watermark = flat_marked_tensor[:self.hash_length]
        
        # Generate the expected watermark from the rest of the tensor data
        # This assumes the watermark was embedded by overwriting the beginning.
        data_part = flat_marked_tensor[self.hash_length:]
        expected_watermark = self._generate_watermark(data_part)
        
        # Compare the extracted and expected watermarks
        return torch.equal(extracted_watermark, expected_watermark.to(extracted_watermark.dtype))
