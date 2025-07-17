import hashlib
import torch

class WatermarkManager:
    """
    Manages watermark embedding and verification.
    For PoC, uses a simple fixed pattern for embedding and direct comparison for verification.
    """
    def __init__(self, watermark_id: str, secret_key: str = "default_secret_key"):
        self.watermark_id = watermark_id
        self.secret_key = secret_key
        self.watermark_pattern_size = 16 # Size of the fixed pattern to embed
        self.watermark_value = 1.0 # The value to embed as a watermark

    def _generate_watermark_pattern(self, shape: tuple) -> torch.Tensor:
        """
        Generates a pseudo-random watermark pattern based on watermark_id and secret_key.
        The pattern will have the specified shape.
        """
        seed_str = self.watermark_id + self.secret_key
        seed = int(hashlib.sha256(seed_str.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)
        
        torch.manual_seed(seed)
        
        # Generate a random tensor of the given shape
        watermark_pattern = torch.randn(shape, dtype=torch.float32)
        
        return watermark_pattern

    def embed(self, plain_tensor: torch.Tensor) -> torch.Tensor:
        """
        Embeds a watermark into the plaintext tensor by setting a fixed pattern
        in the first few elements.
        """
        if plain_tensor.numel() < self.watermark_pattern_size:
            raise ValueError(f"Tensor too small to embed watermark. Requires at least {self.watermark_pattern_size} elements.")

        marked_tensor = plain_tensor.clone()
        # Flatten the tensor to embed the pattern at the beginning
        marked_flat = marked_tensor.flatten()
        
        # Generate a unique pattern based on watermark_id and secret_key
        unique_pattern = self._generate_watermark_pattern((self.watermark_pattern_size,))
        
        # Embed the unique pattern
        marked_flat[:self.watermark_pattern_size] = unique_pattern
        
        return marked_tensor # Return with original shape

    def verify(self, marked_tensor: torch.Tensor, original_input_shape: tuple, expected_watermark_id: str, threshold: float = 0.9) -> bool:
        """
        Verifies the presence of the watermark in the marked_tensor.
        Checks if the fixed pattern is present in the expected location.
        Includes a check for the expected watermark ID.
        """
        # First, check if the watermark_id of this WatermarkManager matches the expected_watermark_id.
        if self.watermark_id != expected_watermark_id:
            print(f"Watermark verification: Mismatch in watermark_id. Expected '{expected_watermark_id}', got '{self.watermark_id}'.")
            return False

        if marked_tensor.numel() < self.watermark_pattern_size:
            print("Watermark verification: Tensor too small to contain watermark.")
            return False

        # Extract the potential watermark pattern from the beginning of the flattened tensor
        extracted_pattern = marked_tensor.flatten()[:self.watermark_pattern_size]
        
        # Generate the expected unique pattern for comparison
        expected_pattern_segment = self._generate_watermark_pattern((self.watermark_pattern_size,))
        
        # Debug print to confirm sizes
        print(f"Debug: extracted_pattern.shape = {extracted_pattern.shape}, expected_pattern_segment.shape = {expected_pattern_segment.shape}")

        # Compare the extracted pattern with the expected pattern
        # Using cosine similarity for a more robust comparison than strict equality
        epsilon = 1e-8
        
        # Check for zero norm to prevent division by zero
        norm_extracted = torch.norm(extracted_pattern)
        norm_expected = torch.norm(expected_pattern_segment)
        
        if norm_extracted < epsilon or norm_expected < epsilon:
            correlation = 0.0 # Or handle as an error case
        else:
            correlation = torch.dot(extracted_pattern, expected_pattern_segment) / \
                          (norm_extracted * norm_expected + epsilon)
        
        # Explicitly convert to string to avoid formatting issues
        print("Watermark verification (ID: " + self.watermark_id + "): Correlation = " + str(correlation.item()) + ", Threshold = " + str(threshold))
        
        return correlation.item() > threshold