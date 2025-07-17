import torch
from dpnn_lib.security.secure_model import SecureModel
from dpnn_lib.security.crypto import HEManager
from dpnn_lib.security.watermark import WatermarkManager

# Define common parameters
input_dim = 16
hidden_dim = 32
output_dim = 16
correct_watermark_id = "secure-model-v1"
secret_key = "super_secret_key_for_model"

# Initialize SecureModel
secure_model = SecureModel(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    watermark_id=correct_watermark_id,
    secret_key=secret_key
)

he = HEManager()
wm_verifier = WatermarkManager(correct_watermark_id, secret_key=secret_key)

# Prepare a plaintext input
x_plain = torch.randn(1, input_dim, dtype=torch.float32) # Batch size of 1

print("\n--- SecureModel Demo ---")

# --- Scenario 1: Forward pass with correct setup ---
print("\nScenario 1: Processing input with correctly configured SecureModel")
try:
    y_encrypted_output = secure_model(x_plain)
    
    # Decrypt the output for verification
    y_decrypted_flat = he.decrypt(y_encrypted_output)
    y_decrypted_reshaped = y_decrypted_flat.reshape(1, output_dim)
    
    print("SecureModel processed input. Decrypted output shape:", y_decrypted_reshaped.shape)
    
    # Verify watermark on the decrypted output
    assert wm_verifier.verify(
        y_decrypted_reshaped,
        (1, output_dim),
        expected_watermark_id=correct_watermark_id,
        threshold=0.9
    ), "Watermark verification failed on SecureModel output!"
    print("SUCCESS: Watermark verified on SecureModel output.")

except AssertionError as e:
    print(f"FAILURE: Assertion error: {e}")
except Exception as e:
    print(f"FAILURE: An unexpected error occurred: {e}")

# --- Scenario 2: Attempt to verify with incorrect watermark ID ---
print("\nScenario 2: Attempting to verify output with incorrect watermark ID")
incorrect_wm_verifier = WatermarkManager("wrong-model-id", secret_key=secret_key)

try:
    # Use the encrypted output from Scenario 1
    y_decrypted_flat = he.decrypt(y_encrypted_output)
    y_decrypted_reshaped = y_decrypted_flat.reshape(1, output_dim)
    
    # This should fail because the wm_verifier is initialized with a different ID
    assert not incorrect_wm_verifier.verify(
        y_decrypted_reshaped,
        (1, output_dim),
        expected_watermark_id="wrong-model-id",
        threshold=0.9
    ), "Verification passed with incorrect ID unexpectedly!"
    print("SUCCESS: Verification with incorrect ID correctly failed.")

except AssertionError as e:
    print(f"FAILURE: Unexpected assertion: {e}")
except Exception as e:
    print(f"FAILURE: An unexpected error occurred: {e}")

# --- Scenario 3: Attempt to tamper with decrypted output and verify ---
print("\nScenario 3: Attempting to tamper with decrypted output and verify")

try:
    # Decrypt the output from Scenario 1
    y_decrypted_flat_original = he.decrypt(y_encrypted_output)
    y_decrypted_reshaped_original = y_decrypted_flat_original.reshape(1, output_dim)
    
    # Tamper with the decrypted output
    y_tampered = y_decrypted_reshaped_original.clone()
    y_tampered[0, 0] += 100.0 # Significant change
    
    # This should fail because the output has been tampered with
    assert not wm_verifier.verify(
        y_tampered,
        (1, output_dim),
        expected_watermark_id=correct_watermark_id,
        threshold=0.9
    ), "Watermark verification passed on tampered output unexpectedly!"
    print("SUCCESS: Watermark verification correctly failed on tampered output.")

except AssertionError as e:
    print(f"FAILURE: Unexpected assertion: {e}")
except Exception as e:
    print(f"FAILURE: An unexpected error occurred: {e}")