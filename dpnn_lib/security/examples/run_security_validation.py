import torch
from torch import nn
from dpnn_lib.security.security_cell import SecurityCell
from dpnn_lib.security.watermark import WatermarkManager
from dpnn_lib.security.crypto import HEManager
import tenseal as ts

def run_validation_demo():
    print("\n--- Security Cell Validation Demo (with TenSEAL HE and Robust Watermarking) ---")

    # Define common parameters
    input_shape = (8, 16) # Example input shape for the base cell
    correct_watermark_id = "my-secret-model-key"
    secret_key = "my_super_secret_key"

    # 1) Define a base cell
    base_cell = nn.Linear(input_shape[-1], input_shape[-1])

    # --- Scenario 1: Valid SecurityCell setup and input ---
    print("\nScenario 1: SecurityCell initialized with CORRECT watermark ID and valid input")
    sec_cell_valid = SecurityCell(base_cell, watermark_id=correct_watermark_id, input_shape=input_shape, secret_key=secret_key)
    he = HEManager()

    x_plain_valid = torch.randn(input_shape, dtype=torch.float32)

    try:
        print("Attempting to process input with correctly configured SecurityCell...")
        y_encrypted_output_valid = sec_cell_valid(x_plain_valid)
        
        # Decrypt and verify the output outside the SecurityCell for demonstration
        y_decrypted_flat_valid = he.decrypt(y_encrypted_output_valid)
        y_decrypted_reshaped_valid = y_decrypted_flat_valid.reshape(input_shape)
        
        wm_verifier = WatermarkManager(correct_watermark_id, secret_key=secret_key)
        assert wm_verifier.verify(y_decrypted_reshaped_valid, input_shape, correct_watermark_id), "Output watermark verification failed outside SecurityCell!"
        
        print("SUCCESS: Input processed and output watermark verified.")
        print("Output shape:", y_decrypted_reshaped_valid.shape)

    except AssertionError as e:
        print(f"FAILURE: Unexpected assertion for valid scenario: {e}")
    except Exception as e:
        print(f"FAILURE: An unexpected error occurred: {e}")

    # --- Scenario 2: Verification with an INCORRECT watermark ID ---
    print("\nScenario 2: Verification with an INCORRECT watermark ID")
    incorrect_watermark_id = "wrong-model-key"
    wm_incorrect_verifier = WatermarkManager(incorrect_watermark_id, secret_key=secret_key)

    try:
        print("Attempting to verify output with an incorrect watermark ID...")
        # Use the output from Scenario 1 (which has the correct watermark embedded)
        # This should fail as the watermark ID is different.
        assert not wm_incorrect_verifier.verify(y_decrypted_reshaped_valid, input_shape, incorrect_watermark_id), "Verification passed with incorrect ID unexpectedly!"
        print("SUCCESS: Verification with incorrect ID correctly failed.")
    except AssertionError as e:
        print(f"FAILURE: Unexpected assertion: {e}")
    except Exception as e:
        print(f"FAILURE: An unexpected error occurred: {e}")

    # --- Scenario 3: Verification after TAMPERING with the output ---
    print("\nScenario 3: Verification after TAMPERING with the output")
    
    # Create a tampered version of the decrypted output from Scenario 1
    y_tampered_output = y_decrypted_reshaped_valid.clone()
    y_tampered_output[0, 0] += 100.0 # Introduce a significant change

    try:
        print("Attempting to verify watermark on tampered output...")
        # This should fail if the watermark is robust enough to detect tampering.
        assert not wm_verifier.verify(y_tampered_output, input_shape, correct_watermark_id), "Watermark verification passed on tampered output unexpectedly!"
        print("SUCCESS: Watermark verification correctly failed on tampered output.")
    except AssertionError as e:
        print(f"FAILURE: Unexpected assertion: {e}")
    except Exception as e:
        print(f"FAILURE: An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_validation_demo()