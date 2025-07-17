import torch
from torch import nn
from dpnn_lib.security.security_cell import SecurityCell
from dpnn_lib.security.watermark import WatermarkManager
from dpnn_lib.security.crypto import HEManager

def run_validation_demo():
    print("\n--- Security Cell Validation Demo ---")

    # 1) Define a base cell and wrap it with SecurityCell
    base_cell = nn.Linear(16, 16)
    correct_watermark_id = "my-secret-model-key"
    sec_cell = SecurityCell(base_cell, watermark_id=correct_watermark_id)
    
    wm_correct = WatermarkManager(correct_watermark_id)
    wm_invalid_tampered = WatermarkManager("wrong-key-or-tampered") # Simulates a tampered key
    wm_invalid_different = WatermarkManager("another-random-key") # Simulates an explicitly different key
    he = HEManager()

    # Prepare a plain input tensor
    x_plain = torch.randn(8, 16, dtype=torch.float32)

    # --- Scenario 1: Valid Encrypted Key (Correct Watermark) ---
    print("\nScenario 1: Input with CORRECT encrypted key (watermark)")
    x_enc_valid = he.encrypt(x_plain)
    x_marked_valid = wm_correct.embed(x_enc_valid)

    try:
        print("Attempting to process input with correct key...")
        y_marked_valid = sec_cell(x_marked_valid)
        print("SUCCESS: Input processed. Output shape:", y_marked_valid.shape)
        # Optionally, verify output watermark and decrypt
        assert wm_correct.verify(y_marked_valid), "Output watermark verification failed!"
        y_dec_valid = he.decrypt(y_marked_valid)
        print("Output decrypted and verified.")
    except AssertionError as e:
        print(f"FAILURE: Unexpected error for valid input: {e}")
    except Exception as e:
        print(f"FAILURE: An unexpected error occurred: {e}")

    # --- Scenario 2: Invalid Encrypted Key (Tampered Watermark) ---
    print("\nScenario 2: Input with INCORRECT encrypted key (Tampered Watermark)")
    x_enc_invalid_tampered = he.encrypt(x_plain) # Encrypt the same plain tensor
    x_marked_invalid_tampered = wm_invalid_tampered.embed(x_enc_invalid_tampered) # Embed with a different, invalid watermark

    try:
        print("Attempting to process input with tampered key...")
        y_marked_invalid_tampered = sec_cell(x_marked_invalid_tampered)
        print("FAILURE: Input processed unexpectedly with tampered key. Output shape:", y_marked_invalid_tampered.shape)
    except AssertionError as e:
        print(f"SUCCESS: Caught expected error: {e}")
        print("Input with tampered key was correctly rejected.")
    except Exception as e:
        print(f"FAILURE: An unexpected error occurred: {e}")

    # --- Scenario 3: Invalid Encrypted Key (Explicitly Different Watermark) ---
    print("\nScenario 3: Input with EXPLICITLY DIFFERENT encrypted key (Watermark)")
    x_enc_invalid_different = he.encrypt(x_plain) # Encrypt the same plain tensor
    x_marked_invalid_different = wm_invalid_different.embed(x_enc_invalid_different) # Embed with a completely different watermark ID

    try:
        print("Attempting to process input with explicitly different key...")
        y_marked_invalid_different = sec_cell(x_marked_invalid_different)
        print("FAILURE: Input processed unexpectedly with explicitly different key. Output shape:", y_marked_invalid_different.shape)
    except AssertionError as e:
        print(f"SUCCESS: Caught expected error: {e}")
        print("Input with explicitly different key was correctly rejected.")
    except Exception as e:
        print(f"FAILURE: An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_validation_demo()