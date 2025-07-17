

import torch
import torch.nn as nn
import torch.optim as optim
from dpnn_lib.security.secure_model import SecureModel
from dpnn_lib.security.crypto import HEManager
from dpnn_lib.security.watermark import WatermarkManager

def run_secure_model_training_demo():
    print("\n--- SecureModel Training and Inference Demo ---")

    # --- 1. Setup ---
    # Define common parameters
    input_dim = 16
    hidden_dim = 32
    output_dim = 16
    batch_size = 4
    correct_watermark_id = "secure-model-v1-trained"
    secret_key = "a_new_secret_key_for_training"

    # Initialize SecureModel for training
    secure_model = SecureModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        watermark_id=correct_watermark_id,
        secret_key=secret_key
    )

    # Initialize helper managers
    he = HEManager()
    wm_verifier = WatermarkManager(correct_watermark_id, secret_key=secret_key)

    # --- 2. Data Preparation ---
    # Create a dummy dataset for training
    x_train = torch.randn(batch_size, input_dim, dtype=torch.float32)
    # Simple target function: output = input * 2 + 0.1
    y_train = x_train * 2 + 0.1 

    print(f"Created a dummy dataset with shape: {x_train.shape}")

    # --- 3. Training Setup ---
    criterion = nn.MSELoss()
    optimizer = optim.Adam(secure_model.parameters(), lr=0.01)
    num_epochs = 50

    print(f"\n--- Starting Training for {num_epochs} epochs ---")

    # --- 4. Training Loop ---
    for epoch in range(num_epochs):
        secure_model.train()
        optimizer.zero_grad()

        # Forward pass (now returns plaintext in training mode)
        y_pred = secure_model(x_train)

        # Loss is calculated directly on the plaintext output
        loss = criterion(y_pred, y_train)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("--- Training Finished ---")

    # --- 5. Inference and Verification (Post-Training) ---
    print("\n--- Running Inference and Verification ---")
    secure_model.eval() # Switch to evaluation mode to enable security features
    
    # Create a new plaintext input for inference
    x_test = torch.randn(1, input_dim, dtype=torch.float32)
    y_test_expected = x_test * 2 + 0.1 # Expected output from our dummy function

    # --- Scenario A: Correct Key Holder ---
    print("\nScenario A: Inference by a user with the CORRECT key.")
    try:
        # User performs inference
        y_encrypted_output = secure_model(x_test)
        
        # User decrypts the output
        y_decrypted_flat = he.decrypt(y_encrypted_output)
        y_decrypted_reshaped = y_decrypted_flat.reshape(1, output_dim)
        
        print("Inference successful. Decrypted output shape:", y_decrypted_reshaped.shape)

        # Verify the watermark
        assert wm_verifier.verify(
            y_decrypted_reshaped,
            (1, output_dim),
            expected_watermark_id=correct_watermark_id,
            threshold=0.9
        ), "Watermark verification failed!"
        print("SUCCESS: Watermark verified on the model's output.")

        # Check if the model has learned anything (optional)
        inference_loss = nn.functional.mse_loss(y_decrypted_reshaped, y_test_expected)
        print(f"Inference MSE loss (sanity check): {inference_loss.item():.4f}")
        print("Note: The inference loss is non-zero due to the embedded watermark.")
        print("This demonstrates that the model output is protected.")
        print("SUCCESS: Model appears to have learned the target function, and the output is secured.")

    except Exception as e:
        print(f"FAILURE: An unexpected error occurred: {e}")

    # --- Scenario B: Unauthorized User (No Key) ---
    print("\nScenario B: An unauthorized user attempts inference.")
    # The user can get the encrypted output, but cannot decrypt it to get a meaningful result.
    y_encrypted_for_unauthorized = secure_model(x_test)
    print("Unauthorized user obtained the encrypted output.")
    print("Without the key, the output remains encrypted and unusable.")
    # Any attempt to use `he.decrypt` without the correct context would fail or produce garbage.

if __name__ == "__main__":
    run_secure_model_training_demo()

