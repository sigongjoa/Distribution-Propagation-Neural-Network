dpnn_lib.security package
==========================

The `dpnn_lib.security` package provides foundational components for integrating
homomorphic encryption (HE) and watermarking into neural network operations.
This package aims to demonstrate the conceptual framework for building secure
and traceable deep learning models, particularly focusing on protecting model
inputs/outputs and ensuring model integrity.

Submodules
----------

dpnn_lib.security.crypto module
-------------------------------

.. automodule:: dpnn_lib.security.crypto
   :members:
   :show-inheritance:
   :undoc-members:

The `crypto` module provides a conceptual interface for Homomorphic Encryption (HE)
operations. It leverages the TenSEAL library to perform approximate encryption
and decryption of tensors, simulating a secure computation environment.

**Key Features:**

*   **`HEManager` Class:** Manages the TenSEAL CKKS context, including parameter
    setup (polynomial modulus degree, coefficient modulus bit sizes, global scale)
    and key generation.
*   **`encrypt(plain: torch.Tensor) -> tenseal.CKKSVector`:** Encrypts a PyTorch
    tensor into a TenSEAL CKKSVector. For simplicity in this PoC, tensors are
    flattened before encryption.
*   **`decrypt(cipher: tenseal.CKKSVector) -> torch.Tensor`:** Decrypts a TenSEAL
    CKKSVector back into a PyTorch tensor. Due to the nature of HE, decryption
    yields an approximate result, and the original shape information needs to be
    managed externally.

**Note on PoC Implementation:**
While `HEManager` uses TenSEAL for actual HE operations, the current integration
within `SecurityCell` and `SecureModel` involves decryption for internal plaintext
computation. A fully HE-compatible model would perform computations directly
on encrypted data, which is a more advanced and complex topic.

dpnn_lib.security.security_cell module
--------------------------------------

.. automodule:: dpnn_lib.security.security_cell
   :members:
   :show-inheritance:
   :undoc-members:

The `security_cell` module defines the `SecurityCell` class, a specialized
neural network cell designed to integrate security features (encryption and
watermarking) around a base neural network layer. It acts as a wrapper that
enforces security policies during the forward pass.

**Key Features:**

*   **Input Encryption/Decryption:** Encrypts the input before passing it to the
    `base_cell` and decrypts the output for watermarking and re-encryption.
    (Note: In this PoC, internal `base_cell` operations are on plaintext).
*   **Output Watermarking:** Embeds a unique watermark pattern into the plaintext
    output of the `base_cell` before final encryption.
*   **Watermark Verification:** Verifies the presence and integrity of the watermark
    in the decrypted output. If the watermark is invalid or tampered with, an
    `AssertionError` is raised, preventing unauthorized processing.
*   **`__init__(base_cell: nn.Module, watermark_id: str, input_shape: tuple, secret_key: str)`:**
    Initializes the `SecurityCell` with a base neural network module, a unique
    watermark identifier, the expected input shape, and a secret key for watermarking.
*   **`forward(x_plain_input: torch.Tensor) -> tenseal.CKKSVector`:**
    Processes a plaintext input tensor. It performs watermarking, encryption,
    passes data through the base cell (conceptually decrypted), re-watermarks,
    re-encrypts, and verifies the watermark before returning an encrypted output.

**Role in Secure Pipeline:**
`SecurityCell` serves as a building block for constructing secure neural networks,
ensuring that data processed through it adheres to defined security policies
and that the model's output is traceable.

dpnn_lib.security.secure_model module
-------------------------------------

.. automodule:: dpnn_lib.security.secure_model
   :members:
   :show-inheritance:
   :undoc-members:

The `secure_model` module introduces the `SecureModel` class, which represents
a conceptual end-to-end secure neural network. It demonstrates how Homomorphic
Encryption and watermarking can be applied at the boundaries of an entire model,
rather than at each individual layer.

**Key Features:**

*   **Boundary Security:** Encrypts the model's input and watermarks/encrypts its
    final output. This simulates a scenario where data is secured during
    transmission or storage, while internal model computations might occur
    in a less secure (plaintext) environment for performance reasons.
*   **Internal Plaintext Computation:** For simplicity and to avoid the complexities
    of HE-compatible non-linear activations, the `internal_model` (a standard
    PyTorch `nn.Sequential` model) operates on decrypted data.
*   **End-to-End Flow Demonstration:** Provides a clear example of the data flow
    from plaintext input, through encryption, model processing, watermarking,
    and encrypted output, with external verification.
*   **`__init__(input_dim: int, hidden_dim: int, output_dim: int, watermark_id: str, secret_key: str)`:**
    Initializes the `SecureModel` with network dimensions, a watermark identifier,
    and a secret key.
*   **`forward(x_plain_input: torch.Tensor) -> tenseal.CKKSVector`:**
    Takes a plaintext input, encrypts it, decrypts for internal processing,
    passes through the internal model, watermarks the output, and then encrypts
    the final watermarked output.

**Conceptual Significance:**
`SecureModel` illustrates a practical approach for securing neural networks
where full homomorphic encryption of all operations is not feasible or
performant. It highlights the trade-offs and design considerations for
implementing security at different levels of a deep learning pipeline.

dpnn_lib.security.watermark module
----------------------------------

.. automodule:: dpnn_lib.security.watermark
   :members:
   :show-inheritance:
   :undoc-members:

The `watermark` module provides the `WatermarkManager` class, responsible for
embedding and verifying digital watermarks within tensors. For this Proof-of-Concept,
a simplified watermarking technique is employed, focusing on demonstrating the
concept of traceability and tamper detection rather than cryptographic robustness.

**Key Features:**

*   **`WatermarkManager` Class:** Manages the generation, embedding, and verification
    of watermark patterns.
*   **`_generate_watermark_pattern(shape: tuple) -> torch.Tensor`:** Generates a
    pseudo-random watermark pattern based on a `watermark_id` and `secret_key`.
    This ensures a unique pattern for each model or user.
*   **`embed(plain_tensor: torch.Tensor) -> torch.Tensor`:** Embeds the watermark
    into a plaintext tensor by subtly modifying a fixed portion of its elements.
    The `strength` parameter controls the visibility of the watermark.
*   **`verify(marked_tensor: torch.Tensor, original_input_shape: tuple, expected_watermark_id: str, threshold: float)`:**
    Verifies the presence of the watermark by comparing an extracted pattern
    from the `marked_tensor` with the expected pattern. It also checks if the
    `WatermarkManager` instance's `watermark_id` matches the `expected_watermark_id`,
    providing a conceptual ID-based verification.
    A correlation-based threshold is used for detection.

**Limitations in PoC:**
The current watermarking method is a simplified noise-addition technique.
It is **not cryptographically robust** against sophisticated attacks (e.g.,
watermark removal, adversarial attacks, or model extraction). Real-world
applications would require significantly more advanced and robust watermarking
algorithms that can survive various neural network transformations and attacks.
This implementation serves to illustrate the *mechanism* of embedding and
verifying a watermark within the secure pipeline.

Module contents
---------------

.. automodule:: dpnn_lib.security
   :members:
   :show-inheritance:
   :undoc-members: