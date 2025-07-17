import torch
import tenseal as ts

class HEManager:
    """
    Manages Homomorphic Encryption (HE) operations using TenSEAL.
    Uses the CKKS scheme for approximate arithmetic on real numbers.
    """
    def __init__(self):
        # CKKS context setup
        # - poly_modulus_degree: Controls the number of coefficients in the plaintext polynomials.
        #   Larger values allow for more complex computations but increase ciphertext size.
        # - coeff_mod_bit_sizes: Defines the bit lengths of the coefficient moduli.
        #   This impacts the precision and noise budget. The last value is for the scale.
        # - global_scale: Determines the initial scaling factor for plaintext values.
        #   Crucial for maintaining precision during computations.
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,  # A common choice for reasonable security and performance
            coeff_mod_bit_sizes=[60, 40, 40, 60] # Example bit sizes, adjust as needed
        )
        self.context.generate_galois_keys() # Required for rotations
        self.context.global_scale = 2**40 # A common scale, adjust based on data range and operations

    def encrypt(self, plain: torch.Tensor) -> ts.CKKSVector:
        """
        Encrypts a plain PyTorch tensor into a TenSEAL CKKSVector.
        The tensor is flattened before encryption.
        """
        # TenSEAL CKKSVector expects a list or numpy array of floats.
        # Flatten the tensor and convert to a list of floats.
        plain_list = plain.flatten().tolist()
        
        # Encrypt the plaintext vector
        enc_vec = ts.ckks_vector(self.context, plain_list)
        return enc_vec

    def decrypt(self, cipher: ts.CKKSVector) -> torch.Tensor:
        """
        Decrypts a TenSEAL CKKSVector back into a PyTorch tensor.
        The decrypted vector is reshaped to the original tensor's shape.
        Note: The original shape is lost during encryption, so this PoC assumes
        the user knows the original shape or it's fixed. In a real system,
        shape information would need to be managed.
        """
        # Decrypt the ciphertext vector
        decrypted_list = cipher.decrypt()
        
        # Convert the decrypted list back to a PyTorch tensor.
        # For this PoC, we'll assume a fixed output shape or that the caller handles reshaping.
        # For now, we'll return a 1D tensor, and the SecurityCell will need to handle reshaping.
        return torch.tensor(decrypted_list, dtype=torch.float32)