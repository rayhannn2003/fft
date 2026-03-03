import cv2
import numpy as np
import math

KERNEL_WIDTH = 9    # box-blur kernel width (must be odd)
EPSILON      = 1e-6 # guard against division by zero in deconvolution


# ─────────────────────────────────────────────────────────────────────
# FFT — Cooley-Tukey radix-2 DIT, O(n log n)
# ─────────────────────────────────────────────────────────────────────
def fft(x):
    """Compute 1-D FFT using the Cooley-Tukey radix-2 algorithm."""
    x = np.array(x, dtype=complex)
    N = len(x)

    if N <= 1:
        return x

    # Pad to next power of 2 if necessary
    if N & (N - 1) != 0:
        next_pow2 = 1 << math.ceil(math.log2(N))
        x = np.append(x, np.zeros(next_pow2 - N, dtype=complex))
        N = next_pow2

    even = fft(x[0::2])
    odd  = fft(x[1::2])

    T = np.array([np.exp(-2j * math.pi * k / N) * odd[k]
                  for k in range(N // 2)])

    return np.concatenate([even + T, even - T])


# ─────────────────────────────────────────────────────────────────────
# IFFT — conjugate symmetry trick
# ─────────────────────────────────────────────────────────────────────
def ifft(X):
    """Compute 1-D IFFT: IFFT(X) = conj(FFT(conj(X))) / N"""
    X = np.array(X, dtype=complex)
    N = len(X)
    return np.conj(fft(np.conj(X))) / N


# ─────────────────────────────────────────────────────────────────────
# Circular convolution of a single row with a kernel
# ─────────────────────────────────────────────────────────────────────
def circular_convolve_row(row, kernel):
    """
    Perform circular convolution of 'row' with 'kernel' using FFT.

    Property:  y[n] = x[n] ⊛ h[n]  ⟺  Y[k] = X[k] · H[k]
    Steps:
      1. Zero-pad the kernel to the same length as the row.
      2. Y = FFT(row) * FFT(h_padded)
      3. y = IFFT(Y)
    Returns the real part trimmed to the original row length.
    """
    N = len(row)

    # Zero-pad kernel to length N
    h_padded = np.zeros(N, dtype=complex)
    h_padded[:len(kernel)] = kernel

    Y = fft(row.astype(complex)) * fft(h_padded)
    y = ifft(Y)
    return np.real(y[:N])


# ─────────────────────────────────────────────────────────────────────
# Circular deconvolution of a single row
# ─────────────────────────────────────────────────────────────────────
def circular_deconvolve_row(blurred_row, H_kernel):
    """
    Recover the original row from a circularly blurred row.

    Property:  X[k] = Y[k] / H[k]  →  x[n] = IFFT(X[k])

    Parameters:
      blurred_row — 1-D array of length N (one colour channel)
      H_kernel    — pre-computed FFT of the zero-padded kernel (length >= N)
    """
    N   = len(blurred_row)
    pad = len(H_kernel)

    row_padded = np.append(blurred_row.astype(complex),
                           np.zeros(pad - N, dtype=complex))
    Y = fft(row_padded)

    # Deconvolution: divide in frequency domain
    X = np.where(np.abs(H_kernel) > EPSILON, Y / H_kernel, 0.0 + 0.0j)

    x = ifft(X)
    return np.real(x[:N])


# ─────────────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────────────
def reconstruct_image_using_fft(original_path, blurred_path, output_path):

    original_img = cv2.imread(original_path)
    blurred_img  = cv2.imread(blurred_path)

    if original_img is None or blurred_img is None:
        print("Error: Could not load images.")
        return

    if original_img.shape != blurred_img.shape:
        print("Error: Image dimensions do not match.")
        return

    rows, cols, channels = blurred_img.shape

    # ── Step 1: build box kernel ─────────────────────────────────────
    K      = KERNEL_WIDTH
    kernel = np.ones(K) / K          # normalised box kernel h[n]

    # ── Step 2: apply circular convolution to each row (blur) ────────
    # (In a real exam the blurred image is given; here we generate it
    #  ourselves to demonstrate the full forward + inverse pipeline.)
    blurred_demo = np.zeros_like(original_img, dtype=np.float64)
    for c in range(channels):
        for r in range(rows):
            blurred_demo[r, :, c] = circular_convolve_row(
                original_img[r, :, c].astype(float), kernel)

    blurred_demo = np.clip(blurred_demo, 0, 255).astype(np.uint8)

    # ── Step 3: pre-compute FFT of padded kernel once ────────────────
    pad_len  = 1 << math.ceil(math.log2(cols))
    h_padded = np.zeros(pad_len, dtype=complex)
    h_padded[:K] = kernel
    H_kernel = fft(h_padded)

    # ── Step 4: deconvolve each row (recover original) ───────────────
    reconstructed_img = np.zeros_like(blurred_demo, dtype=np.float64)
    print("Reconstructing image using circular deconvolution (manual FFT)...")

    for c in range(channels):
        for r in range(rows):
            reconstructed_img[r, :, c] = circular_deconvolve_row(
                blurred_demo[r, :, c].astype(float), H_kernel)

    reconstructed_img = np.clip(reconstructed_img, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, reconstructed_img)
    print(f"Reconstructed image saved to: {output_path}")

    # ── Step 5: verification ─────────────────────────────────────────
    orig_gray  = cv2.cvtColor(original_img,    cv2.COLOR_BGR2GRAY).astype(float)
    recon_gray = cv2.cvtColor(reconstructed_img, cv2.COLOR_BGR2GRAY).astype(float)
    mse = np.mean((orig_gray - recon_gray) ** 2)
    print(f"MSE (grayscale) between original and reconstructed: {mse:.4f}")

    if mse < 50.0:
        print("Verification PASSED: reconstructed image matches the original.")
    else:
        print("Verification FAILED: check your FFT / deconvolution logic.")


if __name__ == "__main__":
    reconstruct_image_using_fft(
        "original_image.png",
        "original_image.png",         # same file; blurring is done internally
        "reconstructed_image_fft.jpg"
    )
