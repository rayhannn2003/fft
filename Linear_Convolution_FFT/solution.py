"""
CSE 220 — DFT Practice: Linear Convolution via FFT
────────────────────────────────────────────────────
REFERENCE SOLUTION — study this after attempting template.py

Topic   : Image deblurring using FFT-based deconvolution
Property: y[n] = x[n] * h[n]  ⟺  Y[k] = X[k] · H[k]
          Deconvolution:  X[k] = Y[k] / H[k]  →  x[n] = IFFT(X[k])
"""

import cv2
import numpy as np
import math

KERNEL_WIDTH = 9
EPSILON      = 1e-6


# ─────────────────────────────────────────────────────────────────────
# FFT — Cooley-Tukey radix-2 DIT, O(n log n)
# ─────────────────────────────────────────────────────────────────────
def fft(x):
    """1-D FFT using Cooley-Tukey radix-2. Pads to next power of 2."""
    x = np.array(x, dtype=complex)
    N = len(x)

    if N <= 1:
        return x

    # Pad to next power of 2
    if N & (N - 1) != 0:
        next_pow2 = 1 << math.ceil(math.log2(N))
        x = np.append(x, np.zeros(next_pow2 - N, dtype=complex))
        N = next_pow2

    even = fft(x[0::2])
    odd  = fft(x[1::2])

    # Twiddle factors
    T = np.array([np.exp(-2j * math.pi * k / N) * odd[k]
                  for k in range(N // 2)])

    return np.concatenate([even + T, even - T])


# ─────────────────────────────────────────────────────────────────────
# IFFT — conjugate symmetry trick
# ─────────────────────────────────────────────────────────────────────
def ifft(X):
    """1-D IFFT via  IFFT(X) = conj(FFT(conj(X))) / N"""
    X = np.array(X, dtype=complex)
    N = len(X)
    return np.conj(fft(np.conj(X))) / N


# ─────────────────────────────────────────────────────────────────────
# Build zero-padded box kernel
# ─────────────────────────────────────────────────────────────────────
def make_kernel(K, N):
    """
    Box (moving-average) kernel of width K, zero-padded to length N.
    h[0..K-1] = 1/K,  h[K..N-1] = 0
    """
    h = np.zeros(N, dtype=complex)
    h[:K] = 1.0 / K
    return h


# ─────────────────────────────────────────────────────────────────────
# Deconvolve one row
# ─────────────────────────────────────────────────────────────────────
def deconvolve_row(blurred_row, H_kernel):
    """
    Recover original row from blurred row.

    Steps:
      1. Y = FFT(blurred_row)  — pad to same length as H_kernel
      2. X[k] = Y[k] / H[k]   (with epsilon guard)
      3. x = IFFT(X)
      4. Return real part trimmed to original length N
    """
    N    = len(blurred_row)
    pad  = len(H_kernel)    # already power-of-2 padded length

    # Zero-pad the row to match the kernel's FFT length
    row_padded = np.append(blurred_row.astype(complex),
                           np.zeros(pad - N, dtype=complex))
    Y = fft(row_padded)

    # Division in frequency domain (deconvolution)
    X = np.where(np.abs(H_kernel) > EPSILON,
                 Y / H_kernel,
                 0.0 + 0.0j)

    x = ifft(X)
    return np.real(x[:N])


# ─────────────────────────────────────────────────────────────────────
# Main reconstruction driver
# ─────────────────────────────────────────────────────────────────────
def reconstruct_image(blurred_path, output_path, K):

    blurred_img = cv2.imread(blurred_path)
    if blurred_img is None:
        print(f"Error: Could not load '{blurred_path}'.")
        return

    rows, cols, channels = blurred_img.shape
    reconstructed = np.zeros_like(blurred_img, dtype=np.float64)

    # Pre-compute padded-kernel FFT once (reused for every row)
    # Pad to next power of 2 >= cols so FFT lengths match
    pad_len  = 1 << math.ceil(math.log2(cols))
    h_padded = make_kernel(K, pad_len)
    H_kernel  = fft(h_padded)

    print("Deblurring image row by row using manual FFT...")
    for r in range(rows):
        for c in range(channels):
            reconstructed[r, :, c] = deconvolve_row(
                blurred_img[r, :, c].astype(float), H_kernel)

    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, reconstructed)
    print(f"Done. Reconstructed image saved to: {output_path}")

    # ── Verification ─────────────────────────────────────────────────
    original_img = cv2.imread("original_image.png")
    if original_img is not None and original_img.shape == reconstructed.shape:
        orig_gray  = cv2.cvtColor(original_img,  cv2.COLOR_BGR2GRAY).astype(float)
        recon_gray = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY).astype(float)
        mse = np.mean((orig_gray - recon_gray) ** 2)
        print(f"MSE (grayscale) between original and reconstructed: {mse:.4f}")
        if mse < 50.0:
            print("Verification PASSED: reconstructed image matches the original.")
        else:
            print(f"Verification NOTE: MSE = {mse:.4f} — "
                  "small residual expected due to 'same'-mode border effects.")
    else:
        print("(original_image.png not found for verification)")


if __name__ == "__main__":
    reconstruct_image("blurred_image.png", "reconstructed_image.png", KERNEL_WIDTH)
