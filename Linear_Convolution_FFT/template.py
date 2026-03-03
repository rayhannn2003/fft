"""
CSE 220 — DFT Practice: Linear Convolution via FFT
────────────────────────────────────────────────────
Topic   : Image deblurring using FFT-based deconvolution
Property: y[n] = x[n] * h[n]  ⟺  Y[k] = X[k] · H[k]
          Therefore: X[k] = Y[k] / H[k]  →  x[n] = IFFT(X[k])

Fill in every section marked  ← YOUR CODE HERE
Do NOT use numpy.fft, scipy.fft, np.convolve, or any built-in FFT.
"""

import cv2
import numpy as np
import math

# ── Known kernel width (given in the problem) ───────────────────────
KERNEL_WIDTH = 9     # K — the box-filter half-used to blur the image
EPSILON      = 1e-6  # small value to avoid division by zero


# ─────────────────────────────────────────────────────────────────────
# TASK 1 — Implement FFT  (Cooley-Tukey radix-2, O(n log n))
# ─────────────────────────────────────────────────────────────────────
def fft(x):
    """
    Compute the 1-D DFT of x using the radix-2 Cooley-Tukey algorithm.

    Hints:
      - Base case: if len(x) == 1, return x.
      - Pad x to the next power of 2 if needed.
      - Split into even / odd index sub-arrays.
      - Recursively FFT each half.
      - Combine with twiddle factors:
            T[k]      = exp(-j2πk/N) * odd[k]
            X[k]      = even[k] + T[k]
            X[k+N/2]  = even[k] - T[k]
    """
    x = np.array(x, dtype=complex)
    N = len(x)

    # ← YOUR CODE HERE
    pass


# ─────────────────────────────────────────────────────────────────────
# TASK 2 — Implement IFFT  (reuse fft above)
# ─────────────────────────────────────────────────────────────────────
def ifft(X):
    """
    Compute the 1-D inverse DFT of X.

    Conjugate-symmetry trick:
        IFFT(X) = conj( FFT( conj(X) ) ) / N
    """
    X = np.array(X, dtype=complex)
    N = len(X)

    # ← YOUR CODE HERE
    pass


# ─────────────────────────────────────────────────────────────────────
# TASK 3 — Build the zero-padded box kernel
# ─────────────────────────────────────────────────────────────────────
def make_kernel(K, N):
    """
    Build a box (moving-average) kernel of width K, zero-padded to length N.

    A box kernel of width K:
        h[0..K-1] = 1/K,  h[K..N-1] = 0

    The kernel must be zero-padded to length N so that FFT(h) has the
    same length as FFT(row).

    Returns a 1-D numpy array of length N (dtype complex).
    """
    # ← YOUR CODE HERE
    pass


# ─────────────────────────────────────────────────────────────────────
# TASK 4 — Deconvolve one row
# ─────────────────────────────────────────────────────────────────────
def deconvolve_row(blurred_row, H_kernel):
    """
    Recover the original row from a blurred row using FFT deconvolution.

    Steps:
      1. Compute Y = FFT(blurred_row)
      2. Divide:  X[k] = Y[k] / H[k]
                  Use EPSILON to avoid division by near-zero values:
                  X[k] = Y[k] / H[k]  if |H[k]| > EPSILON, else 0
      3. Compute x = IFFT(X)
      4. Return the real part, trimmed to the original row length.

    Parameters:
      blurred_row  — 1-D array, length N (one colour channel)
      H_kernel     — FFT of the zero-padded box kernel, length >= N
    """
    N = len(blurred_row)

    # ← YOUR CODE HERE
    pass


# ─────────────────────────────────────────────────────────────────────
# Main reconstruction driver — do NOT modify
# ─────────────────────────────────────────────────────────────────────
def reconstruct_image(blurred_path, output_path, K):

    blurred_img = cv2.imread(blurred_path)
    if blurred_img is None:
        print(f"Error: Could not load '{blurred_path}'.")
        return

    rows, cols, channels = blurred_img.shape
    reconstructed = np.zeros_like(blurred_img, dtype=np.float64)

    # Pre-compute FFT of the kernel once (same for every row)
    h_padded = make_kernel(K, cols)
    H_kernel  = fft(h_padded)

    print("Deblurring image row by row using manual FFT...")
    for r in range(rows):
        for c in range(channels):
            reconstructed[r, :, c] = deconvolve_row(
                blurred_img[r, :, c].astype(float), H_kernel)

    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, reconstructed)
    print(f"Done. Reconstructed image saved to: {output_path}")

    # Verification
    original_img = cv2.imread("original_image.png")
    if original_img is not None and original_img.shape == reconstructed.shape:
        orig_gray  = cv2.cvtColor(original_img,   cv2.COLOR_BGR2GRAY).astype(float)
        recon_gray = cv2.cvtColor(reconstructed,  cv2.COLOR_BGR2GRAY).astype(float)
        mse = np.mean((orig_gray - recon_gray) ** 2)
        print(f"MSE (grayscale) between original and reconstructed: {mse:.4f}")
        if mse < 50.0:
            print("Verification PASSED: reconstructed image matches the original.")
        else:
            print("Verification NOTE: residual difference detected — check your implementation.")


if __name__ == "__main__":
    reconstruct_image("blurred_image.png", "reconstructed_image.png", KERNEL_WIDTH)
