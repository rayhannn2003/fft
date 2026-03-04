import cv2
import numpy as np
import math

# NOTE:
# - This script solves the variant: each COLUMN is circularly shifted (up/down)
#   by a different amount.
# - Even though people often say "shifted to the right" for images, a *column*
#   shift is naturally a *vertical* shift (up/down). The math is identical.


def fft(x):
    """1-D FFT using Cooley-Tukey radix-2. Pads input to next power of 2."""
    x = np.array(x, dtype=complex)
    N = len(x)

    if N <= 1:
        return x

    if N & (N - 1) != 0:
        next_pow2 = 1 << math.ceil(math.log2(N))
        x = np.append(x, np.zeros(next_pow2 - N, dtype=complex))
        N = next_pow2

    even = fft(x[0::2])
    odd = fft(x[1::2])

    T = np.array([np.exp(-2j * math.pi * k / N) * odd[k] for k in range(N // 2)])
    return np.concatenate([even + T, even - T])


def ifft(X):
    """1-D IFFT via IFFT(X) = conj(FFT(conj(X))) / N"""
    X = np.array(X, dtype=complex)
    N = len(X)
    return np.conj(fft(np.conj(X))) / N


def cross_correlate_fft(a, b):
    """Circular cross-correlation Corr = IFFT(conj(FFT(a)) * FFT(b))."""
    N = len(a)
    pad_len = 1 << math.ceil(math.log2(N))

    A = fft(np.append(a, np.zeros(pad_len - N, dtype=complex)))
    B = fft(np.append(b, np.zeros(pad_len - N, dtype=complex)))

    corr = ifft(np.conj(A) * B)
    return np.real(corr)


def detect_shift(ref_1d, shifted_1d):
    """Return shift of shifted_1d relative to ref_1d (positive = shifted down)."""
    corr = cross_correlate_fft(ref_1d, shifted_1d)
    return int(np.argmax(corr))


def reconstruct_image_by_column_shifts(original_path, shifted_path, output_path):
    original_img = cv2.imread(original_path)
    shifted_img = cv2.imread(shifted_path)

    if original_img is None or shifted_img is None:
        print("Error: Could not load images.")
        return

    if original_img.shape != shifted_img.shape:
        print("Error: Image dimensions do not match.")
        return

    orig_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    shift_gray = cv2.cvtColor(shifted_img, cv2.COLOR_BGR2GRAY)

    rows, cols = orig_gray.shape
    reconstructed = np.zeros_like(shifted_img)

    print("Reconstructing image (per-COLUMN circular shifts) using FFT cross-correlation...")

    for c in range(cols):
        shift = detect_shift(orig_gray[:, c].astype(float), shift_gray[:, c].astype(float))

        # Reverse that vertical shift on each color channel
        for ch in range(shifted_img.shape[2]):
            reconstructed[:, c, ch] = np.roll(shifted_img[:, c, ch], -shift)

    cv2.imwrite(output_path, reconstructed)
    print(f"Saved: {output_path}")

    # Verification (grayscale MSE)
    recon_gray = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY).astype(float)
    mse = np.mean((orig_gray.astype(float) - recon_gray) ** 2)
    print(f"MSE (grayscale): {mse:.4f}")


if __name__ == "__main__":
    reconstruct_image_by_column_shifts(
        "original_image.png",
        "shifted_image.jpg",
        "reconstructed_image_column_fix.jpg",
    )
