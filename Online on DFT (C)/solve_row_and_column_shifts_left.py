import cv2
import numpy as np
import math

# Variant solved by this script:
# - Each ROW is circularly shifted to the LEFT by a different amount.
# - Each COLUMN is ALSO circularly shifted UP (equivalently: left in the column index)
#   by a different amount.
#
# Important note:
# If both row-shifts and column-shifts are unknown, there isn't a single
# one-shot formula using only the final image and the original reference.
# But because we *do* have the true original (reference) image, we can solve it
# reliably in two stages:
#   1) Estimate + undo row shifts (using cross-correlation on each row)
#   2) Estimate + undo column shifts (using cross-correlation on each column)
# After stage (1), columns become aligned enough to accurately detect their shifts.


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
    """Return shift of shifted_1d relative to ref_1d (positive = shift forward)."""
    corr = cross_correlate_fft(ref_1d, shifted_1d)
    return int(np.argmax(corr))


def solve_row_and_column_shifts_left(original_path, shifted_path, output_path):
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

    print("Stage 1: detecting + undoing per-row LEFT shifts...")

    # Stage 1: undo row shifts
    stage1 = np.zeros_like(shifted_img)
    row_shifts = np.zeros(rows, dtype=int)

    for r in range(rows):
        # detect shift of shifted row relative to reference row
        s = detect_shift(orig_gray[r].astype(float), shift_gray[r].astype(float))
        row_shifts[r] = s

        # If the row was shifted LEFT by L, then relative to original it appears as
        # shift = (N - L) in argmax convention. Undo by rolling -shift.
        for ch in range(shifted_img.shape[2]):
            stage1[r, :, ch] = np.roll(shifted_img[r, :, ch], -s)

    # Recompute grayscale after stage 1
    stage1_gray = cv2.cvtColor(stage1, cv2.COLOR_BGR2GRAY)

    print("Stage 2: detecting + undoing per-column UP shifts...")

    # Stage 2: undo column shifts
    reconstructed = np.zeros_like(stage1)
    col_shifts = np.zeros(cols, dtype=int)

    for c in range(cols):
        s = detect_shift(orig_gray[:, c].astype(float), stage1_gray[:, c].astype(float))
        col_shifts[c] = s

        for ch in range(stage1.shape[2]):
            reconstructed[:, c, ch] = np.roll(stage1[:, c, ch], -s)

    cv2.imwrite(output_path, reconstructed)
    print(f"Saved: {output_path}")

    recon_gray = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY).astype(float)
    mse = np.mean((orig_gray.astype(float) - recon_gray) ** 2)
    print(f"MSE (grayscale): {mse:.4f}")


if __name__ == "__main__":
    solve_row_and_column_shifts_left(
        "original_image.png",
        "shifted_image.jpg",
        "reconstructed_image_row_col_fix.jpg",
    )
