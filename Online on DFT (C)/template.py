import cv2
import numpy as np
import math


def fft(x):
    """
    Compute 1D FFT using the Cooley-Tukey radix-2 DIT algorithm (O(n log n)).
    Input length must be a power of 2.
    """
    x = np.array(x, dtype=complex)
    N = len(x)

    # Base case
    if N <= 1:
        return x

    # Pad to next power of 2 if necessary
    if N & (N - 1) != 0:
        next_pow2 = 1 << math.ceil(math.log2(N))
        x = np.append(x, np.zeros(next_pow2 - N, dtype=complex))
        N = next_pow2

    # Divide
    even = fft(x[0::2])
    odd  = fft(x[1::2])

    # Combine
    T = np.array([np.exp(-2j * math.pi * k / N) * odd[k] for k in range(N // 2)])
    return np.concatenate([even + T, even - T])


def ifft(X):
    """
    Compute 1D inverse FFT using the FFT function.
    IFFT(X) = conj(FFT(conj(X))) / N
    """
    X = np.array(X, dtype=complex)
    N = len(X)
    # Conjugate input, apply FFT, conjugate output, divide by N
    return np.conj(fft(np.conj(X))) / N


def cross_correlate_fft(a, b):
    """
    Compute circular cross-correlation of 1D signals a and b using FFT.
    Returns the correlation array. The peak index gives the shift of b
    relative to a (i.e., how much b is shifted to the right compared to a).
    """
    N = len(a)
    # Pad both to the same power-of-2 length
    pad_len = 1 << math.ceil(math.log2(N))

    A = fft(np.append(a, np.zeros(pad_len - N, dtype=complex)))
    B = fft(np.append(b, np.zeros(pad_len - N, dtype=complex)))

    # Cross-correlation in frequency domain: conj(A) * B
    corr = ifft(np.conj(A) * B)
    return np.real(corr)


def detect_shift(orig_row, shifted_row):
    """
    Detect horizontal circular shift of shifted_row relative to orig_row.
    Returns the shift amount (positive = shifted right).
    """
    corr = cross_correlate_fft(orig_row, shifted_row)
    shift = int(np.argmax(corr))
    return shift


def reconstruct_image_using_fft(original_path, shifted_path, output_path):

    original_img = cv2.imread(original_path)
    shifted_img  = cv2.imread(shifted_path)

    if original_img is None or shifted_img is None:
        print("Error: Could not load images.")
        return

    if original_img.shape != shifted_img.shape:
        print("Error: Image dimensions do not match.")
        return

    # Convert the original and shifted color images to grayscale.
    orig_gray  = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    shift_gray = cv2.cvtColor(shifted_img,  cv2.COLOR_BGR2GRAY)

    print("Reconstructing image using manual FFT...")

    rows, cols = shift_gray.shape
    reconstructed_img = np.zeros_like(shifted_img)

    for r in range(rows):
        # Detect the horizontal shift for this row using grayscale
        shift = detect_shift(orig_gray[r].astype(float),
                             shift_gray[r].astype(float))

        # Reverse the shift on every colour channel
        for c in range(shifted_img.shape[2]):
            reconstructed_img[r, :, c] = np.roll(shifted_img[r, :, c], -shift)

    cv2.imwrite(output_path, reconstructed_img)
    print(f"Reconstructed image saved to: {output_path}")

    # Verification
    orig_gray_check  = cv2.cvtColor(original_img,      cv2.COLOR_BGR2GRAY).astype(float)
    recon_gray_check = cv2.cvtColor(reconstructed_img, cv2.COLOR_BGR2GRAY).astype(float)
    mse = np.mean((orig_gray_check - recon_gray_check) ** 2)
    print(f"MSE between original and reconstructed (grayscale): {mse:.4f}")
    if mse < 1.0:
        print("Verification PASSED: reconstructed image matches the original.")
    else:
        print("Verification NOTE: small residual difference (may be due to JPEG compression).")


if __name__ == "__main__":
    reconstruct_image_using_fft("original_image.png", "shifted_image.jpg", "reconstructed_image_fft.jpg")