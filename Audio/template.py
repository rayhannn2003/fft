import numpy as np
import scipy.io.wavfile as wav
import math

EPSILON = 1e-6   # guard against division by zero


# ─────────────────────────────────────────────────────────────────────
# FFT — Cooley-Tukey radix-2 DIT, O(n log n)
# ─────────────────────────────────────────────────────────────────────
def fft(x):
    """
    Compute 1-D FFT using the Cooley-Tukey radix-2 algorithm.
    Pads input to the next power of 2 automatically.

    Steps:
      1. Base case: N == 1  →  return x
      2. Pad to next power of 2 if N is not already a power of 2
      3. Split into even / odd sub-arrays
      4. Recursively FFT each half
      5. Combine with twiddle factors:
            T[k]      = exp(-j2πk/N) · odd[k]
            X[k]      = even[k] + T[k]
            X[k+N/2]  = even[k] - T[k]
    """
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

    T = np.array([np.exp(-2j * math.pi * k / N) * odd[k]
                  for k in range(N // 2)])

    return np.concatenate([even + T, even - T])


# ─────────────────────────────────────────────────────────────────────
# IFFT — conjugate symmetry trick
# ─────────────────────────────────────────────────────────────────────
def ifft(X):
    """
    Compute 1-D inverse FFT by reusing fft().

    Identity used:
        IFFT(X) = conj( FFT( conj(X) ) ) / N
    """
    X = np.array(X, dtype=complex)
    N = len(X)
    return np.conj(fft(np.conj(X))) / N


# ─────────────────────────────────────────────────────────────────────
# Circular deconvolution of a 1-D audio block
# ─────────────────────────────────────────────────────────────────────
def deconvolve(processed_block, H):
    """
    Recover the original block from a circularly convolved block.

    Property:  Y[k] = X[k] · H[k]
    Therefore: X[k] = Y[k] / H[k]   →   x[n] = IFFT(X[k])

    Parameters
    ----------
    processed_block : 1-D float array of length N
    H               : pre-computed FFT of the zero-padded impulse (length >= N)

    Returns the real-valued recovered signal trimmed to length N.
    """
    N   = len(processed_block)
    pad = len(H)

    # Zero-pad block to match H's FFT length
    block_padded = np.append(processed_block.astype(complex),
                             np.zeros(pad - N, dtype=complex))
    Y = fft(block_padded)

    # Frequency-domain deconvolution with epsilon guard
    X = np.where(np.abs(H) > EPSILON, Y / H, 0.0 + 0.0j)

    x = ifft(X)
    return np.real(x[:N])


# ─────────────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────────────
def reconstruct_audio_using_fft(original_path, processed_path,
                                 impulse_path, output_path):

    sr1, original  = wav.read(original_path)
    sr2, processed = wav.read(processed_path)
    sr3, impulse   = wav.read(impulse_path)

    if sr1 != sr2 or sr1 != sr3:
        print("Sampling rates do not match.")
        return

    original  = original.astype(float)
    processed = processed.astype(float)
    impulse   = impulse.astype(float)

    N = len(processed)

    print("Reconstructing original audio using circular convolution property...")

    # ── Step 1: zero-pad impulse to length N ─────────────────────────
    # Impulse may be shorter than processed; pad to same block length
    h_padded = np.zeros(N)
    h_padded[:len(impulse)] = impulse[:N]   # truncate if longer than N

    # ── Step 2: compute FFT of impulse once ──────────────────────────
    pad_len  = 1 << math.ceil(math.log2(N))
    h_full   = np.append(h_padded, np.zeros(pad_len - N, dtype=complex))
    H        = fft(h_full)

    # ── Step 3 & 4: X[k] = Y[k] / H[k]  →  x[n] = IFFT(X[k]) ──────
    reconstructed = deconvolve(processed, H)

    # ── Step 5: clip, scale to int16 and save ────────────────────────
    peak = np.max(np.abs(reconstructed)) if np.max(np.abs(reconstructed)) > 0 else 1
    reconstructed_norm = np.clip(reconstructed / peak * 32767, -32768, 32767)

    wav.write(output_path, sr1, reconstructed_norm.astype(np.int16))
    print(f"Reconstructed audio saved to: {output_path}")

    # ── Step 6: verification (MSE against original) ───────────────────
    # Normalise both signals to [-1, 1] for a fair comparison
    orig_norm = original / (np.max(np.abs(original)) + 1e-12)
    rec_norm  = reconstructed / (np.max(np.abs(reconstructed)) + 1e-12)

    # Trim to same length
    L   = min(len(orig_norm), len(rec_norm))
    mse = np.mean((orig_norm[:L] - rec_norm[:L]) ** 2)
    print(f"MSE between original and reconstructed (normalised): {mse:.8f}")

    if mse < 1e-4:
        print("Verification PASSED: reconstructed audio matches the original.")
    else:
        print("Verification NOTE: residual MSE detected — check deconvolution logic.")


if __name__ == "__main__":
    reconstruct_audio_using_fft(
        "original_audio.wav",
        "processed_audio.wav",
        "impulse_response.wav",
        "reconstructed_audio.wav"
    )
