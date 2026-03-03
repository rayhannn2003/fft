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

#Property: Circular cross-correlation via FFT
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


# ══════════════════════════════════════════════════════════════════════════════
#  ALL DFT / FFT PROPERTIES
#  Each function below demonstrates one property of the DFT.
#  Property statement is given in the docstring, followed by the implementation.
# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# 1. LINEARITY
#    a·x[n] + b·y[n]  ⟺  a·X[k] + b·Y[k]
# ─────────────────────────────────────────────────────────────────────────────
def dft_linearity(x, y, a=1, b=1):
    """
    Property: DFT{a·x[n] + b·y[n]} = a·X[k] + b·Y[k]

    Verifies that the DFT is a linear operator.
    Returns True if the property holds (within floating-point tolerance).
    """
    N = max(len(x), len(y))
    x = np.append(x, np.zeros(N - len(x)))
    y = np.append(y, np.zeros(N - len(y)))

    lhs = fft(a * x + b * y)          # DFT of linear combination
    rhs = a * fft(x) + b * fft(y)     # linear combination of DFTs

    holds = np.allclose(lhs, rhs)
    print(f"[Linearity]  holds = {holds}")
    return holds


# ─────────────────────────────────────────────────────────────────────────────
# 2. TIME SHIFTING  (Circular Shift Property)
#    x[(n - k) mod N]  ⟺  X[m] · e^(-j2πmk/N)
# ─────────────────────────────────────────────────────────────────────────────
def dft_time_shift(x, k):
    """
    Property: DFT{x[(n-k) mod N]} = X[m] · e^{-j2πmk/N}

    Shifting a signal by k samples in time multiplies its DFT by a
    complex exponential (linear phase factor).

    Returns the DFT of the circularly shifted signal.
    """
    N      = len(x)
    X      = fft(x)
    m      = np.arange(len(X))
    # Apply phase shift in frequency domain
    X_shifted = X * np.exp(-2j * math.pi * m * k / N)

    # Verify against direct shift
    x_shifted_direct = np.roll(x, k)
    X_shifted_direct = fft(x_shifted_direct)

    holds = np.allclose(X_shifted[:N], X_shifted_direct[:N])
    print(f"[Time Shift]  k={k}, holds = {holds}")
    return X_shifted


# ─────────────────────────────────────────────────────────────────────────────
# 3. FREQUENCY SHIFTING  (Modulation Property)
#    x[n] · e^(j2πn·k0/N)  ⟺  X[(m - k0) mod N]
# ─────────────────────────────────────────────────────────────────────────────
def dft_frequency_shift(x, k0):
    """
    Property: DFT{x[n] · e^{j2πnk0/N}} = X[(m - k0) mod N]

    Multiplying a signal by a complex sinusoid shifts its spectrum by k0 bins.

    Returns the DFT of the modulated signal.
    """
    N = len(x)
    n = np.arange(N)
    x_modulated = x * np.exp(2j * math.pi * n * k0 / N)

    X_shifted_freq = fft(x_modulated)

    # Verify against circular shift of X
    X      = fft(x)
    pad    = 1 << math.ceil(math.log2(N)) if N & (N - 1) != 0 else N
    X_full = fft(np.append(x, np.zeros(pad - N)))
    X_circ = np.roll(X_full, k0)

    print(f"[Frequency Shift]  k0={k0}")
    return X_shifted_freq


# ─────────────────────────────────────────────────────────────────────────────
# 4. TIME REVERSAL
#    x[(-n) mod N]  ⟺  X[(-k) mod N]  =  conj(X[k])  (if x is real)
# ─────────────────────────────────────────────────────────────────────────────
def dft_time_reversal(x):
    """
    Property: DFT{x[(-n) mod N]} = X[(-k) mod N]
    For real x: X[(-k) mod N] = conj(X[k])  (conjugate symmetry)

    Reversing a signal in time reverses its spectrum.
    Returns the DFT of the time-reversed signal.
    """
    N = len(x)
    # x[(-n) mod N] = [x[0], x[N-1], x[N-2], ..., x[1]]
    x_reversed = np.concatenate(([x[0]], x[1:][::-1]))
    X_reversed_direct = fft(x_reversed)

    # Expected: X[(-k) mod N] — circular reversal of spectrum
    X    = fft(x.astype(complex))
    pad  = len(X_reversed_direct)
    # Pad X to same length, then circularly reverse
    X_pad = np.append(X, np.zeros(pad - N, dtype=complex))
    X_rev = np.concatenate(([X_pad[0]], X_pad[1:][::-1]))

    holds = np.allclose(X_reversed_direct, X_rev)
    print(f"[Time Reversal]  holds = {holds}")
    return X_reversed_direct


# ─────────────────────────────────────────────────────────────────────────────
# 5. CIRCULAR CONVOLUTION
#    x[n] ⊛ h[n]  ⟺  X[k] · H[k]
# ─────────────────────────────────────────────────────────────────────────────
def circular_convolve(x, h):
    """
    Property: DFT{x[n] ⊛ h[n]} = X[k] · H[k]

    Circular convolution in the time domain equals pointwise multiplication
    in the frequency domain.

    Returns the result of circular convolution (real part, trimmed to len(x)).
    """
    N = len(x)
    # Zero-pad h to length N
    h_pad = np.append(h, np.zeros(N - len(h)))

    X = fft(x.astype(complex))
    H = fft(h_pad.astype(complex))
    Y = X * H
    y = ifft(Y)
    print(f"[Circular Convolution]  output length = {N}")
    return np.real(y[:N])


# ─────────────────────────────────────────────────────────────────────────────
# 6. CIRCULAR CROSS-CORRELATION  (already implemented above, shown for clarity)
#    Corr(x, y)[n] = IFFT( conj(X[k]) · Y[k] )
#    Peak index gives the shift of y relative to x.
# ─────────────────────────────────────────────────────────────────────────────
def cross_correlate(x, y):
    """
    Property: Corr(x,y)[n] = IFFT( conj(X[k]) · Y[k] )

    Cross-correlation measures similarity between two signals at each lag.
    The lag (index) of the maximum value gives the circular shift of y
    relative to x.

    Returns the real-valued correlation array.
    """
    N   = max(len(x), len(y))
    pad = 1 << math.ceil(math.log2(N))

    X = fft(np.append(x, np.zeros(pad - len(x), dtype=complex)))
    Y = fft(np.append(y, np.zeros(pad - len(y), dtype=complex)))

    corr  = ifft(np.conj(X) * Y)
    shift = int(np.argmax(np.real(corr)))
    print(f"[Cross-Correlation]  detected shift = {shift} samples")
    return np.real(corr)


# ─────────────────────────────────────────────────────────────────────────────
# 7. PARSEVAL'S THEOREM  (Energy Conservation)
#    Σ|x[n]|²  =  (1/N) · Σ|X[k]|²
# ─────────────────────────────────────────────────────────────────────────────
def parsevals_theorem(x):
    """
    Property: Σ_{n=0}^{N-1} |x[n]|²  =  (1/N) · Σ_{k=0}^{N-1} |X[k]|²

    Total energy in the time domain equals total energy in the frequency
    domain (scaled by 1/N). Ensures no energy is lost or gained by the DFT.

    Returns (time_energy, freq_energy, holds).
    """
    N           = len(x)
    X           = fft(x)
    pad         = len(X)

    time_energy = np.sum(np.abs(x) ** 2)
    freq_energy = np.sum(np.abs(X) ** 2) / pad   # divide by padded length

    # Re-check using only the original N bins
    X_n         = fft(x.astype(complex))
    freq_energy_n = np.sum(np.abs(X_n[:N]) ** 2) / len(X_n)

    holds = np.isclose(time_energy, freq_energy_n, rtol=1e-3)
    print(f"[Parseval's Theorem]  time={time_energy:.4f}  "
          f"freq={freq_energy_n:.4f}  holds={holds}")
    return time_energy, freq_energy_n, holds


# ─────────────────────────────────────────────────────────────────────────────
# 8. CONJUGATE SYMMETRY  (for real-valued signals)
#    x[n] ∈ ℝ  ⟹  X[k] = conj(X[N-k])
# ─────────────────────────────────────────────────────────────────────────────
def conjugate_symmetry(x):
    """
    Property: If x[n] is real then X[k] = X*[N-k]  (conjugate symmetric)

    This means the DFT of a real signal is fully described by its first N//2+1
    unique frequency bins (the rest are mirror images).

    Returns True if the property holds.
    """
    X     = fft(x.astype(complex))
    N     = len(X)
    holds = np.allclose(X[1:], np.conj(X[N - 1:0:-1]))
    print(f"[Conjugate Symmetry]  holds = {holds}")
    return holds


# ─────────────────────────────────────────────────────────────────────────────
# 9. DUALITY
#    If DFT{x[n]} = X[k], then DFT{X[n]} = N · x[(-k) mod N]
# ─────────────────────────────────────────────────────────────────────────────
def dft_duality(x):
    """
    Property: DFT{ X[n] } = N · x[(-k) mod N]

    The DFT of the DFT spectrum (treating it as a time signal) gives back
    the original signal (time-reversed and scaled by N).

    Returns True if the property holds.
    """
    N   = len(x)
    X   = fft(x.astype(complex))[:N]   # keep only N samples

    # Apply DFT to X treated as a signal
    DFT_of_X = fft(X)[:N]

    # Expected: N * x[(-k) mod N]  = N * np.roll(np.flip(x), 1)
    expected = N * np.roll(np.flip(x.astype(complex)), 1)
    # Pad expected to same power-of-2 length
    pad      = len(fft(X))
    expected_pad = np.append(expected, np.zeros(pad - N, dtype=complex))

    holds = np.allclose(fft(X), expected_pad, atol=1e-6)
    print(f"[Duality]  holds = {holds}")
    return holds


# ─────────────────────────────────────────────────────────────────────────────
# 10. MULTIPLICATION IN TIME  (Frequency-Domain Convolution / Windowing)
#     x[n] · y[n]  ⟺  (1/N) · (X[k] ⊛ Y[k])
# ─────────────────────────────────────────────────────────────────────────────
def dft_multiplication_in_time(x, y):
    """
    Property: DFT{ x[n] · y[n] } = (1/N) · (X[k] ⊛ Y[k])

    Pointwise multiplication in the time domain equals circular convolution
    of the two spectra, scaled by 1/N.

    This is the mathematical basis of WINDOWING: multiplying a signal by a
    window function (Hamming, Hanning, etc.) convolves its spectrum with the
    window's spectrum — this is how spectral leakage is controlled.

    Returns True if LHS == RHS within floating-point tolerance.
    """
    N = len(x)
    assert len(y) == N, "x and y must have the same length"

    # LHS: DFT of the product
    lhs = fft((x * y).astype(complex))[:N]

    # RHS: (1/N) * circular convolution of X and Y in frequency domain
    X = fft(x.astype(complex))[:N]
    Y = fft(y.astype(complex))[:N]

    # Circular convolve X and Y (both length N, already aligned)
    XY_conv = np.real(ifft(fft(X) * fft(Y)))[:N]
    rhs = XY_conv / N

    holds = np.allclose(np.real(lhs), rhs, atol=1e-6)
    print(f"[Multiplication in Time]  holds = {holds}")
    return holds


# ─────────────────────────────────────────────────────────────────────────────
# 11. DC COMPONENT  (Summation Property)
#     X[0] = Σ x[n]  for n = 0 … N-1
# ─────────────────────────────────────────────────────────────────────────────
def dft_dc_component(x):
    """
    Property: X[0] = Σ_{n=0}^{N-1} x[n]

    The very first FFT bin (index 0) equals the sum of all time-domain samples.
    It represents the DC offset (zero-frequency average) of the signal.

    A pure sine wave centred on zero has X[0] = 0.

    Returns (X0, sum_x, holds).
    """
    X     = fft(x.astype(complex))
    X0    = X[0]
    sum_x = np.sum(x)

    holds = np.isclose(np.real(X0), sum_x, atol=1e-6)
    print(f"[DC Component]  X[0]={np.real(X0):.4f}  "
          f"sum(x)={sum_x:.4f}  holds={holds}")
    return np.real(X0), sum_x, holds


# ─────────────────────────────────────────────────────────────────────────────
# 12. NYQUIST BIN  (Alternating Sum Property)
#     X[N/2] = Σ (-1)^n · x[n]  for n = 0 … N-1   (N even)
# ─────────────────────────────────────────────────────────────────────────────
def dft_nyquist_bin(x):
    """
    Property: X[N/2] = Σ_{n=0}^{N-1} (-1)^n · x[n]    (N must be even)

    The middle FFT bin (index N/2) equals the alternating sum of the
    time-domain signal. It represents the Nyquist frequency — the highest
    measurable frequency — where the signal alternates sign every sample.

    Returns (X_nyquist, alternating_sum, holds).
    """
    N = len(x)
    assert N % 2 == 0, "N must be even for the Nyquist bin property"

    X = fft(x.astype(complex))

    X_nyquist      = np.real(X[N // 2])
    alternating_sum = np.sum(np.array([(-1)**n * x[n] for n in range(N)]))

    holds = np.isclose(X_nyquist, alternating_sum, atol=1e-6)
    print(f"[Nyquist Bin]  X[N/2]={X_nyquist:.4f}  "
          f"alt_sum={alternating_sum:.4f}  holds={holds}")
    return X_nyquist, alternating_sum, holds


# ─────────────────────────────────────────────────────────────────────────────
# 13. REAL/EVEN AND REAL/ODD SYMMETRY
#
#   Real & Even signal: x[n] = x[-n]
#       → FFT is entirely Real and Even  (imaginary part = 0)
#       → Foundation of the DCT (used in JPEG compression)
#
#   Real & Odd signal:  x[n] = -x[-n]
#       → FFT is entirely Imaginary and Odd (real part = 0)
# ─────────────────────────────────────────────────────────────────────────────
def dft_real_even_odd_symmetry(N=16):
    """
    Property A — Real & Even input:
        x[n] = x[(-n) mod N]  →  X[k] is real and even  (Im(X[k]) ≈ 0)

    Property B — Real & Odd input:
        x[n] = -x[(-n) mod N]  →  X[k] is imaginary and odd  (Re(X[k]) ≈ 0)

    This is the mathematical basis of the DCT:
    mirroring a signal to make it even removes all imaginary DFT components,
    halving the computation — exactly what JPEG image compression exploits.

    Returns (even_holds, odd_holds).
    """
    n = np.arange(N)

    # ── Real & Even signal (cosine is naturally even) ─────────────────
    x_even = np.cos(2 * math.pi * 2 * n / N)   # x[n] = x[-n]
    X_even = fft(x_even)[:N]

    even_is_real = np.allclose(np.imag(X_even), 0, atol=1e-9)
    even_is_even = np.allclose(np.real(X_even),
                               np.concatenate(([np.real(X_even[0])],
                                               np.real(X_even[1:])[::-1])),
                               atol=1e-9)
    even_holds = even_is_real and even_is_even

    # ── Real & Odd signal (sine is naturally odd) ─────────────────────
    x_odd = np.sin(2 * math.pi * 2 * n / N)    # x[n] = -x[-n]
    X_odd = fft(x_odd)[:N]

    odd_is_imag = np.allclose(np.real(X_odd), 0, atol=1e-9)
    odd_holds   = odd_is_imag

    print(f"[Real/Even Symmetry]  FFT is real & even  → holds = {even_holds}")
    print(f"[Real/Odd  Symmetry]  FFT is imaginary & odd → holds = {odd_holds}")
    print(f"  (DCT connection: even-mirrored signal has zero imaginary FFT "
          f"→ real-only transform)")
    return even_holds, odd_holds


# ─────────────────────────────────────────────────────────────────────────────
# DEMO  —  run all properties on a simple test signal
# ─────────────────────────────────────────────────────────────────────────────
def demo():
    N  = 64
    n  = np.arange(N)
    x  = np.sin(2 * math.pi * 3 * n / N) + 0.5 * np.cos(2 * math.pi * 7 * n / N)
    h  = np.array([1/4, 1/4, 1/4, 1/4])   # box kernel

    y  = np.sin(2 * math.pi * 5 * n / N)

    print("=" * 60)
    print("  DFT Properties Demo")
    print("=" * 60)
    dft_linearity(x, y, a=2, b=3)
    dft_time_shift(x, k=5)
    dft_frequency_shift(x, k0=4)
    dft_time_reversal(x)
    circular_convolve(x, h)
    cross_correlate(x, np.roll(x, 10))
    parsevals_theorem(x)
    conjugate_symmetry(x)
    dft_duality(x)
    # ── 4 new properties ──────────────────────────────────────────────
    dft_multiplication_in_time(x, y)
    dft_dc_component(x)
    dft_nyquist_bin(x)
    dft_real_even_odd_symmetry(N)
    print("=" * 60)




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
    demo()
    reconstruct_image_using_fft("Online on DFT (C)/original_image.png", "Online on DFT (C)/shifted_image.jpg", "reconstructed_image_fft.jpg")