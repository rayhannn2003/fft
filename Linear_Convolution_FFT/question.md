CSE 220 — Online on DFT
Section C | Time: 40 minutes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    1 of 2

Scenario
────────
Surveillance cameras mounted on city roads are often placed behind
protective glass panels. Over time, dust, moisture and vibration cause
the glass to act as a blurring medium. The optical effect of this glass
can be modelled as a **linear, shift-invariant (LSI) system**: every row
of the captured image is convolved with the same 1-D blur kernel before
it reaches the sensor.

A recently retrieved surveillance image shows this exact problem: every
horizontal row has been blurred by convolution with a known **box
(moving-average) kernel** of a fixed, known width.

Your task is to **undo this blur row-by-row using DFT-based
deconvolution**, thereby recovering the original, sharp image.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    2 of 2

Key Property
────────────
Linear Convolution Property of the DFT:

    If  y[n] = x[n] * h[n]   (linear convolution)

    Then  Y[k] = X[k] · H[k]   (point-wise multiplication in DFT domain)

    Therefore  X[k] = Y[k] / H[k]   (deconvolution = division in DFT domain)

    And  x[n] = IFFT( Y[k] / H[k] )

Your Mission
────────────
You are provided with two images:

  • original_image.png  — the original, sharp image.
  • blurred_image.png   — a distorted image with the following property:
        – Every row was convolved with the same 1-D box kernel.
        – The kernel width K is provided to you (see the driver code).
        – No vertical blurring was applied.

You must:

  1. Treat each row of the blurred image as a 1-D discrete signal.
  2. Build the 1-D box kernel h[n] of width K and zero-pad it to match
     the row length N.
  3. Compute FFT(y_row) and FFT(h_padded) using your own FFT function.
  4. Recover X[k] = Y[k] / H[k]  (use a small epsilon to avoid
     division by zero).
  5. Compute IFFT(X[k]) to recover the original row.
  6. Reconstruct the full image and verify it matches original_image.png.

Constraints
───────────
  • You MUST implement fft() and ifft() from scratch using the
    Cooley-Tukey radix-2 algorithm.  Complexity must be O(n log n).
  • You are NOT allowed to call numpy.fft, scipy.fft, or any
    built-in FFT/convolution function.
  • numpy is allowed only for array operations (indexing, math, etc.).

Expected Output
───────────────
    Deblurring image row by row using manual FFT...
    Done. Reconstructed image saved to: reconstructed_image.png
    MSE (grayscale) between original and reconstructed: 0.xxxx
    Verification PASSED: reconstructed image matches the original.

Submission
──────────
Rename template.py to your student ID and submit:
    2205XXX.py
