"""
Blur Generator for Linear Convolution FFT Practice Problem
──────────────────────────────────────────────────────────
Run this script ONCE to produce blurred_image.png from original_image.png.

It applies a 1-D box (moving-average) kernel of width K to every row of the
image using numpy's convolve (for ground-truth generation only).
The kernel width K is printed so you can hard-code it into template.py.
"""

import cv2
import numpy as np

KERNEL_WIDTH = 9          # must be odd; change freely for harder problems
OUTPUT_BLURRED = "blurred_image.png"
OUTPUT_ORIGINAL = "original_image.png"


def box_kernel(K):
    """Return a normalised box (moving-average) kernel of width K."""
    return np.ones(K) / K


def blur_image(input_path, output_path, K):
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not load '{input_path}'.")
        print("Place an image named 'original_image.png' in this folder and re-run.")
        return

    kernel = box_kernel(K)
    blurred = np.zeros_like(img, dtype=np.float64)

    rows, cols, channels = img.shape
    for c in range(channels):
        for r in range(rows):
            # Circular convolution via FFT — consistent with FFT deconvolution
            row   = img[r, :, c].astype(float)
            h_pad = np.zeros(cols)
            h_pad[:K] = kernel          # zero-pad kernel to row length
            # Multiply spectra and take real IFFT
            blurred[r, :, c] = np.real(
                np.fft.ifft(np.fft.fft(row) * np.fft.fft(h_pad))
            )

    blurred = np.clip(blurred, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, blurred)
    print(f"Blur applied with box kernel of width K = {K}")
    print(f"Original image : {input_path}")
    print(f"Blurred image  : {output_path}")
    print(f"\nUse KERNEL_WIDTH = {K} in your template / solution.")


if __name__ == "__main__":
    blur_image(OUTPUT_ORIGINAL, OUTPUT_BLURRED, KERNEL_WIDTH)
