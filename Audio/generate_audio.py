"""
Audio Generator for Circular Echo Effect Practice Problem
─────────────────────────────────────────────────────────
Run this ONCE to create:
  original_audio.wav     — clean sine-wave audio block
  impulse_response.wav   — short echo filter
  processed_audio.wav    — circularly convolved version
"""

import numpy as np
import scipy.io.wavfile as wav
import math

SAMPLE_RATE = 8000   # Hz
N           = 512    # block length (power of 2)
ECHO_DELAY  = 80     # samples — delay of the echo tap
ECHO_GAIN   = 0.5    # amplitude of the echo tap


def generate():
    n = np.arange(N)

    # ── Original audio: mix of two tones ─────────────────────────────
    original = (
        0.6 * np.sin(2 * math.pi * 440  * n / SAMPLE_RATE) +
        0.4 * np.sin(2 * math.pi * 1000 * n / SAMPLE_RATE)
    )

    # ── Impulse response: direct sound + one echo ─────────────────────
    # h[0] = 1  (direct),  h[ECHO_DELAY] = ECHO_GAIN  (echo)
    impulse = np.zeros(N)
    impulse[0]          = 1.0
    impulse[ECHO_DELAY] = ECHO_GAIN

    # ── Circular convolution via numpy FFT (ground-truth generator) ───
    Y = np.fft.fft(original) * np.fft.fft(impulse)
    processed = np.real(np.fft.ifft(Y))

    # ── Scale to int16 and save ───────────────────────────────────────
    def to_int16(sig):
        peak = np.max(np.abs(sig))
        if peak > 0:
            sig = sig / peak
        return (sig * 32767).astype(np.int16)

    wav.write("original_audio.wav",  SAMPLE_RATE, to_int16(original))
    wav.write("impulse_response.wav", SAMPLE_RATE, to_int16(impulse))
    wav.write("processed_audio.wav", SAMPLE_RATE, to_int16(processed))

    print(f"Generated signals (N={N}, sample_rate={SAMPLE_RATE} Hz)")
    print(f"Echo delay: {ECHO_DELAY} samples, gain: {ECHO_GAIN}")
    print("Files: original_audio.wav, impulse_response.wav, processed_audio.wav")


if __name__ == "__main__":
    generate()
