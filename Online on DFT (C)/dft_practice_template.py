import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# RULE: Do NOT use np.fft / scipy.fft / built-in conv/corr.
# ============================================================

EPS = 1e-12

def max_abs_error(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.max(np.abs(a - b))

def rel_l2_error(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.linalg.norm(a - b) / (np.linalg.norm(a) + EPS)

# -----------------------------
# TODO 1: DFT / IDFT
# -----------------------------
def dft(x):
    """
    X[k] = sum_{n=0}^{N-1} x[n] * exp(-j*2*pi*k*n/N)
    """
    # TODO
    x = np.array(x,dtype=complex)
    N = len(x)
    X=np.zeros_like(x)

    for k in range(N):
        for n in range(N):

            X[k] += x[n] * np.exp(-1j*2*np.pi*k*(n/N))

    return X
    
    # pass


def idft(X):
    """
    x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(+j*2*pi*k*n/N)
    """
    X = np.array(X,dtype=complex)
    N = len(X)
    x=np.zeros_like(X)

    for n in range(N):
        for k in range(N):

            x[n] += X[k] * np.exp(1j*2*np.pi*k*(n/N))

    return x/N
    # TODO
    # pass

# -----------------------------
# Signals (fixed)
# -----------------------------
def x_rect(N):
    x = np.zeros(N, dtype=float)
    x[: N // 8] = 1.0
    return x

def x_cos(N, m=5):
    n = np.arange(N)
    return np.cos(2 * np.pi * m * n / N)

def circular_shift(x, ns):
    # Allowed helper (not a built-in "DFT property", just indexing)
    return np.roll(np.asarray(x), ns)

# -----------------------------
# TODO 2: Circular convolution
# -----------------------------
def circular_convolution(x, h):
    """
    y[n] = sum_{m=0}^{N-1} x[m] * h[(n-m) mod N]
    x and h must have same length N
    """
    # TODO
    N= len(x)
    y = np.zeros(N)
    for n in range(N):
        for m in range(N):
            y[n] += x[m]*h[(n-m) % N]

    return y        
    # pass

# -----------------------------
# TODO 3: Cross-correlation via DFT
# -----------------------------
def cross_correlation_via_dft(x, y):
    """
    r_xy[n] = IDFT( X[k] * conj(Y[k]) )
    """
    # TODO
    return idft(dft(x)*np.conj(dft(y)))
    # pass

# -----------------------------
# Plot helpers
# -----------------------------
def stem_plot(y, title, xlabel="index", ylabel="value"):
    y = np.asarray(y)
    n = np.arange(len(y))
    plt.figure()
    plt.stem(n, y,)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

def plot_spectrum(X, prefix=""):
    X = np.asarray(X, dtype=complex)
    mag = np.abs(X)
    ph = np.angle(X)
    k = np.arange(len(X))

    plt.figure()
    plt.stem(k, mag,)
    plt.title(f"{prefix}Magnitude |X[k]|")
    plt.xlabel("k")
    plt.ylabel("|X[k]|")
    plt.grid(True)

    plt.figure()
    plt.stem(k, ph,)
    plt.title(f"{prefix}Phase ∠X[k]")
    plt.xlabel("k")
    plt.ylabel("angle (rad)")
    plt.grid(True)

def main():
    # -----------------------------
    # Task 1: DFT/IDFT + spectra
    # -----------------------------
    N = 64
    signals = {
        "rect": x_rect(N),
        "cos":  x_cos(N, m=5),
    }

    for name, x in signals.items():
        X = dft(x)
        x_hat = idft(X)

        print(f"\n[{name}] max_abs_error =", max_abs_error(x, x_hat))
        print(f"[{name}] rel_l2_error  =", rel_l2_error(x, x_hat))

        # x and reconstructed
        plt.figure()
        n = np.arange(N)
        plt.stem(n, x)
        plt.stem(n, np.real(x_hat), markerfmt="C1o", linefmt="C1-")
        plt.title(f"{name}: x[n] and Re{{x_hat[n]}}")
        plt.xlabel("n")
        plt.ylabel("value")
        plt.grid(True)
        plt.legend(["x[n]", "Re{x_hat[n]}"])

        # spectra
        plot_spectrum(X, prefix=f"{name}: ")

    # -----------------------------
    # Task 2: Circular convolution theorem (N=4)
    # -----------------------------
    x4 = np.array([1, 2, 3, 4], dtype=float)
    h4 = np.array([4, 3, 2, 1], dtype=float)

    y_time = circular_convolution(x4, h4)

    X4 = dft(x4)
    H4 = dft(h4)
    Y4 = X4 * H4
    y_freq = idft(Y4)

    print("\n[Conv N=4] y_time =", y_time)
    print("[Conv N=4] max_abs_error =", max_abs_error(y_time, np.real(y_freq)))

    plt.figure()
    n = np.arange(4)
    plt.stem(n, np.real(y_time))
    plt.stem(n, np.real(y_freq), markerfmt="C1o", linefmt="C1-")
    plt.title("Circular convolution (N=4): time-domain vs IDFT(DFT(x)*DFT(h))")
    plt.xlabel("n")
    plt.ylabel("value")
    plt.grid(True)
    plt.legend(["y_time", "Re{y_freq}"])

    # -----------------------------
    # Task 3: Cross-correlation via DFT
    # -----------------------------
    ns = 12
    x = signals["cos"]
    y = circular_shift(x, ns)
    # plt.stem
    stem_plot(x,title="Before shifted")
    stem_plot(y,title="AFter shifted")

    rxy = cross_correlation_via_dft(x, y)
    n_star = int(np.argmax(np.real(rxy)))

    print("\n[Corr] ns =", ns, " argmax n* =", n_star)

    stem_plot(np.real(rxy), "Cross-correlation r_xy[n] (real part)", xlabel="n", ylabel="r_xy[n]")

    plt.show()

if __name__ == "__main__":
    main()
