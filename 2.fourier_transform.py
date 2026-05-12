"""
=============================================================
DSP 02 — Transformasi Fourier (DFT, FFT, STFT)
=============================================================
Implementasi DFT manual, perbandingan dengan FFT,
analisis frekuensi, dan STFT / spectrogram.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as sci_signal
import time

# ============================================================
# 1. DFT MANUAL (tanpa numpy.fft)
# ============================================================

def dft_naive(x):
    """
    DFT naif O(N^2).
    X[k] = sum_{n=0}^{N-1} x[n] * exp(-j*2*pi*k*n/N)
    """
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-1j * 2 * np.pi * k * n / N)
    return X

def idft(X):
    """
    Inverse DFT.
    x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(j*2*pi*k*n/N)
    """
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(1j * 2 * np.pi * k * n / N)
    return x / N

def fft_cooley_tukey(x):
    """
    FFT Cooley-Tukey Radix-2 DIT (Decimation-In-Time).
    N harus pangkat 2.
    """
    N = len(x)
    if N == 1:
        return x
    
    # Periksa N adalah power of 2
    if N & (N - 1) != 0:
        raise ValueError(f"N harus power of 2, dapat {N}")
    
    # Pisahkan genap dan ganjil
    X_even = fft_cooley_tukey(x[0::2])  # Indeks genap
    X_odd  = fft_cooley_tukey(x[1::2])  # Indeks ganjil
    
    # Twiddle factors
    T = np.array([np.exp(-2j * np.pi * k / N) * X_odd[k] for k in range(N // 2)])
    
    return np.concatenate([X_even + T, X_even - T])

# ============================================================
# 2. ANALISIS SPEKTRUM
# ============================================================

def compute_spectrum(x, fs=1.0, N=None, window='hann'):
    """
    Hitung magnitude spectrum dengan windowing.
    
    Returns:
        freqs: array frekuensi (Hz)
        magnitude: magnitude spectrum (linear)
        magnitude_db: magnitude spectrum (dB)
        phase: fase spectrum (radian)
    """
    if N is None:
        N = len(x)
    
    # Terapkan window
    if window == 'hann':
        win = np.hanning(len(x))
    elif window == 'hamming':
        win = np.hamming(len(x))
    elif window == 'blackman':
        win = np.blackman(len(x))
    elif window == 'rectangular' or window is None:
        win = np.ones(len(x))
    else:
        win = np.hanning(len(x))
    
    x_windowed = x * win
    
    # Zero-pad jika N > len(x)
    if N > len(x):
        x_padded = np.zeros(N)
        x_padded[:len(x)] = x_windowed
    else:
        x_padded = x_windowed[:N]
    
    # FFT
    X = np.fft.fft(x_padded)
    
    # Frekuensi
    freqs = np.fft.fftfreq(N, d=1/fs)
    
    # Magnitude dan fase
    magnitude = np.abs(X) / (N * np.mean(win))  # Normalisasi
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    phase = np.angle(X)
    
    # Kembalikan hanya sisi positif (single-sided)
    half = N // 2
    return freqs[:half], magnitude[:half], magnitude_db[:half], phase[:half]

def frequency_resolution(fs, N):
    """Hitung resolusi frekuensi dan info dasar"""
    df = fs / N
    T = N / fs
    f_nyquist = fs / 2
    return {
        'resolusi_frekuensi_Hz': df,
        'durasi_sinyal_s': T,
        'frekuensi_nyquist_Hz': f_nyquist,
        'jumlah_bin': N // 2 + 1,
    }

# ============================================================
# 3. ZERO-PADDING DEMO
# ============================================================

def demo_zero_padding(x, fs=1.0, pad_factors=[1, 2, 4, 8]):
    """Demonstrasi efek zero-padding pada resolusi spektrum"""
    N_orig = len(x)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    
    for i, factor in enumerate(pad_factors):
        N_padded = N_orig * factor
        freqs, mag, _, _ = compute_spectrum(x, fs, N=N_padded, window='hann')
        
        axes[i].plot(freqs, mag, linewidth=1.2)
        axes[i].set_title(f'Zero-Padding ×{factor} (N={N_padded})', fontweight='bold')
        axes[i].set_xlabel('Frekuensi (Hz)')
        axes[i].set_ylabel('Magnitude')
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Efek Zero-Padding pada Spektrum', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('02_zero_padding.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================
# 4. WINDOW FUNCTIONS
# ============================================================

def compare_windows(N=64):
    """Bandingkan berbagai window functions dalam domain waktu dan frekuensi"""
    windows = {
        'Rectangular': np.ones(N),
        'Hanning':     np.hanning(N),
        'Hamming':     np.hamming(N),
        'Blackman':    np.blackman(N),
        'Kaiser β=8':  np.kaiser(N, 8),
    }
    
    colors = ['royalblue', 'darkorange', 'forestgreen', 'crimson', 'purple']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Domain waktu
    for (name, w), color in zip(windows.items(), colors):
        ax1.plot(w, label=name, color=color, linewidth=2)
    ax1.set_title('Window Functions — Domain Waktu', fontweight='bold')
    ax1.set_xlabel('Sampel n')
    ax1.set_ylabel('Amplitudo')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Domain frekuensi (response)
    N_fft = 512
    for (name, w), color in zip(windows.items(), colors):
        W = np.fft.fft(w, n=N_fft)
        W_db = 20 * np.log10(np.abs(W[:N_fft//2]) + 1e-10)
        W_db -= np.max(W_db)  # Normalisasi ke 0 dB
        freqs = np.linspace(0, 0.5, N_fft//2)
        ax2.plot(freqs, W_db, label=name, color=color, linewidth=1.5)
    
    ax2.set_title('Window Functions — Respons Frekuensi', fontweight='bold')
    ax2.set_xlabel('Frekuensi Ternormalisasi (× fs)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_ylim(-120, 5)
    ax2.axhline(-3, color='gray', linestyle='--', alpha=0.5, label='-3 dB')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('02_windows.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Statistik window
    print("\n📊 Statistik Window Functions:")
    print(f"{'Window':<20} {'Coherent Gain':>15} {'3dB Bandwidth':>15}")
    print("-" * 52)
    for name, w in windows.items():
        cg = np.mean(w)
        # Cari -3dB bandwidth
        W = np.fft.fft(w, n=N_fft)
        W_db = 20*np.log10(np.abs(W[:N_fft//2]) + 1e-10)
        W_db -= np.max(W_db)
        bw_3db = np.sum(W_db > -3) / (N_fft//2)
        print(f"  {name:<18} {cg:>15.4f} {bw_3db*100:>14.2f}%")

# ============================================================
# 5. STFT DAN SPECTROGRAM
# ============================================================

def compute_stft(x, fs, window_size=256, hop_size=128, window='hann'):
    """
    Hitung STFT manual.
    
    Returns:
        t: array waktu
        f: array frekuensi
        Zxx: matriks STFT kompleks
    """
    N = window_size
    
    if window == 'hann':
        win = np.hanning(N)
    elif window == 'hamming':
        win = np.hamming(N)
    else:
        win = np.ones(N)
    
    # Jumlah frame
    n_frames = (len(x) - N) // hop_size + 1
    
    # Inisialisasi
    Zxx = np.zeros((N // 2 + 1, n_frames), dtype=complex)
    
    for i in range(n_frames):
        start = i * hop_size
        end = start + N
        if end > len(x):
            break
        
        frame = x[start:end] * win
        spectrum = np.fft.rfft(frame)
        Zxx[:, i] = spectrum
    
    # Waktu dan frekuensi
    t = np.arange(n_frames) * hop_size / fs
    f = np.fft.rfftfreq(N, 1/fs)
    
    return t, f, Zxx

def plot_spectrogram(x, fs, title="Spectrogram", window_size=256, hop_size=64):
    """Plot spectrogram dari sinyal"""
    t, f, Zxx = compute_stft(x, fs, window_size, hop_size)
    
    magnitude_db = 20 * np.log10(np.abs(Zxx) + 1e-10)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Waveform
    time_axis = np.arange(len(x)) / fs
    axes[0].plot(time_axis, x, color='royalblue', linewidth=0.8)
    axes[0].set_title('Waveform', fontweight='bold')
    axes[0].set_xlabel('Waktu (detik)')
    axes[0].set_ylabel('Amplitudo')
    axes[0].grid(True, alpha=0.3)
    
    # Spectrogram
    im = axes[1].pcolormesh(t, f, magnitude_db,
                             shading='gouraud', cmap='magma',
                             vmin=np.max(magnitude_db)-80,
                             vmax=np.max(magnitude_db))
    plt.colorbar(im, ax=axes[1], label='Magnitude (dB)')
    axes[1].set_title(f'Spectrogram — {title}', fontweight='bold')
    axes[1].set_xlabel('Waktu (detik)')
    axes[1].set_ylabel('Frekuensi (Hz)')
    
    plt.tight_layout()
    plt.savefig(f'02_spectrogram.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================
# 6. BENCHMARKING DFT vs FFT
# ============================================================

def benchmark_fft_vs_dft():
    """Bandingkan kecepatan DFT naif vs FFT"""
    sizes = [16, 32, 64, 128, 256, 512]
    times_dft = []
    times_fft = []
    
    print("\n⏱️  Benchmark DFT Naif vs FFT:")
    print(f"{'N':>6} | {'DFT (ms)':>10} | {'FFT (ms)':>10} | {'Speedup':>10}")
    print("-" * 44)
    
    for N in sizes:
        x = np.random.randn(N)
        
        # DFT naif
        t0 = time.perf_counter()
        for _ in range(5):
            dft_naive(x)
        t_dft = (time.perf_counter() - t0) / 5 * 1000
        
        # FFT
        t0 = time.perf_counter()
        for _ in range(5):
            np.fft.fft(x)
        t_fft = (time.perf_counter() - t0) / 5 * 1000
        
        speedup = t_dft / t_fft if t_fft > 0 else float('inf')
        times_dft.append(t_dft)
        times_fft.append(t_fft)
        
        print(f"{N:>6} | {t_dft:>10.3f} | {t_fft:>10.3f} | {speedup:>10.1f}×")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.semilogy(sizes, times_dft, 'ro-', label='DFT Naif O(N²)', linewidth=2)
    ax1.semilogy(sizes, times_fft, 'bs-', label='FFT O(N log N)', linewidth=2)
    ax1.set_xlabel('Ukuran N')
    ax1.set_ylabel('Waktu (ms) — Log Scale')
    ax1.set_title('DFT Naif vs FFT', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    speedups = [d/f for d, f in zip(times_dft, times_fft)]
    ax2.plot(sizes, speedups, 'g^-', linewidth=2)
    ax2.set_xlabel('Ukuran N')
    ax2.set_ylabel('Speedup (×)')
    ax2.set_title('Speedup FFT vs DFT Naif', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('02_benchmark.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================
# 7. CIRCULAR CONVOLUTION
# ============================================================

def circular_convolution(x1, x2):
    """Konvolusi sirkular menggunakan DFT"""
    N = max(len(x1), len(x2))
    X1 = np.fft.fft(x1, n=N)
    X2 = np.fft.fft(x2, n=N)
    return np.real(np.fft.ifft(X1 * X2))

def linear_via_fft(x1, x2):
    """Konvolusi linear menggunakan FFT (dengan zero-padding)"""
    N = len(x1) + len(x2) - 1
    X1 = np.fft.fft(x1, n=N)
    X2 = np.fft.fft(x2, n=N)
    return np.real(np.fft.ifft(X1 * X2))

# ============================================================
# 8. DEMO UTAMA
# ============================================================

def main():
    print("=" * 60)
    print("   DSP 02 — Transformasi Fourier (DFT, FFT, STFT)")
    print("=" * 60)
    
    fs = 1000  # Hz
    T = 1.0    # detik
    N = 1000
    t = np.linspace(0, T, N, endpoint=False)
    
    # Buat sinyal test: 3 komponen frekuensi + noise
    f1, f2, f3 = 50, 120, 300  # Hz
    x = (1.0 * np.cos(2*np.pi*f1*t) +
         0.5 * np.cos(2*np.pi*f2*t) +
         0.25 * np.cos(2*np.pi*f3*t) +
         0.1 * np.random.randn(N))
    
    print(f"\n[1] Sinyal: 3 komponen ({f1}, {f2}, {f3} Hz) + noise")
    print(f"    fs={fs}Hz, N={N}, Resolusi frekuensi={fs/N:.1f}Hz")
    
    # --- Verifikasi DFT manual vs NumPy FFT ---
    print("\n[2] Verifikasi DFT manual vs NumPy FFT...")
    x_short = x[:64]  # Gunakan 64 sampel saja untuk DFT naif
    
    X_manual = dft_naive(x_short)
    X_numpy  = np.fft.fft(x_short)
    X_custom = fft_cooley_tukey(x_short)
    
    err_naive = np.max(np.abs(X_manual - X_numpy))
    err_custom = np.max(np.abs(X_custom - X_numpy))
    print(f"    Error DFT naif vs NumPy:   {err_naive:.2e}")
    print(f"    Error FFT custom vs NumPy: {err_custom:.2e}")
    
    # --- Analisis spektrum ---
    print("\n[3] Analisis spektrum (plot)...")
    freqs, mag, mag_db, phase = compute_spectrum(x, fs, N, 'hann')
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(freqs, mag, 'royalblue', linewidth=1.5)
    axes[0].set_title('Magnitude Spectrum (Linear)', fontweight='bold')
    axes[0].set_xlabel('Frekuensi (Hz)'); axes[0].set_ylabel('Magnitude')
    for f in [f1, f2, f3]:
        axes[0].axvline(f, color='r', linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(freqs, mag_db, 'darkorange', linewidth=1.5)
    axes[1].set_title('Magnitude Spectrum (dB)', fontweight='bold')
    axes[1].set_xlabel('Frekuensi (Hz)'); axes[1].set_ylabel('Magnitude (dB)')
    for f in [f1, f2, f3]:
        axes[1].axvline(f, color='r', linestyle='--', alpha=0.5, label=f'{f}Hz')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('02_spectrum.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # --- Benchmark ---
    print("\n[4] Benchmark DFT vs FFT...")
    benchmark_fft_vs_dft()
    
    # --- Window functions ---
    print("\n[5] Perbandingan window functions...")
    compare_windows(N=64)
    
    # --- Zero-padding demo ---
    print("\n[6] Demo zero-padding...")
    x_short_demo = x[:100]
    demo_zero_padding(x_short_demo, fs, pad_factors=[1, 2, 4, 8])
    
    # --- STFT & Spectrogram ---
    print("\n[7] STFT dan Spectrogram...")
    # Sinyal chirp (frekuensi berubah terhadap waktu)
    t_chirp = np.linspace(0, 2, 4000)
    x_chirp = sci_signal.chirp(t_chirp, f0=10, f1=400, t1=2, method='linear')
    x_chirp += 0.1 * np.random.randn(len(x_chirp))
    plot_spectrogram(x_chirp, fs=2000, title="Linear Chirp 10-400 Hz")
    
    # --- Circular vs Linear Convolution ---
    print("\n[8] Circular vs Linear Convolution...")
    a = np.array([1, 2, 3, 4], dtype=float)
    b = np.array([1, 1, 1], dtype=float)
    
    y_linear  = np.convolve(a, b)
    y_circ    = circular_convolution(a, b)
    y_lin_fft = linear_via_fft(a, b)
    
    print(f"  a = {a}")
    print(f"  b = {b}")
    print(f"  Konvolusi linear:        {y_linear}")
    print(f"  Konvolusi sirkular (N=4):{y_circ}")
    print(f"  Linear via FFT:          {np.round(y_lin_fft, 6)}")
    
    # Info frekuensi
    print("\n[9] Info Frekuensi untuk sinyal:")
    info = frequency_resolution(fs, N)
    for k, v in info.items():
        print(f"    {k}: {v}")
    
    print("\n✅ Demo FFT selesai!")

if __name__ == "__main__":
    main()
