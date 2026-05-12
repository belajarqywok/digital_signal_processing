"""
=============================================================
DSP 01 — Dasar-Dasar Sinyal dan Sistem Digital
=============================================================
Implementasi Python lengkap untuk sinyal-sinyal dasar DSP,
analisis sifat sistem, dan visualisasi.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# 1. SINYAL-SINYAL DASAR
# ============================================================

def unit_impulse(n_range):
    """Impuls satuan delta[n]"""
    n = np.arange(n_range[0], n_range[1]+1)
    x = np.zeros(len(n))
    x[n == 0] = 1
    return n, x

def unit_step(n_range):
    """Step satuan u[n]"""
    n = np.arange(n_range[0], n_range[1]+1)
    x = np.where(n >= 0, 1.0, 0.0)
    return n, x

def unit_ramp(n_range):
    """Ramp satuan r[n] = n*u[n]"""
    n = np.arange(n_range[0], n_range[1]+1)
    x = np.where(n >= 0, n.astype(float), 0.0)
    return n, x

def discrete_sinusoid(n_range, A=1, omega0=np.pi/4, phi=0):
    """Sinusoid diskrit: x[n] = A*cos(omega0*n + phi)"""
    n = np.arange(n_range[0], n_range[1]+1)
    x = A * np.cos(omega0 * n + phi)
    return n, x

def geometric_exp(n_range, a=0.8):
    """Deret geometri: x[n] = a^n * u[n]"""
    n = np.arange(n_range[0], n_range[1]+1)
    x = np.where(n >= 0, a**n, 0.0)
    return n, x

def complex_exponential(n_range, omega0=np.pi/4):
    """Eksponensial kompleks: x[n] = e^(j*omega0*n)"""
    n = np.arange(n_range[0], n_range[1]+1)
    x = np.exp(1j * omega0 * n)
    return n, x

# ============================================================
# 2. SIFAT SINYAL
# ============================================================

def signal_energy(x):
    """Hitung energi total sinyal: E = sum(|x[n]|^2)"""
    return np.sum(np.abs(x)**2)

def signal_power(x):
    """Hitung daya rata-rata sinyal: P = mean(|x[n]|^2)"""
    return np.mean(np.abs(x)**2)

def is_energy_signal(x, tol=1e6):
    """Cek apakah sinyal adalah energy signal"""
    E = signal_energy(x)
    return E < tol, E

def even_part(n, x):
    """Bagian genap: x_e[n] = (x[n] + x[-n]) / 2"""
    # Flip sinyal
    x_flipped = np.flip(x)
    # Shift untuk menyelaraskan
    xe = (x + x_flipped) / 2
    return n, xe

def odd_part(n, x):
    """Bagian ganjil: x_o[n] = (x[n] - x[-n]) / 2"""
    x_flipped = np.flip(x)
    xo = (x - x_flipped) / 2
    return n, xo

def is_periodic(omega0, tol=1e-9):
    """
    Cek apakah sinusoid diskrit cos(omega0*n) periodik.
    Periodik jika omega0/(2*pi) adalah rasional.
    """
    ratio = omega0 / (2 * np.pi)
    # Cek apakah rasio mendekati bilangan rasional sederhana
    from fractions import Fraction
    frac = Fraction(ratio).limit_denominator(1000)
    is_per = abs(float(frac) - ratio) < tol
    period = frac.denominator if is_per else None
    return is_per, period

# ============================================================
# 3. OPERASI SINYAL
# ============================================================

def time_shift(n, x, k):
    """Geser sinyal: y[n] = x[n-k]"""
    return n + k, x  # Geser sumbu n

def time_reverse(n, x):
    """Balik sinyal: y[n] = x[-n]"""
    return -np.flip(n), np.flip(x)

def time_scale(n, x, M):
    """Downsampling: y[n] = x[Mn] — ambil setiap M sampel"""
    return n[::M], x[::M]

def signal_add(n1, x1, n2, x2):
    """Tambahkan dua sinyal dengan range n berbeda"""
    n_min = min(n1[0], n2[0])
    n_max = max(n1[-1], n2[-1])
    n = np.arange(n_min, n_max + 1)
    
    y1 = np.zeros(len(n))
    y2 = np.zeros(len(n))
    
    idx1 = n1 - n_min
    idx2 = n2 - n_min
    y1[idx1] = x1
    y2[idx2] = x2
    
    return n, y1 + y2

# ============================================================
# 4. SISTEM LTI — KONVOLUSI MANUAL
# ============================================================

def lti_convolve(x, h):
    """
    Konvolusi linear manual (tanpa numpy.convolve).
    y[n] = sum_k x[k] * h[n-k]
    """
    N = len(x)
    M = len(h)
    L = N + M - 1  # Panjang output
    y = np.zeros(L)
    
    for n in range(L):
        for k in range(N):
            if 0 <= n - k < M:
                y[n] += x[k] * h[n - k]
    return y

def lti_difference_equation(x, b, a):
    """
    Simulasi sistem LTI menggunakan persamaan differens.
    y[n] = sum(b[k]*x[n-k]) - sum(a[k]*y[n-k])
    
    Params:
        x: sinyal input
        b: koefisien feedforward (numerator)
        a: koefisien feedback (denominator), a[0] diasumsikan = 1
    """
    N = len(x)
    M_b = len(b)
    M_a = len(a)
    y = np.zeros(N)
    
    for n in range(N):
        # Feedforward (MA)
        for k in range(M_b):
            if n - k >= 0:
                y[n] += b[k] * x[n - k]
        # Feedback (AR)
        for k in range(1, M_a):
            if n - k >= 0:
                y[n] -= a[k] * y[n - k]
    
    return y

# ============================================================
# 5. ANALISIS STABILITAS
# ============================================================

def check_stability_from_h(h, tol=1e4):
    """Cek stabilitas BIBO: sum|h[n]| < inf"""
    abssum = np.sum(np.abs(h))
    return abssum < tol, abssum

def check_stability_from_poles(a):
    """
    Cek stabilitas dari koefisien AR.
    Poles: akar dari polinomial A(z).
    Stabil jika semua |poles| < 1.
    """
    poles = np.roots(a)
    stable = np.all(np.abs(poles) < 1)
    return stable, poles

# ============================================================
# 6. VISUALISASI
# ============================================================

def plot_stem(n, x, title="Sinyal Diskrit", xlabel="n", ylabel="x[n]", color='steelblue'):
    """Plot stem untuk sinyal diskrit"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stem(n, x, linefmt=f'{color}-', markerfmt=f'{color}o', basefmt='k-')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.8)
    plt.tight_layout()
    return fig

def plot_basic_signals():
    """Plot semua sinyal dasar dalam satu figure"""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)
    
    n_range = (-5, 20)
    signals = [
        ("Impuls δ[n]",          *unit_impulse(n_range),      'royalblue'),
        ("Step u[n]",             *unit_step(n_range),         'forestgreen'),
        ("Ramp r[n]",             *unit_ramp(n_range),         'darkorange'),
        ("Sinusoid cos(π/4·n)",   *discrete_sinusoid(n_range, omega0=np.pi/4), 'crimson'),
        ("Geo exp a=0.8",         *geometric_exp(n_range, 0.8), 'purple'),
        ("Geo exp a=1.2 (unstbl)", *geometric_exp(n_range, 1.2), 'brown'),
        ("Geo exp a=-0.8",        *geometric_exp(n_range, -0.8), 'teal'),
        ("Re{e^(jπ/4·n)}",        *( lambda: (complex_exponential(n_range, np.pi/4)[0],
                                                np.real(complex_exponential(n_range, np.pi/4)[1])))(), 'navy'),
        ("Im{e^(jπ/4·n)}",        *( lambda: (complex_exponential(n_range, np.pi/4)[0],
                                                np.imag(complex_exponential(n_range, np.pi/4)[1])))(), 'darkred'),
    ]
    
    for i, (title, n, x, color) in enumerate(signals):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.stem(n, x, linefmt=f'{color}-', markerfmt=f'{color}o', basefmt='k-')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('n', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.8)
    
    fig.suptitle('Sinyal-Sinyal Dasar DSP', fontsize=16, fontweight='bold', y=1.01)
    plt.savefig('01_basic_signals.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Gambar disimpan: 01_basic_signals.png")

def plot_even_odd_decomposition():
    """Dekomposisi genap-ganjil sebuah sinyal"""
    n = np.arange(-10, 11)
    x = np.where(n >= 0, (0.8)**n, 0)  # Exponensial kausal
    
    _, xe = even_part(n, x)
    _, xo = odd_part(n, x)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].stem(n, x, linefmt='b-', markerfmt='bo', basefmt='k-')
    axes[0].set_title('Sinyal Asli x[n]', fontweight='bold')
    
    axes[1].stem(n, xe, linefmt='g-', markerfmt='go', basefmt='k-')
    axes[1].set_title('Bagian Genap x_e[n]', fontweight='bold')
    
    axes[2].stem(n, xo, linefmt='r-', markerfmt='ro', basefmt='k-')
    axes[2].set_title('Bagian Ganjil x_o[n]', fontweight='bold')
    
    for ax in axes:
        ax.set_xlabel('n')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.8)
    
    plt.suptitle('Dekomposisi Even-Odd', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('01_even_odd.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================
# 7. DEMO UTAMA
# ============================================================

def main():
    print("=" * 60)
    print("   DSP 01 — Dasar-Dasar Sinyal dan Sistem Digital")
    print("=" * 60)
    
    # --- Sinyal dasar ---
    print("\n[1] Plot sinyal-sinyal dasar...")
    plot_basic_signals()
    
    # --- Analisis energi & daya ---
    print("\n[2] Analisis Energi dan Daya")
    n_range = (0, 100)
    
    signals_to_check = [
        ("Impuls δ[n]",          unit_impulse(n_range)[1]),
        ("Step u[n]",             unit_step(n_range)[1]),
        ("Sinusoid cos(π/4·n)",   discrete_sinusoid(n_range, omega0=np.pi/4)[1]),
        ("Geo exp a=0.8",         geometric_exp(n_range, 0.8)[1]),
    ]
    
    print(f"{'Sinyal':<25} {'Energi':>12} {'Daya':>12} {'Jenis':>15}")
    print("-" * 65)
    for name, x in signals_to_check:
        E = signal_energy(x)
        P = signal_power(x)
        if E < 1e8:
            kind = "Energy Signal"
        elif P > 0 and P < 1e8:
            kind = "Power Signal"
        else:
            kind = "Neither"
        print(f"{name:<25} {E:>12.4f} {P:>12.6f} {kind:>15}")
    
    # --- Periodisitas ---
    print("\n[3] Cek Periodisitas Sinusoid Diskrit")
    freqs = [np.pi/4, np.pi/3, 1.0, 0.1]
    for w in freqs:
        per, N = is_periodic(w)
        print(f"  ω₀ = {w:.4f} rad/sample → Periodik: {per}, Periode N = {N}")
    
    # --- Dekomposisi genap-ganjil ---
    print("\n[4] Dekomposisi Genap-Ganjil...")
    plot_even_odd_decomposition()
    
    # --- Sistem LTI sederhana ---
    print("\n[5] Sistem LTI — Moving Average 3 Tap")
    x_input = np.array([1, 2, 3, 4, 3, 2, 1, 0, 0, 0], dtype=float)
    h_ma3   = np.array([1/3, 1/3, 1/3])  # Filter MA orde-3
    
    y_conv   = np.convolve(x_input, h_ma3)
    y_manual = lti_convolve(x_input, h_ma3)
    
    print(f"  Input:          {x_input}")
    print(f"  Output (numpy): {np.round(y_conv[:len(x_input)], 4)}")
    print(f"  Output (manual):{np.round(y_manual[:len(x_input)], 4)}")
    print(f"  Selisih max:    {np.max(np.abs(y_conv - y_manual)):.2e}")
    
    # --- Cek stabilitas ---
    print("\n[6] Analisis Stabilitas Sistem")
    systems = [
        ("Moving Average",   [1/3, 1/3, 1/3], [1]),
        ("Filter Stabil",    [1, 0], [1, -0.5]),
        ("Filter Unstable",  [1, 0], [1, -1.5]),
    ]
    
    for name, b, a in systems:
        stable, poles = check_stability_from_poles(np.array(a))
        print(f"  {name:<20} | Poles: {np.round(poles, 3)} | Stabil: {stable}")
    
    # --- Persamaan differens ---
    print("\n[7] Simulasi Persamaan Differens: y[n] = x[n] + 0.5*x[n-1] - 0.3*y[n-1]")
    x_test = np.zeros(20)
    x_test[0] = 1  # Impuls input
    b_coef = [1, 0.5]
    a_coef = [1, 0.3]
    y_out = lti_difference_equation(x_test, b_coef, a_coef)
    print(f"  Respons impuls (20 sampel pertama):")
    print(f"  {np.round(y_out, 4)}")
    
    print("\n✅ Demo selesai!")

if __name__ == "__main__":
    main()
