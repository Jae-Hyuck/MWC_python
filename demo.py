import numpy as np
import math


def mix_signal(sig, t_axis, pattern, Tp):

    _, M = pattern.shape
    ind = np.floor(t_axis/(Tp/M)) % M
    ind = ind.astype(int)
    return sig * pattern[:, ind]


def interpft(x, ny):
    siz = x.size
    a = np.fft.fft(x)
    nyqst = int(np.ceil((siz+1)/2))
    b = np.concatenate([a[:nyqst], np.zeros(ny-siz), a[nyqst:]])
    if siz // 2 == 0:
        b[nyqst-1] = b[nyqst-1] / 2
        b[nyqst-1+ny-siz] = b[nyqst-1]
    y = np.fft.ifft(b)
    y = np.real(y)
    y = y * ny / siz
    return y


def filter_decimate(sig, dec, fir_h):
    f_lpf = np.fft.fft(fir_h)
    f_sig = np.fft.fft(sig, axis=1)
    filter_out = np.fft.ifft(f_lpf * f_sig)
    return np.real(filter_out[:, ::dec])


# Decompose given matrix Z to eigenvalues&vectors
# But only r largest abs(eigenvalues)
# Z ≒ V * diag(d) * V'
def eig_r(Z, r):
    dz, Vz = np.linalg.eig(Z)

    ind = np.argsort(dz)
    ind = ind[::-1]

    d = dz[ind[:r]]
    V = Vz[:, ind[:r]]

    return V, d


def runOMPforMB(Y, A, N):
    _, nA = A.shape

    residual = Y
    oneside_supp = []
    supp = []
    symmetric_supp = []

    normAcols = np.sqrt(np.sum(np.abs(A)**2, axis=0))

    for step in range(N):
        # Matching Step
        Z_1 = A.T.conj() @ residual
        Z = np.sqrt(np.sum(np.abs(Z_1) ** 2, axis=1)) / normAcols
        best_loc = np.argmax(Z)
        # Update Support
        oneside_supp += [best_loc]
        symmetric_loc = nA - 1 - best_loc
        if best_loc != symmetric_loc:
            symmetric_supp += [symmetric_loc]
        supp = oneside_supp + symmetric_supp

        # Project residual
        As = A[:, supp]
        solution = As @ np.linalg.pinv(As) @ Y
        residual = Y - solution

    # Search adjacent positions
    '''
    for dominant_band in range(oneside_supp):
        check_supp = [
    '''

    return supp


def main():
    # ------------
    # Signal Model
    # ------------
    SNR = 10  # input SNR
    N = 6  # Number of bands
    B = 50e6  # Maximal width of each band
    Bi = np.ones(N//2) * B
    fnyq = 10e9
    Tnyq = 1 / fnyq
    Ei = np.random.rand(N//2) * 10  # Energy of the i'th band

    print('--------------------------------------------------')
    print('Signal Mode')
    print(f'    N = {N}, B = {B/1e6:.2f}MHz, fnyq = {fnyq/1e9:.2f}GHz')

    # -------------------
    # Sampling Parameters
    # -------------------
    alias_factor = 195
    fp = fnyq / alias_factor
    Tp = 1 / fp  # Period of mixing signals p_i(t).
    fs = fp      # Sampling rate
    Ts = 1 / fs
    m = 50       # Number of channels

    Mmin = 2 * math.ceil(0.5 * fnyq / fp + 0.5) - 1
    M = Mmin
    L0 = math.ceil((fnyq + fs) / 2 / fp) - 1
    L = 2 * L0 + 1

    sign_patterns = np.random.randint(low=0, high=2, size=[m, M]) * 2 - 1

    print('--------------------------------------------------')
    print('Sampling Parameters')
    print(f'    fp = {fp/1e6:.2f}MHz, m = {m}, M={M}')
    print(f'    L0 = {L0}, L = {L}, Tp = {Tp*1e6:.2f}uSec, Ts = {Ts*1e6:.2f}uSec')

    # -------------------------
    # Continuous Representation
    # -------------------------
    K = 91
    t_axis = np.arange(0, K*Tp, Tnyq)
    Taui = np.array([0.7, 0.4, 0.3]) * t_axis[-1]  # time offset of the i'th band

    print('--------------------------------------------------')
    print('Continuous Representation')
    print(f'    Time Window = [{t_axis[0]/1e-6:.2f}, {t_axis[-1]/1e-6:.2f}] uSec')
    print(f'    Time Resolution = {Tnyq/1e-9:.2f} nSec, Grid Length = {t_axis.size}')

    # -----------------
    # Signal Generation
    # -----------------
    x = np.zeros_like(t_axis)
    fi = np.random.rand(N//2) * (fnyq/2 - 2*B) + B  # Draw random carrier within [0, fnyq/2]

    for n in range(N//2):
        x += np.sqrt(Ei[n]) * np.sqrt(Bi[n]) * np.sinc(Bi[n] * (t_axis - Taui[n])) * \
            np.cos(2 * np.pi * fi[n] * (t_axis - Taui[n]))

    x *= np.hanning(x.size)

    # ------------------------------
    # Calculate Original Support Set
    # ------------------------------
    # [-fnyq/2, fnyq/2]을 "..... (-3*fp/2, fp/2], (-fp/2, fp/2], (fp/2, 3*fp/2] ....."
    # 작은 구간들로 쪼갬
    # (-fp/2, fp/2]가 L0+1번째 구간 (zero-based indexing으로는 L0)
    # 각 밴드 신호들이 존재하는 모든 구간의 union을 찾는다.
    supp_orig = np.array([])
    starts = np.ceil((fi - B/2) / fp - 0.5 + L0)
    ends = np.ceil((fi + B/2) / fp - 0.5 + L0)
    for i in range(N//2):
        supp_orig = np.union1d(supp_orig, np.arange(starts[i], ends[i]+1))

    # Add Negative Frequencies
    supp_orig = np.union1d(supp_orig, 2 * L0 - supp_orig)
    # array -> list
    supp_orig = list(supp_orig.astype(int))

    # -------------------------------------------
    # Noise Generation, AWGN within [-fnyq, fnyq]
    # -------------------------------------------
    noise_nyq = np.random.randn(t_axis.size)
    noise = noise_nyq  # resfactor 안써서 interpolation은 안함.

    # Calculate SNR
    noise_energy = np.linalg.norm(noise) ** 2
    signal_energy = np.linalg.norm(x) ** 2
    current_snr = signal_energy / noise_energy

    # ------
    # Mixing
    # ------
    mixed_sig_seqs = mix_signal(x, t_axis, sign_patterns, Tp)
    mixed_noise_seqs = mix_signal(noise, t_axis, sign_patterns, Tp)

    # ------------------------------
    # Analog LPF and Actual Sampling
    # ------------------------------
    tmp = np.zeros(K)
    tmp[0] = 1
    lpf_z = interpft(tmp, t_axis.size) / L

    decfactor = L
    digital_sig_samples = filter_decimate(mixed_sig_seqs, decfactor, lpf_z)
    digital_noise_samples = filter_decimate(mixed_noise_seqs, decfactor, lpf_z)

    digital_time_axis = t_axis[::decfactor]

    print('--------------------------------------------------')
    print('Sampling Summary')
    print(f'    {m} channels, each gives {digital_time_axis.size} samples')

    # ---------
    # CTF Block
    # ---------
    # Define S
    S = sign_patterns
    # Difine F
    theta = np.exp(-1j*2*np.pi/L)
    F = np.outer(np.arange(L), np.arange(-L0, L0+1))
    F = theta ** F
    # Define D
    # This is for digital input only,
    # Note that when R -> inf, D then coincides with that of the paper.
    D = np.diag(np.ones(L) / L)
    # Define A
    A = np.conj(S @ F @ D)

    # Combine Signal and Noise
    snr_val = 10 ** (SNR / 10)
    digital_samples = digital_sig_samples + digital_noise_samples * np.sqrt(current_snr / snr_val)

    # Frame construction
    Q = digital_samples @ digital_samples.T

    # Decompose Q to find frame V
    # 귀찮아서 NumDomEigvals 일단 Pass
    V, d = eig_r(Q, 2*N)
    v = V @ np.diag(np.sqrt(d))

    #
    rec_supp = runOMPforMB(v, A, N)
    rec_supp_sorted = sorted(rec_supp)
    print(rec_supp_sorted)
    print(supp_orig)


if __name__ == '__main__':
    main()
