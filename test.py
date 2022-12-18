import numpy as np
from scipy import signal
from scipy.signal import resample_poly, argrelmax


def calculate(target, fps):
    Nyk_fps = fps/2
    norm_pass = (Nyk_fps/2)/Nyk_fps     # 7.4925Hzの時
    norm_stop = (Nyk_fps/1.5)/Nyk_fps   # 9.99Hzの時
    N, Wn = signal.buttord(wp=norm_pass, ws=norm_stop, gpass=2, gstop=30)
    b, a = signal.butter(N, Wn, btype='low')
    after_lowpass = signal.filtfilt(b, a, target)

    print("data length", len(target))
    print("after_lowpass length", len(after_lowpass))
    print(after_lowpass[:2],
          after_lowpass[-2:])
#    after_lowpass = after_lowpass[:-2]  # remove (0,0) at the end

    N = len(after_lowpass)                   # サンプル数
    print("N:", N)
    F = np.fft.fft(after_lowpass)          # 高速フーリエ変換
    F[0] = F[0]/2                           # 直流成分の振幅を揃える
    amp = [np.sqrt(c.real ** 2 + c.imag ** 2)
            for c in F]               # 振幅スペクトル
    # 周波数軸の値を計算
    freq = np.fft.fftfreq(len(after_lowpass), 1/fps)
    freq = freq[0:int(N/2)]                 # ナイキスト周波数の範囲内のデータのみ取り出し
    amp = amp[0:int(N/2)]

    print("len(freq):", len(freq))
    print("freq:", freq[:2], freq[-2:])

    return freq, amp, after_lowpass

data=np.sin(range(1000))
freq, amp, after_lowpass = calculate(data, 100)