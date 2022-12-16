# coding=UTF-8
from __future__ import print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
import sys
from scipy.signal import resample_poly, argrelmax
import matplotlib.ticker as tick  # 目盛り操作に必要なライブラリを読み込みます
from matplotlib.ticker import MultipleLocator
import os

# Mouse version
ESC_KEY = 0x1b  # Esc key
S_KEY = 0x73  # S key
R_KEY = 0x72  # R key

# 特徴点の最大数
MAX_FEATURE_NUM = 500
# 反復アルゴリズムの終了条件
CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
# ビデオデータ(引数から取得)
args = sys.argv
VIDEO_DATA = args[1]
dele = args[2] + "/"
FILE_NAME = VIDEO_DATA.lstrip(dele)
FILE_NAME = FILE_NAME.rstrip(".mov")
# ビデオデータと同じ名前のディレクトリ作成
os.makedirs(FILE_NAME, exist_ok=True)

#fps取得
cap = cv2.VideoCapture(VIDEO_DATA)  # パスを指定
fps = cap.get(cv2.CAP_PROP_FPS)
# インターバル （1000 / フレームレート）
INTERVAL = int(1000/fps)


class Paramecium:
    # コンストラクタ
    def __init__(self, verbose=False):
        # 表示ウィンドウ
        cv2.namedWindow("Paramecium")
        # マウスイベントのコールバック登録
        cv2.setMouseCallback("Paramecium", self.onMouse)
        # 映像
        self.video = cv2.VideoCapture(VIDEO_DATA)
        # インターバ
        self.interval = INTERVAL
        # 現在のフレーム（カラー）
        self.frame = None
        # 現在のフレーム（グレー）
        self.gray_next = None
        # 前回のフレーム（グレー）
        self.gray_prev = None
        #しきい値値処理した前回
        self.th_prev = None
        #しきい値処理した現在
        self.th_next = None
        # 特徴点
        self.features = None
        # 特徴点のステータス
        self.status = None
        # 色
        self.colors = ["b", "r", "m", "g", "c", "k", "y"]
        self.colors2 = ["b", "r", "m", "g", "c", "k", "y"]

        self.verbose = verbose

    # 周波数解析、ローパス（バターワース）フィルタ処理
    def calculate(self, i, data, fps):
        target = data[:, i]       # 5列目を抽出

        Nyk_fps = fps/2
        norm_pass = (Nyk_fps/2)/Nyk_fps     # 7.4925Hzの時
        norm_stop = (Nyk_fps/1.5)/Nyk_fps   # 9.99Hzの時
        # print("norm_pass:", norm_pass)
        # print("norm_stop:", norm_stop)
        N, Wn = signal.buttord(wp=norm_pass, ws=norm_stop, gpass=2, gstop=30)
        b, a = signal.butter(N, Wn, btype='low')
        after_lowpass = signal.filtfilt(b, a, target)
        after_lowpass=after_lowpass[:-2] # remove (0,0) at the end

        N = len(after_lowpass)                   # サンプル数
        F = np.fft.fft(after_lowpass)          # 高速フーリエ変換
        if self.verbose:
            print(F)
        F[0] = F[0]/2                           # 直流成分の振幅を揃える
        amp = [np.sqrt(c.real ** 2 + c.imag ** 2)
               for c in F]               # 振幅スペクトル
        # 周波数軸の値を計算
        freq = np.fft.fftfreq(len(after_lowpass), 1/fps)
        freq = freq[0:int(N/2)]                 # ナイキスト周波数の範囲内のデータのみ取り出し
        amp = amp[0:int(N/2)]
        return freq, amp, after_lowpass

    # 周波数解析の結果から、ピークとなる周波数を探す
    def freq_amp_peak_csv(self, amp, freq):
        array_amp = np.array(amp, dtype=np.float64)
        ind_max = argrelmax(array_amp)
        peak_freq = freq[ind_max]
        peak_amp = array_amp[ind_max]
        ind_max = np.array(ind_max)
        ind_max_1d = [flatten for inner in ind_max for flatten in inner]
        freq_amp = np.zeros((len(ind_max_1d), 2))
        for i in range(len(ind_max_1d)):
            freq_amp[i][0] = peak_freq[i]
            freq_amp[i][1] = peak_amp[i]
        return freq_amp

    # 周波数成分とその振幅を一つの配列にまとめる
    def freq_amp_csv(self, amp, freq):
        freq_amp = np.zeros((len(freq), 2))
        for i in range(len(freq)):
            freq_amp[i][0] = freq[i]
            freq_amp[i][1] = amp[i]
        return freq_amp

    def run(self):
        frame_n = 0
        #fps = 29.97   # fpsはこの値を使う！！！
        frame_list = np.empty(0)
        X_list_total = np.empty(0)
        Y_list_total = np.empty(0)
        speed_list_total = np.empty(0)
        freq_amp_X_peak_list = []
        freq_amp_Y_peak_list = []
        cap = cv2.VideoCapture(VIDEO_DATA)  # パスを指定
        print("loaded video: ", VIDEO_DATA)

        # 最初のフレームの処理
        end_flag, self.frame = self.video.read()
        self.gray_prev = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        ret, self.th_prev = cv2.threshold(
            self.gray_prev, 50, 255, cv2.THRESH_BINARY)

        cv2.imshow("Paramecium", self.frame)
        key = cv2.waitKey(0)

        if self.verbose:
            print("self.features=", self.features)

        if self.features is None:
            print("特徴点が登録されていません")
            exit()

        # 特徴点を中心に円を描く
        para_n=len(self.features)
        for feature in self.features:
            center = np.array((feature[0], feature[1])).astype(np.int32)
            cv2.circle(self.frame, center=center,
                        radius=16, color=(15, 241, 255), thickness=1, lineType=cv2.LINE_8, shift=0)

        # 特徴点の追跡（動画終了 or Escを押すまで）
        while end_flag:
            # グレースケールに変換
            self.gray_next = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            ret, self.th_next = cv2.threshold(
                self.gray_next, 50, 255, cv2.THRESH_BINARY)

            # オプティカルフローの計算
            features_prev = self.features
            self.features, self.status, err = cv2.calcOpticalFlowPyrLK(
                self.th_prev,
                self.th_next,
                features_prev,
                None,
                winSize=(10, 10),
                maxLevel=3,
                criteria=CRITERIA,
                flags=0)

            # 有効な特徴点のみ残す
            self.refreshFeatures()  # self.featuresには、最初のフレームでの特徴点は入っていない（２フレーム以降が格納されている）
            # しかし、速さは1フレーム目と2フレーム目の差から計算している（＝軌道データと速さデータは同じ長さ）

            # 引き続き特徴点がある場合は、速さを計算
            if len(self.features) != para_n:
                print("Lost some parameciums...")
                break

            speed_list = []

            for i in range(para_n):
                Vx = (self.features[i][0] -
                        features_prev[i][0])*float(args[3])*fps
                Vy = (self.features[i][1] -
                        features_prev[i][1])*float(args[4])*fps
                V = np.sqrt(np.square(Vx) + np.square(Vy))
                speed_list=np.append(speed_list, V)
                speed_list_total=np.append(speed_list_total, V)

            for para_id, feature in enumerate(self.features):
                # 特徴点を中心に円を描く
                center = np.array(
                    (feature[0], feature[1])).astype(np.int32)
                cv2.circle(self.frame, center=center,
                            radius=16, color=(15, 241, 255), thickness=1, lineType=cv2.LINE_8, shift=0)

                # 特徴点の(X,Y)を配列に格納
                X_list_total=np.append(X_list_total, feature[0])
                Y_list_total=np.append(Y_list_total, feature[1])
                if self.verbose:
                    print("appended feature")

                #リアルタイムでの遊泳軌跡描写
                plt.figure("monitor")
                if frame_n % 30 != 0:
                    plt.plot(
                        feature[0], feature[1], color=self.colors[para_id], marker="o", markersize=0.5)
                else:
                    plt.plot(
                        feature[0], feature[1], color="k", marker="D", markersize=3)

                plt.xlabel(str(frame_n/fps)+"  [s]")
                plt.title("(x, y)")
                plt.xlim(0, 640)
                plt.ylim(480, 0)
                #plt.show(block=False)
                plt.pause(0.001) # plt.show()をつかうと実行が止まるので、plt.pause()を使う
                frame_list=np.append(frame_list, frame_n/fps)
                
            frame_n += 1

            # 表示
            cv2.imshow("Paramecium", self.frame)

            # 次のループ処理の準備
            self.th_prev = self.th_next
            end_flag, self.frame = self.video.read()

            if end_flag:
                self.gray_next = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                ret, self.th_next = cv2.threshold(
                    self.gray_next, 50, 255, cv2.THRESH_BINARY)

            # インターバル
            key = cv2.waitKey(self.interval)
            # "Esc"キー押下で終了
            if key == ESC_KEY:
                break
            # "s"キー押下で一時停止
            elif key == S_KEY:
                self.interval = 0
            elif key == R_KEY:
                self.interval = INTERVAL
            
        fname = str(FILE_NAME)+"/"+str(FILE_NAME)+'_fig1.png'
        plt.savefig(fname)
        print("saved {}".format(fname))            

        #-------ここから追跡終了後の処理（直下は解析に使うデータや配列の作成：データの前処理）---------------------------
        print("# of params :", para_n)
        print("# of frames : ", frame_n)
        print("time length : {} [ms]".format(frame_n*fps))
        if para_n*(frame_n+1) != len(X_list_total):
            print("x_total: ", X_list_total.shape)
            print("y_total: ", Y_list_total.shape)
            print("[Warning] Some parameciums were lost during chasing...")

        # X座標データ（複数匹分）
        X_list_total = X_list_total.reshape((frame_n, para_n))
        # Y座標データ（複数匹分）
        Y_list_total = Y_list_total.reshape((frame_n, para_n))
        # 速さデータ（複数匹分）
        Speed_list_total = speed_list_total.reshape((frame_n, para_n))

        # 時間データ（単位は秒）
        reshape_frame = frame_list.reshape((frame_n, para_n))

        # 1秒ごとのX,Y座標データ
        # one_sec_X = np.array(one_sec_X)
        # one_sec_X = one_sec_X.reshape((int(frame_n/30),para_n))
        # one_sec_Y = np.array(one_sec_Y)
        # one_sec_Y =one_sec_Y.reshape((int(frame_n/30),para_n))

        #-------ここからデータ解析　-------------------------------------------------------------------------------
        # グラフ作成
        # fig = plt.figure(5, figsize=(10, 7.5))
        # gs = gridspec.GridSpec(17, 17)
        # plt.subplots_adjust(wspace=0.4, hspace=0.8)

        V_list = []
        # 追跡した全てのゾウリムシに対する行動解析
        for i in range(para_n):
            # 各ゾウリムシのX,Y座標や速さを格納する配列の作成（空の配列）
            XY_list = np.zeros((len(X_list_total), 3))
            original_speed = np.zeros((frame_n, 2))
            XY_lowpass = np.zeros((len(X_list_total)-2, 3))

            speed_lowpass_XY = np.zeros((frame_n-3, 2))
            Speed_lowpass = np.zeros((frame_n-5, 2))

            one_sec_X = []
            one_sec_Y = []

            # 遊泳軌跡の周波数解析, 軌道データのローパス（バタワース）処理
            freq_X, amp_X, after_lowpass_X = self.calculate(
                i, X_list_total, fps)
            freq_Y, amp_Y, after_lowpass_Y = self.calculate(
                i, Y_list_total, fps)

            print("after_lowpass_X:", after_lowpass_X.shape)
            # 軌道データ(X,Y)および速さの生データを前処理で作った配列に格納（生データ、ローパス処理後データ）
            for k in range(len(X_list_total)): # lowpass をかけると2個データが減るので、-2
                XY_list[k][0] = reshape_frame[k][0]
                XY_list[k][1] = X_list_total[k][i]
                XY_list[k][2] = Y_list_total[k][i]
                original_speed[k][0] = reshape_frame[k][0]
                original_speed[k][1] = Speed_list_total[k][i]                

            for k in range(len(after_lowpass_X)): # lowpass をかけると2個データが減るので、-2
                XY_lowpass[k][0] = reshape_frame[k][0]
                XY_lowpass[k][1] = after_lowpass_X[k]
                XY_lowpass[k][2] = after_lowpass_Y[k]  # !!!!ここからみなおす


            # 軌道データにdotをplotするために，一秒毎のデータを格納
            for k in range(len(after_lowpass_X)):
                if k == 0:
                   origin_X = XY_lowpass[k, 1]
                   origin_Y = XY_lowpass[k, 2]
                elif k % 30 == 0:
                   one_sec_X.append(XY_lowpass[k, 1])
                   one_sec_Y.append(XY_lowpass[k, 2])

            # ローパス処理後の軌道データから、速さを計算
            for k in range(len(after_lowpass_X)-1): #差分を取るので1へる
                Vx = (after_lowpass_X[k+1] -
                      after_lowpass_X[k])*float(args[3])*fps
                Vy = (after_lowpass_Y[k+1] -
                      after_lowpass_Y[k])*float(args[4])*fps
                V = np.sqrt(np.square(Vx) + np.square(Vy))
                speed_lowpass_XY[k][0] = reshape_frame[k][0]
                speed_lowpass_XY[k][1] = V
            

            freq_Speed, amp_Speed, after_lowpass_Speed = self.calculate(
                1, speed_lowpass_XY, fps)
            # print("fspeed",freq_Speed[-4:])
            # print("aspeed",amp_Speed[-4:])

            # print("X",after_lowpass_X[-4:])
            # print("shape X",after_lowpass_X.shape)

            # print("XY",speed_lowpass_XY[-4:])
            # print("shape speed_lowpass_XY",speed_lowpass_XY.shape)

            # print("Speed",after_lowpass_Speed[-4:])
            # print("shape after_lowpass_Speed",after_lowpass_Speed.shape) #len(speed_lowpass_XY - 2)

            for k in range(len(after_lowpass_Speed)):
                Speed_lowpass[k][0] = reshape_frame[k][0]
                Speed_lowpass[k][1] = after_lowpass_Speed[k]

            # 周波数解析の結果から、ピークとなる周波数を探す（ピークをとる周波数, その周波数での振幅）
            freq_amp_X = self.freq_amp_peak_csv(amp_X, freq_X)
            freq_amp_Y = self.freq_amp_peak_csv(amp_Y, freq_Y)
            for k in range(len(freq_amp_X)):
                freq_amp_X_peak_list.append(freq_amp_X[k])
            for k in range(len(freq_amp_Y)):
                freq_amp_Y_peak_list.append(freq_amp_Y[k])

            # 周波数成分とその振幅を一つの配列にまとめる
            freq_amp_X = self.freq_amp_csv(amp_X, freq_X)
            freq_amp_Y = self.freq_amp_csv(amp_Y, freq_Y)
            freq_amp_Speed = self.freq_amp_csv(amp_Speed, freq_Speed)

            #-------ここからデータ保存　-------------------------------------------------------------------------------
            # 生データ
            np.savetxt(str(FILE_NAME)+"/Para"+str(i)+"_" +
                       str(FILE_NAME)+"_XY.csv", XY_list, delimiter=",")
            np.savetxt(str(FILE_NAME)+"/Para"+str(i)+"_"+str(FILE_NAME) +
                       "_Speed.csv", original_speed[1:-1, :], delimiter=",")
            # ローパス処理後データ
            np.savetxt(str(FILE_NAME)+"/Para"+str(i)+"_"+str(FILE_NAME) +
                       "_XY_lowpass.csv", XY_lowpass, delimiter=",")
            np.savetxt(str(FILE_NAME)+"/Para"+str(i)+"_"+str(FILE_NAME) +
                       "_Speed_lowpass.csv", Speed_lowpass, delimiter=",")
            # 周波数解析の結果
            np.savetxt(str(FILE_NAME)+"/Para"+str(i)+"_"+str(FILE_NAME) +
                       "_freq_amp_X.csv", freq_amp_X, delimiter=",")
            np.savetxt(str(FILE_NAME)+"/Para"+str(i)+"_"+str(FILE_NAME) +
                       "_freq_amp_Y.csv", freq_amp_Y, delimiter=",")
            np.savetxt(str(FILE_NAME)+"/Para"+str(i)+"_"+str(FILE_NAME) +
                       "_freq_amp_Speed.csv", freq_amp_Speed, delimiter=",")
            # 周波数解析の結果から求めたピーク値
            np.savetxt(str(FILE_NAME)+"/Para"+str(i)+"_"+str(FILE_NAME) +
                       "_X_peak_Fourier.csv", freq_amp_X_peak_list, delimiter=",")
            np.savetxt(str(FILE_NAME)+"/Para"+str(i)+"_"+str(FILE_NAME) +
                       "_Y_peak_Fourier.csv", freq_amp_Y_peak_list, delimiter=",")

            if self.verbose:
                print("frame_listの長さ", len(frame_list))
                print("original_speedの長さ", len(original_speed))
                print("Speed_lowpassの長さ", len(Speed_lowpass))

            #-------ここからグラフ作成　-------------------------------------------------------------------------------
            # 軌道(X,Y)
            plt.figure("Trajectory (x,y)")
            plt.title("Speed average: " +
                      str(sum(after_lowpass_Speed)/frame_n-1)+"μm/s")
            plt.plot(XY_lowpass[:, 1], XY_lowpass[:, 2],
                     color=self.colors[i], linewidth=0.5)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.savefig(str(FILE_NAME)+"/Para"+str(i)+"_" +
                        str(FILE_NAME)+'_XY_lowpass_zoom.png')


            # 軌道(X,Y):1秒毎にdotをplot
            plt.figure("Trajectory (x,y) with position per sec")
            plt.plot(XY_lowpass[:, 1], XY_lowpass[:, 2],
                     color=self.colors[i], linewidth=0.5)
            plt.title("trajectory")
            plt.xlabel("x")
            plt.ylabel("y")

            cm = plt.get_cmap('copper')
            cm_interval = [i / (len(one_sec_X))
                           for i in range(1, len(one_sec_X)+1)]  # ここを変える !!!!!!
            cm = cm(cm_interval)
            plt.scatter(one_sec_X, one_sec_Y, c=cm, alpha=0.7)
            plt.scatter(origin_X, origin_Y, color="k",
                        marker="*", s=100, alpha=0.5)
            plt.grid()
            plt.savefig(str(FILE_NAME)+"/Para"+str(i) +
                        "_"+str(FILE_NAME)+"_XY_dots.png")

            # パワースペクトル（X方向）
            fig = plt.figure("power spectrum")
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.plot(freq_X[1:-1], amp_X[1:-1], color=self.colors[i])
            ax1.set_xlabel("Fequency(X)")
            ax1.set_ylabel("Amplitude(X)")
            ax1.set_xlim(0, 8)
            plt.grid()

            # パワースペクトル（Y方向）
            ax2 = fig.add_subplot(2, 1, 2)
            ax2.plot(freq_Y[1:-1], amp_Y[1:-1], color=self.colors[i])
            ax2.set_xlabel("Fequency(Y)")
            ax2.set_ylabel("Amplitude(Y)")
            ax2.set_xlim(0, 8)
            plt.grid()

            if (i == para_n-1):
                plt.savefig(str(FILE_NAME)+"/"+str(FILE_NAME) +
                            "_Power_spectrum.png")

            # 生データでの速さ
            plt.figure("speed (raw data)")
            plt.plot(original_speed[:, 0],
                     original_speed[:, 1], color="k")
            plt.xlabel("time (s)")
            plt.ylabel("speed")
            plt.grid()
            plt.savefig(str(FILE_NAME)+"/Para"+str(i)+"_" +
                        str(FILE_NAME)+"_original_Speed.png")

            # ローパス処理後の速さ
            plt.figure("speed (lowpass data))")
            # ここも軌道データとフレーム数同じ！？？？
            plt.plot(Speed_lowpass[:, 0],
                     Speed_lowpass[:, 1], color="b")
            plt.xlabel("time (s)")
            plt.ylabel("speed")
            plt.grid()
            plt.savefig(str(FILE_NAME)+"/Para"+str(i)+"_" +
                        str(FILE_NAME)+"_Speed_lowpass.png")

            # ローパス処理前と処理後の速さ比較
            plt.figure("speed (raw vs lowpass data)")
            plt.plot(original_speed[:, 0],
                     original_speed[:, 1], color="k")
            # ここも軌道データとフレーム数同じ！？？？
            plt.plot(Speed_lowpass[:, 0],
                     Speed_lowpass[:, 1], color=self.colors[i])
            plt.xlabel("time (s)")
            plt.ylabel("speed")
            plt.grid()
            plt.savefig(str(FILE_NAME)+"/Para"+str(i)+"_" +
                        str(FILE_NAME)+"_Speed_lowpass_check.png")

            plt.show()

        self.video.release()
        cv2.destroyAllWindows()

    # マウスクリックで特徴点を指定する
    # クリックされた近傍に既存の特徴点がある場合は既存の特徴点を削除する
    # クリックされた近傍に既存の特徴点がない場合は新規に特徴点を追加する
    def onMouse(self, event, x, y, flags, param):
        # 左クリック以外
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # 探索半径（pixel）
        radius = 15
        cv2.circle(self.frame, center=(x,y),
                    radius=radius, color=(0, 241, 0), thickness=1, lineType=cv2.LINE_8, shift=0)
        cv2.imshow("Paramecium", self.frame)        
        
        # 最初の特徴点追加
        if self.features is None:
            self.addFeature(x, y)
            return

        # 既存の特徴点が近傍にあるか探索
        index = self.getFeatureIndex(x, y, radius)

        # クリックされた近傍に既存の特徴点がある場合、既存の特徴点を削除する
        if index >= 0:
            self.features = np.delete(self.features, index, 0)
            self.status = np.delete(self.status, index, 0)
            print("今選択した特徴点は削除されました。")

        # クリックされた近傍に既存の特徴点がないので新規に特徴点を追加する
        else:
            self.addFeature(x, y)
        return

    # 指定した半径内にある既存の特徴点のインデックスを１つ取得する
    #     指定した半径内に特徴点がない場合 index = -1 を応答
    def getFeatureIndex(self, x, y, radius):
        index = -1
        # 特徴点が１つも登録されていない
        if self.features is None:
            return index

        max_r2 = radius ** 2
        index = 0
        for point in self.features:
            dx, dy = x - point[0], y - point[1]
            r2 = dx ** 2 + dy ** 2
            if r2 <= max_r2:
                # この特徴点は指定された半径内
                return index
            else:
                # この特徴点は指定された半径外
                index += 1
        # 全ての特徴点が指定された半径の外側にある
        return -1

    # 特徴点を新規に追加する
    def addFeature(self, x, y):
        # 特徴点が未登録
        if self.features is None:
            # ndarrayの作成し特徴点の座標を登録
            self.features = np.array([[x, y]], np.float32)
            self.status = np.array([1])
            # 特徴点を高精度化
            cv2.cornerSubPix(self.th_prev, self.features,
                             (10, 10), (-1, -1), CRITERIA)

        # 特徴点の最大登録個数をオーバー
        elif len(self.features) >= MAX_FEATURE_NUM:
            print("max feature num over: " + str(MAX_FEATURE_NUM))

        # 特徴点を追加登録
        else:
            # 既存のndarrayの最後に特徴点の座標を追加
            self.features = np.append(
                self.features, [[x, y]], axis=0).astype(np.float32)
            self.status = np.append(self.status, 1)
            # 特徴点を高精度化
            cv2.cornerSubPix(self.th_prev, self.features,
                             (10, 10), (-1, -1), CRITERIA)

    # 有効な特徴点のみ残す
    def refreshFeatures(self):
        # 特徴点が未登録
        if self.features is None:
            return
        # 全statusをチェックする
        i = 0
        while i < len(self.features):

            # 特徴点として認識できず
            if self.status[i] == 0:
                # 既存のndarrayから削除
                self.features = np.delete(self.features, i, 0)
                self.status = np.delete(self.status, i, 0)
                i -= 1
            i += 1


if __name__ == '__main__':
    Paramecium().run()

# 実行コマンド　python Patra.py 解析したいディレクトリ/解析したいファイル　解析したいディレクトリ
