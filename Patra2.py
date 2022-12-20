# coding=UTF-8
from __future__ import print_function
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
import sys
from scipy.signal import resample_poly, argrelmax
import matplotlib.ticker as tick  # 目盛り操作に必要なライブラリを読み込みます
from matplotlib.ticker import MultipleLocator
import os
import warnings
warnings.filterwarnings('ignore')

# Mouse version
ESC_KEY = 0x1b  # Esc key
S_KEY = 0x73  # S key
R_KEY = 0x72  # R key

# 特徴点の最大数
MAX_FEATURE_NUM = 500
# 反復アルゴリズムの終了条件
CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)


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
        # しきい値値処理した前回
        self.th_prev = None
        # しきい値処理した現在
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
    def lpf(self, data, fps):
        nonans = data.dropna()
        Nyk_fps = fps/2
        norm_pass = (Nyk_fps/2)/Nyk_fps     # 7.4925Hzの時
        norm_stop = (Nyk_fps/1.5)/Nyk_fps   # 9.99Hzの時

        N, Wn = signal.buttord(wp=norm_pass, ws=norm_stop, gpass=2, gstop=30)
        b, a = signal.butter(N, Wn, btype='low')
        after_lowpass = signal.filtfilt(b, a, nonans)
        #after_lowpass = after_lowpass[:-2]  # remove (0,0) at the end

        return(after_lowpass)

    def freq_analysis(self, data, fps):
        N = len(data)                   # サンプル数
        F = np.fft.fft(data)          # 高速フーリエ変換
        F[0] = F[0]/2                           # 直流成分の振幅を揃える
        amp = [np.sqrt(c.real ** 2 + c.imag ** 2)
               for c in F]               # 振幅スペクトル
        # 周波数軸の値を計算
        freq = np.fft.fftfreq(len(data), 1/fps)
        freq = freq[0:int(N/2)]                 # ナイキスト周波数の範囲内のデータのみ取り出し
        amp = amp[0:int(N/2)]
        return freq, amp

    def get_freq_amp_peak(self, freq, amp):
    # 周波数解析の結果から、ピークとなる周波数を探す
        np_amp = np.array(amp, dtype=np.float64)
        np_freq = np.array(freq, dtype=np.float64)

        ind_max = argrelmax(np_amp)
        peak_freq = np_freq[ind_max]
        peak_amp = np_amp[ind_max]
        #ind_max = np.array(ind_max)

        df_ret=pd.DataFrame({'freq':peak_freq, 'amp':peak_amp})
        return df_ret

    def getOptFlow(self, features_prev):
        gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        ret, self.th_next = cv2.threshold(
            gray_frame, 50, 255, cv2.THRESH_BINARY)

        # オプティカルフローの計算
        features, status, err = cv2.calcOpticalFlowPyrLK(
            self.th_prev,
            self.th_next,
            features_prev,
            None,
            winSize=(10, 10),
            maxLevel=3,
            criteria=CRITERIA,
            flags=0)

        # 特徴点が未登録の場合
        if features is None:
            return

        # 全statusをチェックする
        i = 0
        while i < len(features):
            # 特徴点として認識できず
            if status[i] == 0:
                # 既存のndarrayから削除
                features = np.delete(features, i, 0)
                status = np.delete(status, i, 0)
                i -= 1
            i += 1

        return features, status

    def onMouse(self, event, x, y, flags, param):
        '''
        マウスクリックで特徴点を指定する
        クリックされた近傍に既存の特徴点がある場合は既存の特徴点を削除する
        クリックされた近傍に既存の特徴点がない場合は新規に特徴点を追加する
        '''

        # 左クリック以外
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # 探索半径（pixel）
        radius = 15

        # draw circle
        cv2.circle(self.frame, center=(x,y),
                   radius=radius, color=(0, 241, 0), thickness=1, lineType=cv2.LINE_8, shift=0)
        cv2.imshow("Paramecium", self.frame)

        # 既存の特徴点が近傍にあるか探索
        index = self.getFeatureIndex(pos=[x, y], radius=radius)

        # クリックされた近傍に既存の特徴点がある場合、既存の特徴点を削除する
        if index >= 0:
            self.features = np.delete(self.features, index, 0)
            self.status = np.delete(self.status, index, 0)
            print("今選択した特徴点は削除されました。")

        # クリックされた近傍に既存の特徴点がないので新規に特徴点を追加する
        else:
            self.addFeature(x, y)
        return

    def getFeatureIndex(self, pos, radius):
        '''
        指定した半径内にある既存の特徴点のインデックスを１つ取得する
        指定した半径内に特徴点がない場合 index = -1 を返り値にする       
        '''
        # 特徴点が１つも登録されていない
        if self.features is None:
            return -1

        for index, point in enumerate(self.features):
            delta = np.array(pos)-np.array(point)
            if np.linalg.norm(delta, ord=2) <= radius:
                return index

        # 全ての特徴点が指定された半径の外側にある
        return -1

    def addFeature(self, x, y):
        '''
        特徴点を新規に追加する
        '''
        # 特徴点が未登録の場合
        if self.features is None:
            self.features = np.array([[x, y]], np.float32)
            self.status = np.array([1])
            # 特徴点を高精度化
            cv2.cornerSubPix(self.th_prev, self.features,
                            (10, 10), (-1, -1), CRITERIA)
            return 1

        # 特徴点の最大登録個数をオーバー
        elif len(self.features) >= MAX_FEATURE_NUM:
            print("max feature num over: " + str(MAX_FEATURE_NUM))
            return -1

        # 既存のndarrayの最後に特徴点の座標を追加
        self.features = np.append(
            self.features, [[x, y]], axis=0).astype(np.float32)
        self.status = np.append(self.status, 1)
        # 特徴点を高精度化
        cv2.cornerSubPix(self.th_prev, self.features,
                        (10, 10), (-1, -1), CRITERIA)

        return 1

    def get_frame(self):
        end_flag, frame = self.video.read()
        th_image=None
        if end_flag:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, th_image = cv2.threshold(
                gray_frame, 50, 255, cv2.THRESH_BINARY)

        return frame, th_image, end_flag

    def run(self):
        frame_n = 0
        frame_list = np.empty(0)
        cols=['time','para_id','X', 'Y']
        df_pos=pd.DataFrame([], columns=cols)

        cap = cv2.VideoCapture(VIDEO_DATA)  # パスを指定
        print("loaded video: ", VIDEO_DATA)

        # 最初のフレームの処理
        self.frame, self.th_prev, end_flag = self.get_frame()

        cv2.imshow("Paramecium", self.frame)
        key = cv2.waitKey(0)

        if self.features is None:
            print("特徴点が登録されていません")
            exit()

        # skip some frames to get stable tracking
        for i in range(3):
            # 次のループ処理の準備
            self.frame, self.th_next, end_flag = self.get_frame()
            features_prev = self.features
            self.features, self.status = self.getOptFlow(
                features_prev=features_prev)
            self.th_prev=self.th_next

        # 特徴点を中心に円を描く
        para_n = len(self.features)
        for feature in self.features:
            center = np.array((feature[0], feature[1])).astype(np.int32)
            cv2.circle(self.frame, center=center,
                       radius=16, color=(15, 241, 255), thickness=1, lineType=cv2.LINE_8, shift=0)

        # 特徴点の追跡（動画終了 or Escを押すまで）
        while end_flag:
            features_prev = self.features
            self.features, self.status = self.getOptFlow(
                features_prev=features_prev)

            # 引き続き特徴点がある場合は、速さを計算
            if len(self.features) != para_n:
                print("Lost some parameciums...")
                break

            for para_id, feature in enumerate(self.features):
                # 特徴点を中心に円を描く
                center = np.array(
                    (feature[0], feature[1])).astype(np.int32)
                cv2.circle(self.frame, center=center,
                           radius=16, color=(15, 241, 255), thickness=1, lineType=cv2.LINE_8, shift=0)

                # 特徴点データ(X,Y)を配列に格納
                df_new=pd.DataFrame([[frame_n/fps,para_id,feature[0],feature[1]]],columns=cols)
                df_pos=pd.concat([df_pos, df_new],axis=0,ignore_index=True)

                # リアルタイムでの遊泳軌跡描写
                plt.figure("monitor")
                if frame_n % fps != 0:
                    plt.plot(feature[0], feature[1], color=self.colors[para_id], marker="o", markersize=0.5)
                else:
                    plt.plot(feature[0], feature[1], color="k", marker="o", markersize=0.5)
                plt.xlabel("{:5.3f} [s]".format(frame_n/fps))
                plt.title("(x, y)")
                plt.xlim(0, 640)
                plt.ylim(480, 0)
                #plt.grid()
                plt.pause(0.001)  # plt.show()をつかうと実行が止まるので、plt.pause()を使う
                frame_list = np.append(frame_list, frame_n/fps)

            frame_n += 1

            # 表示
            cv2.imshow("Paramecium", self.frame)

            # 次のループ処理の準備
            self.th_prev = self.th_next
            self.frame, self.th_next, end_flag = self.get_frame()

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

        fname=distroot+str(FILE_NAME)+"/"+str(FILE_NAME)+'_fig1.png'
        plt.savefig(fname)
        print("saved {}".format(fname))
        print("# of params :", para_n)
        print("# of frames : ", frame_n)
        print("time length : {} [ms]".format(frame_n*fps))

        self.analyze_data(df_pos, para_n)

        self.video.release()
        cv2.destroyAllWindows()

    def analyze_data(self, df_pos, para_n):
        '''
        Get low passed trajectory, speed, and frequency distribution data
        '''
        # 追跡した全てのゾウリムシに対する行動解析
        df_pos['X']=df_pos['X']*CALX
        df_pos['Y']=df_pos['Y']*CALY
        for para_id in range(para_n):
            # 各ゾウリムシのX,Y座標や速さを格納する配列の作成（空の配列）
            df_traj=df_pos[df_pos['para_id']==para_id]
            df_freq=pd.DataFrame([])

            # 軌道データ(X,Y)（生データ）
            df_traj.set_index('time', inplace=True)

            # ローパスフィルターをかける + 周波数解析
            for col in ['X','Y']:
                df_traj[col+'_lpf']=self.lpf(df_traj[col], fps)

                freq, amp = self.freq_analysis(df_traj[col+'_lpf'], fps)
                df_freq[col+'_freq']=pd.Series(freq)
                df_freq[col+'_amp']=pd.Series(amp)

            # 速度の計算
            df_vel = df_traj.diff()
            df_vel = df_vel.rename(
                columns={'X': 'Vx', 'Y': 'Vy', 'X_lpf': 'Vx_lpf',
                         'Y_lpf': 'Vy_lpf'}).drop(columns='para_id')
            
            df_vel*=fps
            df_traj=pd.concat([df_traj, df_vel], axis=1)
            df_traj['V']=np.sqrt(np.square(df_traj['Vx'])+np.square(df_traj['Vy']))                  
            df_traj['V_lpf']=np.sqrt(np.square(df_traj['Vx_lpf'])
                +np.square(df_traj['Vy_lpf']))                  

            # 速度V_lpfにローパスフィルターをかける + 周波数解析
            ret= self.lpf(df_traj['V_lpf'], fps)
            ret=pd.Series(ret, name='V_lpf2')
            ret.index=df_traj['V_lpf'].dropna().index
            df_traj =pd.concat([df_traj, ret], axis=1)

            freq, amp = self.freq_analysis(df_traj['V_lpf2'].iloc[1:], fps)
            df_freq['V_lpf2_freq']=pd.Series(freq)
            df_freq['V_lpf2_amp']=pd.Series(amp)

            # 周波数解析の結果から、ピークとなる周波数を探す（ピークをとる周波数, その周波数での振幅）
            df_freq_peak_X = self.get_freq_amp_peak(df_freq['X_freq'], df_freq['X_amp'])
            df_freq_peak_Y = self.get_freq_amp_peak(df_freq['Y_freq'], df_freq['Y_amp'])
            df_freq_peak_V = self.get_freq_amp_peak(df_freq['V_lpf2_freq'], df_freq['V_lpf2_amp'])

            # -------データ保存-------------------------------------------
            distpath = distroot+str(FILE_NAME)+"/Para"+str(para_id)+"_"+str(FILE_NAME)

            df_traj.to_csv(distpath+".csv", index=True)
            df_freq.to_csv(distpath+"_freq.csv", index=True)
            df_freq_peak_X.to_csv(distpath+"_X_peak_freq.csv", index=True)
            df_freq_peak_Y.to_csv(distpath+"_Y_peak_freq.csv", index=True)
            df_freq_peak_V.to_csv(distpath+"_V_peak_freq.csv", index=True)


            self.plot_graph(df_traj, df_freq, para_id, distpath, pltshow=(para_id==para_n-1))

        return 0

    def plot_graph(self, df_traj, df_freq, para_id, distpath, pltshow=True):
        def plot(x, y, x2=-1, y2=None,
                    color=None, color2=None,
                    linewidth=0.5, linewidth2=0.5,
                    xlabel="", ylabel="",
                    figtitle="", title="", save_name=None):
            plt.figure(figtitle)
            plt.title(title)
            plt.plot(x, y, color=color, linewidth=linewidth)
            if type(x2) != int:
                plt.plot(
                    x2, y2, color=color2, linewidth=linewidth2)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid()
            if save_name != None:
                plt.savefig(save_name)

        # 軌道(X,Y)
        # figtitle = "Trajectory ({})".format(para_id)
        # plot(
        #     x=df_traj['X_lpf'], y=df_traj['Y_lpf'],
        #     color=self.colors[para_id], linewidth=0.5,
        #     xlabel="x", ylabel="y",
        #     title="trajectory",
        #     figtitle=figtitle,
        #     save_name=distpath+'_XY_lowpass_zoom.png'
        #     )

        # 軌道(X,Y):1秒毎にdotをplot
        figtitle = "Trajectory ({})".format(para_id)
        # figtitle = "Trajectory (x,y) ({}) with position per sec".format(para_id)
        plot(
            x=df_traj['X_lpf'], y=df_traj['Y_lpf'],
            color=self.colors[para_id], linewidth=0.5,
            xlabel="x ($\mu$m)", ylabel="y ($\mu$m)",
            title="trajectory",
            figtitle=figtitle,
            save_name=distpath+'_XY_lowpass_zoom.png'
            )

        cm=plt.get_cmap('copper') 
        cm_interval=[ i / (len(df_traj['X_lpf'][::int(fps)])) for i in \
            range(1,len(df_traj['X_lpf'][::int(fps)])+1) ]
        cm=cm(cm_interval)
        plt.scatter(df_traj['X_lpf'][::int(fps)], df_traj['Y_lpf'][::int(fps)], 
            c=cm, alpha=0.7)
        plt.grid()
        plt.savefig(distpath+"_XY_dots.png")

        # # 生データでの速さ
        # figtitle = "speed (raw data) ({})".format(para_id)
        # plot(x=df_traj.index, y=df_traj['V'],
        #         color="k",
        #         title="speed (raw data)",
        #         figtitle=figtitle,
        #         xlabel="time (s)", ylabel="speed",
        #         save_name=distpath+"_original_Speed.png")

        # # ローパス処理後の速さ
        # figtitle = "speed (lowpass data)) ({})".format(para_id)
        # plot(x=df_traj.index, y=df_traj['V_lpf2'],
        #         color="b",
        #         title="speed (lowpass data)",
        #         figtitle=figtitle,
        #         xlabel="time (s)", ylabel="speed",
        #         save_name=distpath+"_Speed_lowpass.png")

        # # ローパス処理前と処理後の速さ比較
        # figtitle = "speed (raw vs lowpass data) ({})".format(para_id)
        # plot(x=df_traj.index, y=df_traj['V'],
        #         x2=df_traj.index, y2=df_traj['V_lpf2'],
        #         color="k", color2=self.colors[para_id],
        #         title="speed (raw vs lowpass data)",
        #         figtitle=figtitle,
        #         xlabel="time (s)", ylabel="speed",
        #         save_name=distpath+"_Speed_lowpass_check.png")

        # speed
        fig = plt.figure("speed ({})".format(para_id))
        fig.subplots_adjust(hspace=0.8, wspace=0.4)
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(df_traj.index, df_traj['V'], color="k")
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("raw speed ($\mu$m/s)")
        plt.grid()
        #save_name=distpath+"_original_Speed.png"
        #extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        #fig.savefig(save_name, bbox_inches=extent.expanded(1.3, 1.6))


        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(df_traj.index, df_traj['V_lpf2'], color="b")
        ax2.set_xlabel("time (s)")
        ax2.set_ylabel("lpf speed ($\mu$m/s)")
        plt.grid()
        #save_name=distpath+"_Speed_lowpass.png"
        #extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        #fig.savefig(save_name, bbox_inches=extent.expanded(1.3, 1.6))

        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(df_traj.index, df_traj['V'], color="k")
        ax3.plot(df_traj.index, df_traj['V_lpf2'], color="b")
        ax3.set_xlabel("time (s)")
        ax3.set_ylabel("raw vs lpf speed ($\mu$m/s)")
        plt.grid()
        #save_name=distpath+"_Speed_lowpass_check.png"
        #extent = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        #fig.savefig(save_name, bbox_inches=extent.expanded(1.3, 1.6))
        save_name=distpath+"_Speed.png"
        fig.savefig(save_name)


        # パワースペクトル（X,Y方向それぞれsubfigureに）
        fig = plt.figure("power spectrum ({})".format(para_id))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)

        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(df_freq['X_freq'][1:], df_freq['X_amp'][1:], color=self.colors[para_id])
        ax1.set_xlabel("Fequency(X)")
        ax1.set_ylabel("Amplitude(X)")
        ax1.set_xlim(0, 8)
        plt.grid()

        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(df_freq['Y_freq'][1:], df_freq['Y_amp'][1:], color=self.colors[para_id])
        ax2.set_xlabel("Fequency(Y)")
        ax2.set_ylabel("Amplitude(Y)")
        ax2.set_xlim(0, 8)
        plt.grid()

        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(df_freq['V_lpf2_freq'][1:], df_freq['V_lpf2_amp'][1:], color=self.colors[para_id])
        ax3.set_xlabel ("Fequency(V)")
        ax3.set_ylabel ("Amplitude(V)")
        ax3.set_xlim(0, 8)
        plt.grid()
        plt.savefig(distpath+"_Power_spectrum.png")

        if pltshow:
            plt.show()

if __name__ == '__main__':
    '''
    python Patra.py [video_data] [calx] [caly]
    '''
    args = sys.argv
    VIDEO_DATA = args[1]
    CALX = float(args[2]) # micro m/pixel
    CALY = float(args[3]) # micro m/pixel

    FILE_NAME=VIDEO_DATA.split("/")[-1].split(".")[0]

    # ビデオデータと同じ名前のディレクトリ作成
    distroot = "outputs/"
    os.makedirs("outputs/"+FILE_NAME, exist_ok=True)

    # fps取得
    cap = cv2.VideoCapture(VIDEO_DATA)  # パスを指定
    fps = cap.get(cv2.CAP_PROP_FPS)
    # インターバル （1000 / フレームレート）
    INTERVAL = int(1000/fps)

    print("file path: ",VIDEO_DATA)
    print("file name: ",FILE_NAME)
    print("output folder: ", distroot)
    print("fps: ", fps)
    print("INTERVAL(1/fps): ", INTERVAL)

    Paramecium().run()

