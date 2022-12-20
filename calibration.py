# coding=UTF-8
from __future__ import print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
import sys
from scipy.signal import resample_poly, argrelmax
import matplotlib.ticker as tick # 目盛り操作に必要なライブラリを読み込みます
from matplotlib.ticker import MultipleLocator
import os

# Mouse version
# Esc キー
ESC_KEY = 0x1b
# s キー
S_KEY = 0x73
# r キー
R_KEY = 0x72

# 特徴点の最大数
MAX_FEATURE_NUM = 500
# 反復アルゴリズムの終了条件
CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
# ビデオデータ(引数から取得)
args = sys.argv
VIDEO_DATA = args[1]

class Paramecium:
    # コンストラクタ
    def __init__(self):
        # 表示ウィンドウ
        cv2.namedWindow("Paramecium")  
        # マウスイベントのコールバック登録
        cv2.setMouseCallback("Paramecium", self.onMouse)
        # 映像
        self.video = cv2.VideoCapture(VIDEO_DATA)
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

    def run(self):
        frame_n = 1
        cap = cv2.VideoCapture(VIDEO_DATA) # パスを指定
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("fps=",fps)
        frame_list = []
        frame_list.append(1/fps)
        frame_list_number = []
        frame_list_number.append(frame_n)
        X_list_total = []
        Y_list_total = []
        speed_list_total = []
        freq_amp_X_peak_list = []
        freq_amp_Y_peak_list = []

        #グラフの準備
        plt.figure(1)
        gs = gridspec.GridSpec(14,14)

        # 最初のフレームの処理
        end_flag, self.frame = self.video.read()
        self.gray_prev = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        ret,self.th_prev = cv2.threshold(self.gray_prev,50,255,cv2.THRESH_BINARY)
        cv2.imshow("Paramecium",self.frame)
        key = cv2.waitKey(0)
        print(self.features)    

        # ピクセルからマイクロメートルに単位を変換するための定数(速さにこの数をかけたら、マイクロメートルになる！！)
        X_scale = 1000*5/abs(self.features[2][0]-self.features[3][0])
        Y_scale = 1000*5/abs(self.features[0][1]-self.features[1][1])   # 縦上、縦下（Y座標:0-5mm）、横左、横右（X座標:0-5mm）

        print("X_scale:",X_scale)
        print("Y_scale:",Y_scale)

        # 特徴点が登録された場合、その点を中心に円を描く
        if self.features is not None:
            for feature in self.features:
                cv2.circle(self.frame, (feature[0], feature[1]), 16, (15, 241, 255), 1, 8, 0)          
            # 表示
            cv2.imshow("Paramecium", self.frame)
        cv2.destroyAllWindows()

    # マウスクリックで特徴点を指定する
    # クリックされた近傍に既存の特徴点がある場合は既存の特徴点を削除する
    # クリックされた近傍に既存の特徴点がない場合は新規に特徴点を追加する
    def onMouse(self, event, x, y, flags, param):
        # 左クリック以外
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # 最初の特徴点追加
        if self.features is None:
            self.addFeature(x, y)
            return

        # 探索半径（pixel）
        radius = 5
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
            dx = x - point[0]
            dy = y - point[1]
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
            cv2.cornerSubPix(self.th_prev, self.features, (10, 10), (-1, -1), CRITERIA)

        # 特徴点の最大登録個数をオーバー
        elif len(self.features) >= MAX_FEATURE_NUM:
            print("max feature num over: " + str(MAX_FEATURE_NUM))

        # 特徴点を追加登録
        else:
            # 既存のndarrayの最後に特徴点の座標を追加
            self.features = np.append(self.features, [[x, y]], axis = 0).astype(np.float32)
            self.status = np.append(self.status, 1)
            # 特徴点を高精度化
            cv2.cornerSubPix(self.th_prev, self.features, (10, 10), (-1, -1), CRITERIA)

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

