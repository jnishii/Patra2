# Patra2のインストール

ゾウリムシの追跡プログラム

- https://github.com/bcl-group/motion-capture からの派生
- 上記motion-captureは[OpenCVを使ったモーション テンプレート解析](https://qiita.com/hitomatagi/items/a4ecf7babdbe710208ae)を参考に作成

## 環境構築

Python環境のインストール

```
$ pyenv install 3.10.2
$ pyenv rehash
$ pyenv global 3.10.2 # インストールしたPython 3.10.2 を利用するための設定
```

確認

```
$ pyenv versions
	system
* 3.10.2 (set by /Users/jun/.pyenv/version)
$ which python
  /Users/ユーザ名/.pyenv/shims/python
$ python --version
Python 3.10.2
```

Poetryのインストール

```
$ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

Patraのダウンロード

```
git clone git@github.com:jnishii/patra2.git
```

必要なライブラリのインストール

```
$ cd patra2
$ poetry install
```

## Patra2の使い方
### 使用上の注意

- **解析動画のフレームレートは整数化して処理**します。
  - 小数部分がある場合，1秒ごとのゾウリムシの位置表示に影響します
  - 記録データの時刻情報は，小数部分も計算しています
- 解析したいファイルは patara2/Data/ の下に置く。サブフォルダを作っても良い。
- 出力ファイルは patra2/Data/ の下に作られるサブフォルダ(ファイル名と同じ名前)に保存される。

### 解析の順番

- キャリブレーション（calibration.py）
- ゾウリムシの追跡 & 行動解析 (Patra2.py)
- 速さのフーリエ解析（Speed_fourier.py）

### キャリブレーションの取り方 (calibration.py)

- python calibration.py マイクロメータの動画ファイル で実行
- 5mm × 5mm(1目盛り50μm) のマイクロメータの目盛り線を、縦目盛りの5mmの線上 → 縦目盛りの0mmの線上 → 横目盛りの0mmの線上 → 横目盛りの5mmの線上 の順にクリック
- マイクロメータの動画ファイルは解析ファイルと同様に460×640ピクセルとし、10mm×10mmのマイクロメータが収まるようにを撮影すること！！！
- 実行するとターミナルに以下のような出力
- X_scale: 15.1515151515 
- Y_scale: 15.5279503106 

### ゾウリムシの追跡 & 行動解析 (Patra2.py)

**実行方法その1

Patra2のディレクトリ内で以下を実行

```
$ poetry run python Patra2.py 解析したいディレクトリ/解析したい動画ファイル　X_scaleの値　Y_scaleの値
```

**実行方法その2

`run.sh`の中に，上記コマンドが書いてある。必要に応じて修正して以下を実行
```
$ ./run.sh
```

- `outputs/`の下に解析した動画ファイル名のフォルダができ、軌道・速さデータや、それらのフーリエ変換結果をファイル出力
- 位置データは生データとローパス処理後のものが有り
- ローパス（バターワース）処理： 直流成分は1/2, 約0-7.4Hzまではほぼ100%透過, 減衰域は7.4925Hz-9.99Hz
- 軌道データ一覧`Para<>_<filename>.csv`
```
time,para_id,X,Y,X_lpf,Y_lpf,para_id,Vx,Vy,Vx_lpf,Vy_lpf,V,V_lpf,V_lpf2
```
  - time: 時間
  - para_id: ゾウリムシのID
  - X,Y: 位置情報(μm)
  - X_lpf, Y_lpf: X,Yをローパス処理したもの
  - Vx, Vy: X,Yの差分で求めた速度(μm/s)
  - V: Vx,Vyから求めた速さ
  - Vx_lpf, Vy_lpf: X_lpf, Y_lpfから求めた速度
  - V_lpf: Vx_lpf, Vy_lpfから求めた速度
  - V_lpf2: V_lpfをローパス処理したもの

- 周波数分析データ一覧`Para<>_<filename>_freq.csv`
```
,X_freq,X_amp,Y_freq,Y_amp,V_lpf2_freq,V_lpf2_amp
```
  - X_freq,X_amp,Y_freq,Y_amp: X, Y軌道の周波数分析データ
  - V_lpf2_freq,V_lpf2_amp:  V_lpf2の周波数分析データ
- 周波数ピーク分析データ
  - `Para<>_<filename>_X_peak_freq.csv`
  - `Para<>_<filename>_Y_peak_freq.csv`
  - `Para<>_<filename>_V_peak_freq.csv`
