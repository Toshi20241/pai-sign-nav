# SIGN NAV

矢印の標識を画像AIで読み、左・直進・右の進路を選ぶ、小規模な学習用シミュレーションです。

## 仕組み

標識パネルの合成画像（96×96画素）を24×24画素へ縮小し、小さなニューラルネットワーク（MLP）で3種類に分類します。モデルは576 → 32 → 3の構成です。学習用900枚、学習途中の確認用150枚を用意し、60回の学習を行いました。

分類結果に応じて、あらかじめ決めた経由点の経路を選びます。左右輪の速度制御で経由点をたどり、3つの標識を経てゴールへ進みます。学習するのは画像分類の部分です。

移動は車輪の一次遅れと角加速度上限を含む平面モデルです。撮影地点・標識の画像範囲・自己位置は既知としています。実機や実カメラ映像からの標識検出は扱っていません。

## 小規模な評価結果

学習と別に生成した画像で評価しました。

| 画像の条件 | 正解数 | 正解率 |
|---|---:|---:|
| 標準条件 | 300 / 300 | 100.00% |
| 大きさ・位置・回転・見え方などを強く変えた条件 | 139 / 300 | 46.33% |

走行は同じ6条件で、AIの判定を使う方式と、常に直進する方式を比べました。

| 方式 | 3標識すべてに従った走行 |
|---|---:|
| AIの判定を使う | 6 / 6 |
| 常に直進する | 0 / 6 |

12走行ともゴールに到着し、接触は0回でした。分岐が後で合流するコースなので、ゴール到着と標識遵守を分けて評価しています。コース形状は固定で、強い変動条件は画像分類だけの評価です。

記録は `results/vision_metrics.json`、`results/benchmark/`、`results/demo/` にあります。

## 実行

Python 3.12.14でローカル動作を確認しました。保存済みモデルを使う実行にはNumPyとPillowが必要です。GPUや外部AI APIは使いません。

```bash
python -m pip install -r requirements-core.txt
python main.py
python experiments.py
python verify_navigation.py
```

`main.py` はデモ、`experiments.py` は6条件×2方式の比較、`verify_navigation.py` は保存した画像・判断・軌跡などの照合を行います。

学習から動画・PDFまで再生成する場合は、次の順に実行します。動画生成には別途FFmpegの `ffmpeg` コマンドが必要です。

```bash
python -m pip install -r requirements.txt
python train.py
python main.py
python experiments.py
python verify_navigation.py
python make_video.py
python build_report.py
```

同梱版の動画は82.10秒、1920×1080、20 fps、走行区間1倍速です。MP4と説明資料PDFは `deliverables/` に生成されます。

`run_colab.ipynb` はコードZIPをアップロードするための実行手順です。Colab上での実行確認は行っていません。使用ソフトウェアとフォントの出典は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。

モデルの参照資料：[scikit-learn MLPClassifier公式資料](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)。

制作支援：設計・コード・資料の作成に生成AIを使用し、数値は同梱プログラムの実行結果を使用しています。

