# Third-party software and fonts

SIGN NAVは次の第三者ソフトウェアを使用しています。Pythonパッケージ本体とFFmpeg実行ファイルは、このソース一式には同梱していません。各プロジェクトの著作権表記・ライセンスは、それぞれの配布物と公式情報を参照してください。

| プロジェクト | 使用箇所 | 公式情報・ソース |
|---|---|---|
| NumPy | 数値処理・学習済み重みの推論 | https://numpy.org/ / https://github.com/numpy/numpy |
| Pillow | 合成標識・図・動画フレームの生成 | https://python-pillow.org/ / https://github.com/python-pillow/Pillow |
| scikit-learn | MLPの学習・画像認識評価 | https://scikit-learn.org/stable/ / https://github.com/scikit-learn/scikit-learn |
| threadpoolctl | 学習・評価時の数値計算スレッド数制御 | https://github.com/joblib/threadpoolctl |
| ReportLab | 説明資料PDFの生成 | https://docs.reportlab.com/ |
| FFmpeg | RGBフレームからMP4への変換 | https://ffmpeg.org/ |

`requirements*.txt` に記載したのは直接使用するパッケージです。インストール時には、各パッケージに必要な依存ライブラリも解決されます。

## 同梱フォント

`assets/fonts/NotoSansJP-Regular.ttf` と `assets/fonts/NotoSansJP-Bold.ttf` は、Google FontsのNoto Sans JP可変フォントから、ウェイト400・700をそれぞれ静的化したものです。

- 出典： https://github.com/google/fonts/tree/main/ofl/notosansjp
- 配布上のライセンス：SIL Open Font License 1.1
- 著作権表記とライセンス全文：`assets/fonts/OFL.txt`

同梱のOFL文書はフォントの出典に付属するものを保持しています。この文書によって、SIGN NAVの独自コード全体にフォントのライセンスを適用するものではありません。

## 生成データ

標識・コース・ロボット図形は、本プロジェクトのプログラムで生成しています。学習済み重みは、その合成標識画像を用いて作成しました。

