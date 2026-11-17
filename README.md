# CG_modeling

Gaussian Splatting を使った CG モデリングの検証プロジェクト。

## 3DGS (3D Gaussian Splatting)

[TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) で、1 枚の画像から 3D モデル (`.ply` / `.splat`) を生成する。
macOS (Apple Silicon / MPS) で実行し、生成したモデルはブラウザのビューアで確認する。

## 4DGS (4D Gaussian Splatting)

[splats4D](https://github.com/adamraudonis/splats4D) で、時系列の 3DGS フレーム (`.splat`) を
1 つのストリーミング再生できる `.splat4d` ファイルに圧縮・再生する。

splats4D は動画を直接は読まないので、動画の各フレームを TripoSplat で 3DGS にしてから渡す
(`scripts/video_to_splat4d.py`)。MPS で 1 フレーム約 70〜120 秒かかる (連続実行すると遅くなる)。

```bash
uv run python scripts/video_to_splat4d.py data/input/maki_yoko_27s.mp4 --fps 10 --stride 3
uv run python viewer/view_splat.py --scene4d maki_yoko_27s   # splats4D のビューアで再生
```

## フォルダ構成

```
CG_modeling/
├── notebooks/
│   └── test_triposplat.ipynb   # 画像 → 3DGS の生成・検証
├── scripts/
│   └── video_to_splat4d.py     # 動画 → フレームごとに 3DGS → .splat4d
├── viewer/
│   ├── view_splat.py           # ビューアのローカルサーバ (3DGS と 4D)
│   └── index.html              # 3DGS ビューア (Spark.js)。4D は splats4D のビューアを開く
├── external/                   # 外部リポジトリ (改変しない)
│   ├── TripoSplat/             # 3DGS 生成モデル
│   └── splats4D/               # 4DGS フォーマット・変換ツール・ビューア
├── checkpoints/                # モデルの重み
├── data/
│   ├── input/                  # 入力画像・動画
│   ├── output/                 # 生成した 3D モデル
│   └── output4d/               # 動画から作った 4D シーン
├── pyproject.toml, uv.lock     # Python 環境 (uv)
└── CLAUDE.md                   # Claude Code 向けの説明
```
