# CG_modeling

Gaussian Splatting を使った CG モデリングの検証プロジェクト。

## 3DGS (3D Gaussian Splatting)

[TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) で、1 枚の画像から 3D モデル (`.ply` / `.splat`) を生成する。
macOS (Apple Silicon / MPS) で実行し、生成したモデルはブラウザのビューアで確認する。

## 4DGS (4D Gaussian Splatting)

[splats4D](https://github.com/adamraudonis/splats4D) で、時系列の 3DGS フレーム (`.splat`) を
1 つのストリーミング再生できる `.splat4d` ファイルに圧縮・再生する。

## フォルダ構成

```
CG_modeling/
├── notebooks/
│   └── test_triposplat.ipynb   # 画像 → 3DGS の生成・検証
├── viewer/
│   ├── view_splat.py           # 3DGS ビューアのローカルサーバ
│   └── index.html              # 3DGS ビューア (Spark.js)
├── external/                   # 外部リポジトリ (改変しない)
│   ├── TripoSplat/             # 3DGS 生成モデル
│   └── splats4D/               # 4DGS フォーマット・変換ツール・ビューア
├── checkpoints/                # モデルの重み
├── data/
│   ├── input/                  # 入力画像
│   └── output/                 # 生成した 3D モデル
├── pyproject.toml, uv.lock     # Python 環境 (uv)
└── CLAUDE.md                   # Claude Code 向けの説明
```
