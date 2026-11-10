# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

単一画像 → 3D Gaussian Splatting を生成する TripoSplat (`external/TripoSplat`) を、macOS (Apple M5 / 32GB, MPS) 上で検証するプロジェクト。
成果物は生成ノートブック `notebooks/test_triposplat.ipynb` と、ブラウザビューア `viewer/`。ユーザーとのやり取りは日本語。

## コマンド

```bash
uv sync                                   # .venv を作成 (Python 3.12 固定, package = false)
uv add <pkg> / uv add --dev <pkg>         # 依存追加。pip や グローバル環境・kernelspec 登録は使わない
uv run jupyter lab                        # ノートブックを対話実行

# ノートブックをヘッドレスで全セル実行 (結果をノートブックに保存)。全 4 画像で約 15 分かかる
cd notebooks && TRIPOSPLAT_LAUNCH_VIEWER=0 uv run jupyter nbconvert --to notebook --execute --inplace \
  test_triposplat.ipynb --ExecutePreprocessor.timeout=-1

uv run python viewer/view_splat.py [--file <data/output からの相対パス>] [--port 8765] [--no-browser]
```

テスト・lint の仕組みは無い。動作確認はノートブック実行と、ビューアをブラウザで開いての目視確認。

## 環境・リポジトリの決まりごと

- Python 環境は uv でこのプロジェクト内に閉じる (ユーザーの明示的な要望)。
- `external/*` は上流リポジトリのクローン (それぞれ独自の `.git` を持ち、外側の git では未追跡)。**改変しない。** 使うときは `sys.path` に追加して import する。
  - `TripoSplat`: VAST-AI-Research/TripoSplat (検証時のコミット `d8db9e0`)。
  - `splats4D`: adamraudonis/splats4D (4D splat フォーマット)。クローン済みだが、このプロジェクトのコードからはまだ使っていない。
  - 新しくクローンした環境では `git clone https://github.com/VAST-AI-Research/TripoSplat.git external/TripoSplat` (splats4D も同様) が必要。
- `.gitignore` 済み: `.venv/`, `checkpoints/` (重み約 3.5GB), `data/` (入力画像・生成物)。
- 重みはノートブックのセクション 3 が HuggingFace `VAST-AI/TripoSplat` から必要な 5 ファイルだけを `checkpoints/triposplat/` に取得する (既にあればスキップ)。

## アーキテクチャ

### 生成パイプライン (`external/TripoSplat/triposplat.py` の `TripoSplatPipeline`)

`run()` は 4 段階で、ノートブックのセクション 6 は `run_gradio.py` と同じようにこれを分けて呼んでいる:

1. `preprocess_image`: BiRefNet で背景除去 → 物体中心に 1024² でクロップし、黒背景に合成
2. `encode_image`: DINOv3 ViT-H (`feature1`) と FLUX.2 VAE (`feature2`) で画像特徴を作る
3. `sample_latent`: Flow モデルで Euler サンプリング。`guidance_scale > 1` だと CFG で 1 ステップあたり 2 回推論する
4. `decode_latent`: 潜在 → `Gaussian` (数は 32,768〜262,144 で 32 の倍数)

- 時間の 95% 以上が段階 3 (MPS で 1 ステップ約 7.5 秒、既定 20 ステップで約 150 秒)。Flow モデルの潜在は 8192 トークン固定なので、`num_gaussians` を変えても生成時間はほぼ変わらない。
- `num_gaussians` にリストを渡すと、サンプリングは 1 回で、デコードだけを数ごとに繰り返す。
- 入力は 1 枚専用。`_image_to_pil` は複数枚を拒否し、Flow モデル (`model.py` の `LatentSeqMMFlowModel.forward`) は「潜在・画像 1 枚分のトークン・カメラ 1 個」を結合して処理する。複数枚の写真から 1 つのモデルを作る用途には使えない。

### macOS / MPS 対応 (すべて上流を改変せずに呼び出し側で対応)

- 上流の既定は `device="cuda"`。`TripoSplatPipeline(..., device="mps")` で渡す。`model.py` に CUDA 専用 API は無い。
- `PYTORCH_ENABLE_MPS_FALLBACK=1` は **torch の import 前**に設定する (ノートブック冒頭)。BiRefNet の `torchvision.ops.deform_conv2d` や RoPE の `torch.polar` / complex64 への保険。torch 2.15 ではフォールバック警告は出なかった。
- 上流は `Image.open` するだけで EXIF の回転を適用しない。iPhone 写真は `ImageOps.exif_transpose` してから PIL 画像として渡す。
- 時間計測の前後には `torch.mps.synchronize()` を呼ぶ (ノートブックの `timer()`)。

### 座標系 (ノートブックとビューアで揃えている)

- 保存時に `Gaussian._DEFAULT_TRANSFORM` が内部座標を変換する。ビューアは上流の `static/viewer/viewer.html` と同じ補正 (`mesh.rotation.y = π/2` を `rotation.x = π` の Group に入れる) で、three.js の +Y up・正面 +Z に合わせる。
- 合成すると、内部座標 `(x, y, z)` → ビューア座標 `(y, z, x)`。ノートブックの `gaussian_to_numpy` はこの対応で 4 視点プレビューを描く。**どちらかを変えるときは両方直す。**
- モデルの「正面」は入力写真の撮影方向 (斜めから撮った写真なら、正面ビューは物体の側面になる)。

### 出力形式 (ノートブックとビューア間の取り決め)

ノートブックの `save_outputs` が `data/output/<画像の stem>/` に `gaussians_<N>.ply`、`gaussians_<N>.splat`、`preprocessed.png`、`meta.json` を書く。
`viewer/view_splat.py` の `/api/files` は `.ply` / `.splat` を再帰的に探し、同じフォルダの `preprocessed.png` をサムネイルに、`meta.json` の `params.{seed,steps,guidance_scale,shift}`・`device`・`timings_sec.total`・`input` を情報欄に使う。`meta.json` の構造を変えたら `viewer/index.html` の `metaRows` も合わせる。

### ビューア

- `view_splat.py`: 標準ライブラリだけの `ThreadingHTTPServer` を `127.0.0.1` で起動する。ルートは `/` (index.html)、`/api/files`、`/files/<相対パス>` (出力ディレクトリの外は 404)。ポートが使用中なら空きポートを使う。
- `index.html`: three@0.180.0 と @sparkjsdev/spark@2.0.0 を CDN からバージョン固定で読み込む (上流 viewer と同じ)。`SplatMesh({url})` で読み、ドラッグ&ドロップは `SplatMesh({fileBytes, fileName})`。読み込み直後に 1 秒ほど何も映らないことがある (Spark の非同期ソート待ち)。
