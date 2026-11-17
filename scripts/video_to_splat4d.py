"""動画 → フレームごとに TripoSplat で 3DGS を生成 → splats4D の .splat4d にまとめる.

TripoSplat は 1 枚の画像から 1 つの 3DGS を作るだけなので、動画は「フレームごとに独立に生成した
3DGS の連番」として扱う。splats4D (external/splats4D) はその連番 .splat を 1 ファイルに圧縮する。

フレーム間をできるだけ揃えるために次のことをする:
  - 全フレームで同じ seed (= 同じ初期ノイズ) を使う
  - .splat を書くときに上流の to_splat_bytes の並べ替え (不透明度×体積順) をしない。
    デコーダの出力順 (octree 順) のまま書く。splats4D は「i 番目 = 同じ splat」を前提にしている
  - TripoSplat の出力は毎フレーム物体が単位立方体に収まるよう正規化されているので、
    元動画での物体の bbox (背景除去マスクから取る) を使って、大きさと画面上の位置を
    動画全体で揃える (固定カメラ・平行投影の近似。奥行き方向は揃えられない)

座標は splats4D ビューア (antimatter15 互換) に合わせて OpenCV 系 (x 右, y 下, z 奥) で、単位は m。

Usage:
    uv run python scripts/video_to_splat4d.py data/input/maki_yoko_27s.mp4 --fps 10 --stride 3  # 10/3 fps で生成
    uv run python scripts/video_to_splat4d.py data/input/jellyous_10s.mp4 --fps 10 --frames 0:3   # 試し実行
    uv run python scripts/video_to_splat4d.py data/input/jellyous_10s.mp4 --fps 10 --export-only  # 再デコードと圧縮だけ

出力: data/output4d/<動画の stem>/
    input/frame_0000.png        動画から抽出したフレーム
    work/frame_0000.pt          Flow サンプリング後の潜在と bbox (生成をやり直さずに再デコードできる)
    work/frame_0000.png         前処理後の画像 (TripoSplat への実際の入力)
    frames/frame_0000.splat     splat4d encode の入力 (+ frames.json)
    scene.splat4d               圧縮結果
    encode_report.json          splat4d encode --report の出力
    meta.json                   パラメータ・時間・ビューア用のカメラ
"""
import os

# MPS に未実装の op を CPU で実行させる。torch の import 前に設定する必要がある。
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "triposplat"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "output4d"

# external/TripoSplat は改変せず、import パスに追加して使う
sys.path.insert(0, str(PROJECT_ROOT / "external" / "TripoSplat"))
from triposplat import TripoSplatPipeline  # noqa: E402

CKPT_FILES = {
    "ckpt_path":              "diffusion_models/triposplat_fp16.safetensors",
    "decoder_path":           "vae/triposplat_vae_decoder_fp16.safetensors",
    "dinov3_path":            "clip_vision/dino_v3_vit_h.safetensors",
    "flux2_vae_encoder_path": "vae/flux2-vae.safetensors",
    "rmbg_path":              "background_removal/birefnet.safetensors",
}
CANVAS_SIZE = 1024  # triposplat._CANVAS_SIZE と同じ。前処理で短辺をこの大きさにする
SH_C0 = 0.28209479177387814

# TripoSplat 内部座標 → OpenCV 系 (x 右, y 下, z 奥)。
# 内部 (x, y, z) → three.js 系 (y, z, x) [CLAUDE.md 参照] → OpenCV 系は (x, -y, -z) なので (y, -z, -x)。
CV_FROM_TRIPO = [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]


def device_name() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def sync(device: str):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def extract_frames(video: Path, out_dir: Path, fps: float) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(out_dir.glob("frame_*.png"))
    if frames:
        return frames
    # 前処理で短辺 1024 にするので、4K などはここで縮めておく (短辺 1080)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
         "-vf", f"fps={fps},scale='if(gt(iw,ih),-2,min(1080,iw))':'if(gt(iw,ih),min(1080,ih),-2)'",
         "-start_number", "0", str(out_dir / "frame_%04d.png")],
        check=True,
    )
    return sorted(out_dir.glob("frame_*.png"))


def matte(pipe: TripoSplatPipeline, image: Image.Image, erode_radius: int):
    """背景除去してから前処理に渡す RGBA と、元画像座標での物体 bbox を返す.

    TripoSplat の preprocess_image と同じ縮尺・収縮をここで済ませておき、alpha 付きで渡す
    (alpha があると preprocess_image は背景除去と収縮を飛ばしてクロップだけする)。
    """
    w, h = image.size
    s = CANVAS_SIZE / min(w, h)
    resized = image.convert("RGB").resize((round(w * s), round(h * s)), Image.LANCZOS)
    rgba = pipe.rmbg.remove_background(resized)
    if erode_radius > 0:
        rgba.putalpha(rgba.getchannel(3).filter(ImageFilter.MinFilter(2 * erode_radius + 1)))
    ys, xs = np.nonzero(np.array(rgba.getchannel(3)))
    if len(xs) < 100:  # 暗転などで物体が見つからない
        return rgba, None
    bbox = [xs.min() / s, ys.min() / s, xs.max() / s, ys.max() / s]  # 元画像のピクセル座標
    return rgba, [float(v) for v in bbox]


def generate(pipe, frames: list[Path], work: Path, indices: list[int], args, device: str):
    work.mkdir(parents=True, exist_ok=True)
    for n, i in enumerate(indices):
        out_pt = work / f"frame_{i:04d}.pt"
        if out_pt.is_file():
            continue
        sync(device)
        t0 = time.perf_counter()
        rgba, bbox = matte(pipe, Image.open(frames[i]), args.erode_radius)
        if bbox is None:
            torch.save({"latent": None, "bbox": None}, out_pt)
            print(f"[{i:04d}] no object found; will reuse a neighbor frame", flush=True)
            continue
        prepared = pipe.preprocess_image(rgba, erode_radius=0)
        prepared.save(work / f"frame_{i:04d}.png")
        t1 = time.perf_counter()
        # 全フレームで同じ seed → 同じ初期ノイズ (フレーム間の一貫性のため)
        gen = torch.Generator(device=device).manual_seed(args.seed)
        cond = pipe.encode_image(prepared, generator=gen)
        out = pipe.sample_latent(cond, steps=args.steps, guidance_scale=args.guidance_scale,
                                 shift=args.shift, generator=gen)
        sync(device)
        t2 = time.perf_counter()
        torch.save({"latent": out["latent"].cpu(), "bbox": bbox,
                    "timings_sec": {"matte_preprocess": t1 - t0, "encode_sample": t2 - t1}}, out_pt)
        print(f"[{i:04d}] ({n + 1}/{len(indices)}) bbox={np.round(bbox).astype(int).tolist()} "
              f"matte {t1 - t0:.1f}s  encode+sample {t2 - t1:.1f}s", flush=True)


def decode_records(pipe, latent: torch.Tensor, num_gaussians: int, seed: int, device: str) -> dict:
    """潜在 → Gaussian → OpenCV 系の配列 (並べ替えなし)."""
    torch.manual_seed(seed)  # octree の点サンプリングが torch.rand を使うので、再デコードでも同じ結果にする
    g = pipe.decode_latent(latent.to(device), num_gaussians=num_gaussians)
    xyz, rot = g._transformed_xyz_rot(transform=CV_FROM_TRIPO)
    f_dc = g._features_dc.detach().float().cpu().numpy()
    return {
        "xyz": xyz.astype(np.float32),
        "scale": g.get_scaling.detach().float().cpu().numpy().astype(np.float32),
        "rgb": np.clip((f_dc[:, 0, :] * SH_C0 + 0.5) * 255, 0, 255).astype(np.uint8),
        "alpha": np.clip(g.get_opacity.detach().float().cpu().numpy()[:, 0] * 255, 0, 255).astype(np.uint8),
        "rot": rot / np.linalg.norm(rot, axis=-1, keepdims=True),
    }


def splat_bytes(r: dict) -> bytes:
    """antimatter15 .splat (32 B/splat)。上流 Gaussian.to_splat_bytes から並べ替えだけを除いたもの."""
    rot_u8 = np.clip(r["rot"] * 128 + 128, 0, 255).astype(np.uint8)
    rgba = np.concatenate([r["rgb"], r["alpha"][:, None]], axis=1)
    return np.concatenate([
        r["xyz"].astype(np.float32).view(np.uint8).reshape(-1, 12),
        r["scale"].astype(np.float32).view(np.uint8).reshape(-1, 12),
        rgba.reshape(-1, 4),
        rot_u8.reshape(-1, 4),
    ], axis=1).tobytes()


def visible_extent(r: dict) -> tuple[np.ndarray, np.ndarray]:
    """見えている Gaussian の範囲 (外れ値を除いた 0.5〜99.5 パーセンタイル)."""
    p = r["xyz"][r["alpha"] >= 77]  # 不透明度 0.3 以上
    if len(p) < 100:
        p = r["xyz"]
    return np.percentile(p, 0.5, axis=0), np.percentile(p, 99.5, axis=0)


def export(pipe, frames_dir: Path, work: Path, indices: list[int], fps: float, args, device: str) -> dict:
    data = [torch.load(work / f"frame_{i:04d}.pt") for i in indices]
    n_frames = len(indices)
    ok = [i for i, d in enumerate(data) if d["latent"] is not None]
    if not ok:
        raise SystemExit("物体が見つかったフレームがありません")
    # 物体が見つからなかったフレームは一番近いフレームで代用 (splat4d は全フレーム同じ数が必要)
    src = [min(ok, key=lambda j: abs(j - i)) for i in range(n_frames)]

    bboxes = np.array([data[j]["bbox"] for j in src])  # (T, 4) x0, y0, x1, y1 [px]
    m_per_px = args.subject_height / np.median(bboxes[:, 3] - bboxes[:, 1])
    center_ref = np.median((bboxes[:, :2] + bboxes[:, 2:]) / 2, axis=0)

    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    manifest = {"fps": fps, "frames": []}
    lo_all, hi_all = np.full(3, np.inf), np.full(3, -np.inf)
    for i in range(n_frames):
        r = decode_records(pipe, data[src[i]]["latent"], args.num_gaussians, args.seed, device)
        if not args.no_align:
            # 画面内の大きさ (bbox の長辺) と位置 (bbox の中心) を元動画に合わせる
            lo, hi = visible_extent(r)
            box = bboxes[i]
            size_px = max(box[2] - box[0], box[3] - box[1])
            s = size_px * m_per_px / max(hi[0] - lo[0], hi[1] - lo[1])
            offset = np.array([*(((box[:2] + box[2:]) / 2 - center_ref) * m_per_px), 0.0])
            r["xyz"] = ((r["xyz"] - (lo + hi) / 2) * s + offset).astype(np.float32)
            r["scale"] = (r["scale"] * s).astype(np.float32)
        lo, hi = visible_extent(r)
        lo_all, hi_all = np.minimum(lo_all, lo), np.maximum(hi_all, hi)
        name = f"frame_{i:04d}.splat"
        (frames_dir / name).write_bytes(splat_bytes(r))
        manifest["frames"].append({"file": name, "timestamp": round(i / fps, 6)})
    (frames_dir / "frames.json").write_text(json.dumps(manifest, indent=1))

    # ビューアの初期カメラ: 正面 (入力動画のカメラ側 = -z) から全体が収まる距離
    fov = 50.0
    center = (lo_all + hi_all) / 2
    extent = float(max(hi_all[0] - lo_all[0], hi_all[1] - lo_all[1]))
    dist = extent / 2 / np.tan(np.radians(fov / 2)) * 1.25 + (hi_all[2] - lo_all[2]) / 2
    camera = {"position": [float(center[0]), float(center[1]), float(center[2] - dist)],
              "target": [float(v) for v in center], "fov": fov}
    return {"reused_frames": [indices[i] for i in range(n_frames) if src[i] != i], "m_per_px": float(m_per_px),
            "aabb": [*map(float, lo_all), *map(float, hi_all)], "camera": camera}


def encode_splat4d(frames_dir: Path, out_dir: Path, args) -> dict:
    out = out_dir / "scene.splat4d"
    report = out_dir / "encode_report.json"
    cmd = ["splat4d", "encode", "-i", str(frames_dir), "-o", str(out), "--report", str(report),
           "--pos-mm", str(args.pos_mm), "--color-levels", str(args.color_levels),
           "--scale-pct", str(args.scale_pct)]
    print("$", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    return {"encode_sec": time.perf_counter() - t0, "report": json.loads(report.read_text())}


def parse_frames(spec: str | None, n: int) -> list[int]:
    if spec is None:
        return list(range(n))
    a, _, b = spec.partition(":")
    return list(range(int(a or 0), min(n, int(b) if b else n)))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("video", type=Path)
    p.add_argument("--fps", type=float, default=10.0, help="動画から取り出すフレームレート")
    p.add_argument("--stride", type=int, default=1,
                   help="取り出したフレームを N 枚おきに使う (生成するフレームレートは fps / N)")
    p.add_argument("--out", type=Path, default=None, help="出力先 (default: data/output4d/<stem>)")
    p.add_argument("--frames", default=None, help="生成するフレーム範囲 'a:b' (試し実行用。export はしない)")
    p.add_argument("--export-only", action="store_true", help="生成済みの潜在から再デコードと圧縮だけ行う")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--guidance-scale", type=float, default=3.0)
    p.add_argument("--shift", type=float, default=3.0)
    p.add_argument("--erode-radius", type=int, default=1)
    p.add_argument("--num-gaussians", type=int, default=131072)
    p.add_argument("--subject-height", type=float, default=1.7,
                   help="bbox の高さ (中央値) を何 m とみなすか。単位を m にそろえるためだけに使う")
    p.add_argument("--no-align", action="store_true", help="TripoSplat の正規化座標のまま書き出す")
    p.add_argument("--pos-mm", type=float, default=2.0)
    p.add_argument("--color-levels", type=int, default=4)
    p.add_argument("--scale-pct", type=float, default=2.0)
    args = p.parse_args()

    out_dir = (args.out or OUTPUT_ROOT / args.video.stem).resolve()
    device = device_name()
    frames = extract_frames(args.video, out_dir / "input", args.fps)
    # 生成結果 (work/frame_XXXX.pt) は抽出時の番号で保存するので、stride を変えても使い回せる
    selected = list(range(0, len(frames), args.stride))
    out_fps = args.fps / args.stride
    print(f"{args.video}: {len(frames)} frames @ {args.fps} fps, using every {args.stride} "
          f"→ {len(selected)} frames @ {out_fps:.3g} fps → {out_dir}", flush=True)

    t0 = time.perf_counter()
    pipe = TripoSplatPipeline(**{k: str(CKPT_DIR / f) for k, f in CKPT_FILES.items()}, device=device)
    print(f"pipeline loaded in {time.perf_counter() - t0:.1f}s on {device}", flush=True)

    t_gen = time.perf_counter()
    if not args.export_only:
        todo = [i for i in parse_frames(args.frames, len(frames)) if i % args.stride == 0]
        generate(pipe, frames, out_dir / "work", todo, args, device)
        if args.frames is not None:
            return
    t_gen = time.perf_counter() - t_gen

    t0 = time.perf_counter()
    info = export(pipe, out_dir / "frames", out_dir / "work", selected, out_fps, args, device)
    t_export = time.perf_counter() - t0
    enc = encode_splat4d(out_dir / "frames", out_dir, args)

    timings = [t for i in selected
               if (t := torch.load(out_dir / "work" / f"frame_{i:04d}.pt").get("timings_sec"))]
    meta = {
        "input": args.video.name,
        "device": device,
        "torch": torch.__version__,
        "num_frames": len(selected),
        "fps": out_fps,
        "extract_fps": args.fps,
        "stride": args.stride,
        "num_gaussians": args.num_gaussians,
        "params": {"seed": args.seed, "steps": args.steps, "guidance_scale": args.guidance_scale,
                   "shift": args.shift, "erode_radius": args.erode_radius, "align": not args.no_align},
        "timings_sec": {
            "generate_per_frame_mean": float(np.mean([sum(t.values()) for t in timings])) if timings else None,
            "generate_this_run": t_gen,
            "export": t_export,
            "splat4d_encode": enc["encode_sec"],
        },
        **info,
        "splat4d": enc["report"],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps({k: meta[k] for k in ("num_frames", "reused_frames", "camera", "timings_sec")}, indent=2))


if __name__ == "__main__":
    main()
