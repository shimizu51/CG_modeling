"""TripoSplat 出力 (.ply / .splat) を見るためのローカルビューア.

data/output 以下の Gaussian Splatting ファイルを一覧表示し、ブラウザ上で
Spark.js (three.js) を使って回転・ズームしながら確認できる。標準ライブラリのみで動く。

data/output4d 以下の 4D シーン (scripts/video_to_splat4d.py の出力 scene.splat4d) は、
external/splats4D/docs/demo のビルド済みビューアを /4d/ で配信して再生する。

Usage:
    uv run python viewer/view_splat.py                       # data/output を一覧表示
    uv run python viewer/view_splat.py --file doraemon/gaussians_262144.ply
    uv run python viewer/view_splat.py --scene4d jellyous_10s  # 4D シーンを再生
    uv run python viewer/view_splat.py --dir path/to/outputs --port 9000 --no-browser
"""
import argparse
import json
import mimetypes
import re
import shutil
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlencode, urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = Path(__file__).resolve().parent / "index.html"
DEFAULT_DIR = PROJECT_ROOT / "data" / "output"
DEFAULT_DIR_4D = PROJECT_ROOT / "data" / "output4d"
SPLAT4D_VIEWER_DIR = PROJECT_ROOT / "external" / "splats4D" / "docs" / "demo"  # ビルド済み (npm 不要)
SPLAT_SUFFIXES = (".ply", ".splat")
THUMB_NAME = "preprocessed.png"
META_NAME = "meta.json"
SCENE4D_NAME = "scene.splat4d"
THUMB4D_NAME = "work/frame_0000.png"


def list_models(root: Path) -> list[dict]:
    """root 以下の .ply / .splat を列挙し、同じフォルダのサムネイル・メタ情報を添える."""
    models = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in SPLAT_SUFFIXES or not p.is_file():
            continue
        rel = p.relative_to(root)
        thumb = p.parent / THUMB_NAME
        meta = None
        meta_path = p.parent / META_NAME
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                meta = None
        stat = p.stat()
        models.append({
            "path": rel.as_posix(),
            "name": p.name,
            "group": rel.parent.as_posix() if rel.parent != Path(".") else "",
            "format": p.suffix.lower().lstrip("."),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "thumb": thumb.relative_to(root).as_posix() if thumb.is_file() else None,
            "meta": meta,
        })
    return models


def scene4d_url(root4d: Path, scene: Path, meta: dict | None) -> str:
    """splats4D ビューアで scene を開く URL。meta.json のカメラ (OpenCV 系) を初期視点にする."""
    params = {"file": "/files4d/" + quote(scene.relative_to(root4d).as_posix())}
    cam = (meta or {}).get("camera")
    if cam:
        params["cam"] = ",".join(f"{v:.4f}" for v in [*cam["position"], *cam["target"], cam.get("fov", 60)])
    return "/4d/?" + urlencode(params, safe="/,")


def list_scenes4d(root4d: Path) -> list[dict]:
    """root4d/<name>/scene.splat4d を列挙する."""
    scenes = []
    if not root4d.is_dir():
        return scenes
    for p in sorted(root4d.glob(f"*/{SCENE4D_NAME}")):
        meta = None
        try:
            meta = json.loads((p.parent / META_NAME).read_text())
        except (OSError, json.JSONDecodeError):
            pass
        thumb = p.parent / THUMB4D_NAME
        stat = p.stat()
        scenes.append({
            "name": p.parent.name,
            "url": scene4d_url(root4d, p, meta),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "thumb": "/files4d/" + quote(thumb.relative_to(root4d).as_posix()) if thumb.is_file() else None,
            "frames": (meta or {}).get("num_frames"),
            "fps": (meta or {}).get("fps"),
        })
    return scenes


def make_handler(root: Path, root4d: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # アクセスログは静かにしておく
            pass

        def _send_json(self, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, head_only: bool = False):
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            size = path.stat().st_size
            # splats4D ビューアは HTTP Range で必要な部分だけ取りに来る (シーク・ストリーミング)
            start, end = 0, size - 1
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "").strip())
            if m and (m[1] or m[2]):
                if m[1]:
                    start, end = int(m[1]), min(int(m[2]) if m[2] else size - 1, size - 1)
                else:  # bytes=-N (末尾 N バイト)
                    start = max(0, size - int(m[2]))
                if start > end:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")  # 再生成したファイルを確実に読み直す
            self.end_headers()
            if head_only:
                return
            with open(path, "rb") as f:
                f.seek(start)
                remaining = end - start + 1
                try:
                    while remaining > 0:
                        chunk = f.read(min(1 << 20, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def _send_under(self, base: Path, rel: str, head_only: bool) -> bool:
            target = (base / rel).resolve()
            # base の外は配信しない（パストラバーサル防止）
            if target.is_relative_to(base) and target.is_file():
                self._send_file(target, head_only)
                return True
            return False

        def _route(self, head_only: bool):
            url_path = unquote(urlparse(self.path).path)
            if url_path in ("/", "/index.html"):
                return self._send_file(INDEX_HTML, head_only)
            if url_path == "/api/files":
                return self._send_json({"root": str(root), "models": list_models(root),
                                        "scenes4d": list_scenes4d(root4d)})
            if url_path.startswith("/files/") and self._send_under(root, url_path[len("/files/"):], head_only):
                return
            if url_path.startswith("/files4d/") and self._send_under(root4d, url_path[len("/files4d/"):], head_only):
                return
            if url_path == "/4d":
                self.send_response(HTTPStatus.MOVED_PERMANENTLY)
                self.send_header("Location", "/4d/" + (f"?{urlparse(self.path).query}" if urlparse(self.path).query else ""))
                self.end_headers()
                return
            if url_path.startswith("/4d/"):
                rel = url_path[len("/4d/"):] or "index.html"
                if self._send_under(SPLAT4D_VIEWER_DIR, rel, head_only):
                    return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_GET(self):
            self._route(head_only=False)

        def do_HEAD(self):
            self._route(head_only=True)

    return Handler


def start_server(root: Path, root4d: Path, port: int) -> ThreadingHTTPServer:
    handler = make_handler(root, root4d)
    try:
        return ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError:
        print(f"[viewer] port {port} is in use; picking a free port")
        return ThreadingHTTPServer(("127.0.0.1", 0), handler)


def main():
    parser = argparse.ArgumentParser(description="TripoSplat 出力 (.ply/.splat) ビューア")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="モデルを探すディレクトリ (default: data/output)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--file", type=Path, default=None, help="最初に開くモデル (--dir からの相対パス or 絶対パス)")
    parser.add_argument("--dir4d", type=Path, default=DEFAULT_DIR_4D,
                        help="4D シーン (<name>/scene.splat4d) を探すディレクトリ (default: data/output4d)")
    parser.add_argument("--scene4d", default=None, help="最初に開く 4D シーンの名前 (--dir4d 直下のフォルダ名)")
    parser.add_argument("--no-browser", action="store_true", help="ブラウザを自動で開かない")
    args = parser.parse_args()

    root = args.dir.resolve()
    if not root.is_dir():
        parser.error(f"directory not found: {root}")
    root4d = args.dir4d.resolve()

    server = start_server(root, root4d, args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    if args.scene4d is not None:
        scene = next((s for s in list_scenes4d(root4d) if s["name"] == args.scene4d), None)
        if scene is None:
            parser.error(f"--scene4d: {root4d / args.scene4d / SCENE4D_NAME} がありません")
        url = url.rstrip("/") + scene["url"]
    elif args.file is not None:
        f = args.file if args.file.is_absolute() else root / args.file
        f = f.resolve()
        if not (f.is_relative_to(root) and f.is_file()):
            parser.error(f"--file must be an existing file under {root}: {f}")
        url += "?file=" + quote(f.relative_to(root).as_posix())

    n = len(list_models(root))
    print(f"[viewer] serving {n} model(s) from {root}, {len(list_scenes4d(root4d))} 4D scene(s) from {root4d}")
    print(f"[viewer] open {url}   (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
