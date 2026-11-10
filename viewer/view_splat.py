"""TripoSplat 出力 (.ply / .splat) を見るためのローカルビューア.

data/output 以下の Gaussian Splatting ファイルを一覧表示し、ブラウザ上で
Spark.js (three.js) を使って回転・ズームしながら確認できる。標準ライブラリのみで動く。

Usage:
    uv run python viewer/view_splat.py                       # data/output を一覧表示
    uv run python viewer/view_splat.py --file doraemon/gaussians_262144.ply
    uv run python viewer/view_splat.py --dir path/to/outputs --port 9000 --no-browser
"""
import argparse
import json
import mimetypes
import shutil
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = Path(__file__).resolve().parent / "index.html"
DEFAULT_DIR = PROJECT_ROOT / "data" / "output"
SPLAT_SUFFIXES = (".ply", ".splat")
THUMB_NAME = "preprocessed.png"
META_NAME = "meta.json"


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


def make_handler(root: Path):
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
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Cache-Control", "no-store")  # 再生成したファイルを確実に読み直す
            self.end_headers()
            if head_only:
                return
            with open(path, "rb") as f:
                try:
                    shutil.copyfileobj(f, self.wfile)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def _route(self, head_only: bool):
            url_path = unquote(urlparse(self.path).path)
            if url_path in ("/", "/index.html"):
                return self._send_file(INDEX_HTML, head_only)
            if url_path == "/api/files":
                return self._send_json({"root": str(root), "models": list_models(root)})
            if url_path.startswith("/files/"):
                target = (root / url_path[len("/files/"):]).resolve()
                # 出力ディレクトリの外は配信しない（パストラバーサル防止）
                if target.is_relative_to(root) and target.is_file():
                    return self._send_file(target, head_only)
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_GET(self):
            self._route(head_only=False)

        def do_HEAD(self):
            self._route(head_only=True)

    return Handler


def start_server(root: Path, port: int) -> ThreadingHTTPServer:
    handler = make_handler(root)
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
    parser.add_argument("--no-browser", action="store_true", help="ブラウザを自動で開かない")
    args = parser.parse_args()

    root = args.dir.resolve()
    if not root.is_dir():
        parser.error(f"directory not found: {root}")

    server = start_server(root, args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    if args.file is not None:
        f = args.file if args.file.is_absolute() else root / args.file
        f = f.resolve()
        if not (f.is_relative_to(root) and f.is_file()):
            parser.error(f"--file must be an existing file under {root}: {f}")
        url += "?file=" + quote(f.relative_to(root).as_posix())

    n = len(list_models(root))
    print(f"[viewer] serving {n} model(s) from {root}")
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
