"""
AutoMix Web アプリケーション
"""
import threading
import webbrowser
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from ..cli import process_audio
from ..plugins.base import NoiseGatePlugin, PitchShiftPlugin, PluginManager
from ..plugins.custom_effects import (
    HarmonicExciterPlugin,
    StereoEnhancerPlugin,
    VintageWarmthPlugin,
    VocalEnhancerPlugin,
)

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# アップロードフォルダの設定
UPLOAD_FOLDER = Path("uploads")
OUTPUT_FOLDER = Path("outputs")
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["OUTPUT_FOLDER"] = str(OUTPUT_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB

# 許可される拡張子
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "flac", "ogg"}

# プラグインマネージャー
plugin_manager = PluginManager()

# 処理状態を管理
processing_status = {"is_processing": False, "progress": 0, "message": "Ready", "error": None}


def allowed_file(filename: str) -> bool:
    """ファイルが許可された拡張子かチェック"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def init_plugins() -> None:
    """プラグインを初期化"""
    plugin_manager.register_plugin(PitchShiftPlugin())
    plugin_manager.register_plugin(NoiseGatePlugin())
    plugin_manager.register_plugin(VintageWarmthPlugin())
    plugin_manager.register_plugin(VocalEnhancerPlugin())
    plugin_manager.register_plugin(StereoEnhancerPlugin())
    plugin_manager.register_plugin(HarmonicExciterPlugin())


@app.route("/")
def index() -> Any:
    """メインページを返す"""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/status")
def get_status() -> Any:
    """処理状態を取得"""
    return jsonify(processing_status)


@app.route("/api/plugins")
def get_plugins() -> Any:
    """利用可能なプラグインリストを取得"""
    plugins = []
    for plugin_info in plugin_manager.list_plugins():
        plugin = plugin_manager.get_plugin(plugin_info["name"])
        if plugin:
            info = plugin.get_info()
            plugins.append(
                {
                    "name": info["name"],
                    "type": info["type"],
                    "version": info["version"],
                    "description": info.get("description", "No description"),
                    "enabled": plugin.enabled,
                }
            )
    return jsonify(plugins)


@app.route("/api/upload", methods=["POST"])
def upload_file() -> Any:
    """ファイルをアップロード"""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    file_type = request.form.get("type", "unknown")

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # タイプ別にプレフィックスを付ける
        filename = f"{file_type}_{filename}"
        filepath = Path(app.config["UPLOAD_FOLDER"]) / filename
        file.save(str(filepath))

        return jsonify({"success": True, "filename": filename, "path": str(filepath)})

    return jsonify({"error": "Invalid file type"}), 400


@app.route("/api/process", methods=["POST"])
def process() -> Any:
    """音声処理を実行"""
    global processing_status

    if processing_status["is_processing"]:
        return jsonify({"error": "Already processing"}), 400

    data = request.json

    # 必須パラメータのチェック
    vocal_path = data.get("vocal_path")
    bgm_path = data.get("bgm_path")

    if not vocal_path or not bgm_path:
        return jsonify({"error": "Missing required files"}), 400

    # 出力ファイル名を生成
    output_filename = f"output_{Path(vocal_path).stem}_{Path(bgm_path).stem}.mp4"
    if data.get("audio_only"):
        output_filename = output_filename.replace(".mp4", ".wav")

    output_path = Path(app.config["OUTPUT_FOLDER"]) / output_filename

    # 設定を準備
    settings = {
        "preset": data.get("preset", "pop"),
        "vocal_volume": data.get("vocal_volume", 0),
        "bgm_volume": data.get("bgm_volume", 0),
        "video_template": data.get("video_template", "modern"),
        "audio_only": data.get("audio_only", False),
        "reverb": data.get("reverb"),
        "chunk_processing": data.get("chunk_processing", False),
        "streaming": data.get("streaming", False),
        "preview_mode": data.get("preview_mode", False),
        "plugin_manager": plugin_manager,
    }

    # バックグラウンドで処理を実行
    def process_thread() -> None:
        global processing_status
        processing_status["is_processing"] = True
        processing_status["progress"] = 0
        processing_status["message"] = "Processing..."
        processing_status["error"] = None

        try:
            # Webプログレスクラス
            class WebProgress:
                def __init__(self) -> None:
                    self.tasks: list[dict[str, Any]] = []

                def add_task(self, description: str, total: int) -> int:
                    task_id = len(self.tasks)
                    self.tasks.append({"description": description, "total": total, "completed": 0})
                    processing_status["message"] = description
                    return task_id

                def update(self, task_id: int, advance: int) -> None:
                    if 0 <= task_id < len(self.tasks):
                        self.tasks[task_id]["completed"] += advance
                        # 全体の進捗を計算
                        total_progress = (
                            sum(t["completed"] / t["total"] for t in self.tasks)
                            / len(self.tasks)
                            * 100
                        )
                        processing_status["progress"] = int(total_progress)

            progress = WebProgress()

            # 処理を実行
            process_audio(
                Path(vocal_path), Path(bgm_path), output_path, settings, progress, verbose=False
            )

            processing_status["progress"] = 100
            processing_status["message"] = "Complete!"
            processing_status["output_path"] = str(output_path)
            processing_status["output_filename"] = output_filename

        except Exception as e:
            processing_status["error"] = str(e)
            processing_status["message"] = "Error occurred"
        finally:
            processing_status["is_processing"] = False

    thread = threading.Thread(target=process_thread)
    thread.daemon = True
    thread.start()

    return jsonify({"success": True, "message": "Processing started"})


@app.route("/api/download/<filename>")
def download_file(filename: str) -> Any:
    """処理済みファイルをダウンロード"""
    try:
        return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404


@app.route("/api/presets")
def get_presets() -> Any:
    """利用可能なプリセットを取得"""
    presets = [
        {"id": "pop", "name": "Pop", "description": "Bright and vocal-forward mix"},
        {"id": "rock", "name": "Rock", "description": "Powerful and balanced mix"},
        {"id": "ballad", "name": "Ballad", "description": "Warm and vocal-focused mix"},
    ]
    return jsonify(presets)


def run_server(
    host: str = "127.0.0.1", port: int = 5000, debug: bool = False, open_browser: bool = True
) -> None:
    """Webサーバーを起動"""
    init_plugins()

    if open_browser:
        # ブラウザを開く
        def open_browser_delayed() -> None:
            import time

            time.sleep(1.5)  # サーバー起動を待つ
            webbrowser.open(f"http://{host}:{port}")

        thread = threading.Thread(target=open_browser_delayed)
        thread.daemon = True
        thread.start()

    print(f"Starting AutoMix Web Interface at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server()
