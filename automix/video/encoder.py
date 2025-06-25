"""
動画エンコードモジュール
"""

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    import moviepy.editor as mpe
except ImportError:
    mpe = None


@dataclass
class VideoSettings:
    """動画設定"""

    width: int = 1920
    height: int = 1080
    fps: int = 30
    bitrate: str = "8M"
    codec: str = "h264"
    preset: str = "medium"
    crf: int = 23
    audio_bitrate: str = "192k"
    audio_codec: str = "aac"

    @classmethod
    def from_preset(cls, preset_name: str) -> "VideoSettings":
        """プリセットから設定を作成"""
        presets = {
            "4k": cls(width=3840, height=2160, bitrate="20M", crf=20),
            "1080p": cls(width=1920, height=1080, bitrate="8M", crf=23),
            "720p": cls(width=1280, height=720, bitrate="5M", crf=23),
            "480p": cls(width=854, height=480, bitrate="2.5M", crf=25),
            "hq": cls(bitrate="15M", crf=18, preset="slow"),
            "fast": cls(preset="veryfast", crf=28),
        }

        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}")

        return presets[preset_name]


class CodecOptions:
    """コーデックオプション"""

    @staticmethod
    def get_options(codec: str) -> dict:
        """コーデック別のオプションを取得"""
        options = {
            "h264": {
                "codec": "libx264",
                "preset": "medium",
                "crf": 23,
                "profile:v": "high",
                "level": "4.1",
            },
            "h265": {"codec": "libx265", "preset": "medium", "crf": 28, "tag:v": "hvc1"},
            "vp9": {"codec": "libvpx-vp9", "crf": 31, "b:v": "0"},
            "av1": {"codec": "libsvtav1", "preset": 6, "crf": 35},
        }

        return options.get(codec, options["h264"])


class VideoEncoder:
    """動画エンコーダークラス"""

    def __init__(self, settings: VideoSettings):
        """
        Args:
            settings: 動画設定
        """
        self.settings = settings
        self.codec_options = CodecOptions.get_options(settings.codec)

    def encode(
        self,
        frames: list[NDArray[np.uint8]] | NDArray[np.uint8],
        audio: NDArray[np.float32],
        output_path: str | Path,
        sample_rate: int = 44100,
        show_progress: bool = True,
    ) -> None:
        """
        フレームと音声から動画を作成

        Args:
            frames: フレームのリストまたは配列
            audio: 音声データ
            output_path: 出力パス
            sample_rate: 音声サンプルレート
            show_progress: 進捗表示
        """
        output_path = Path(output_path)

        # 一時ディレクトリ
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)

            # フレームを一時ファイルに保存
            frames_pattern = str(temp_dir / "frame_%06d.png")
            self._save_frames(frames, frames_pattern)

            # 音声を一時ファイルに保存
            audio_path = temp_dir / "audio.wav"
            self._save_audio(audio, audio_path, sample_rate)

            # 動画を作成
            self._create_video(frames_pattern, audio_path, output_path, show_progress)

    def encode_with_moviepy(
        self,
        frames: list[NDArray[np.uint8]],
        audio: NDArray[np.float32],
        output_path: str | Path,
        sample_rate: int = 44100,
    ) -> None:
        """
        MoviePyを使用してエンコード

        Args:
            frames: フレームのリスト
            audio: 音声データ
            output_path: 出力パス
            sample_rate: 音声サンプルレート
        """
        output_path = Path(output_path)

        # ビデオクリップを作成
        def make_frame(t: float) -> NDArray[np.uint8]:
            """時刻tのフレームを返す"""
            frame_idx = int(t * self.settings.fps)
            if frame_idx >= len(frames):
                frame_idx = len(frames) - 1
            return frames[frame_idx]

        duration = len(frames) / self.settings.fps
        video_clip = mpe.VideoClip(make_frame, duration=duration)
        video_clip = video_clip.set_fps(self.settings.fps)

        # オーディオクリップを作成
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        audio_clip = mpe.AudioClip(
            lambda t: audio[int(t * sample_rate)] if t * sample_rate < len(audio) else 0,
            duration=duration,
            fps=sample_rate,
        )

        # 合成
        final_clip = video_clip.set_audio(audio_clip)

        # エンコード
        final_clip.write_videofile(
            str(output_path),
            codec=self.codec_options["codec"],
            bitrate=self.settings.bitrate,
            audio_codec=self.settings.audio_codec,
            audio_bitrate=self.settings.audio_bitrate,
            preset=self.settings.preset,
            ffmpeg_params=["-crf", str(self.settings.crf)],
        )

    def _save_frames(
        self, frames: list[NDArray[np.uint8]] | NDArray[np.uint8], pattern: str
    ) -> None:
        """フレームを画像ファイルとして保存"""
        if isinstance(frames, np.ndarray):
            # 4次元配列の場合
            if frames.ndim == 4:
                frames_list = [frames[i] for i in range(frames.shape[0])]
            else:
                frames_list = [frames]
        else:
            frames_list = frames

        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is not installed. Please install opencv-python.")

        for i, frame in enumerate(frames_list):
            filename = pattern % i
            cv2.imwrite(filename, frame)

    def _save_audio(self, audio: NDArray[np.float32], audio_path: Path, sample_rate: int) -> None:
        """音声をWAVファイルとして保存"""
        import soundfile as sf

        # 正規化
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio_normalized = audio / max_val * 0.95
        else:
            audio_normalized = audio

        sf.write(audio_path, audio_normalized, sample_rate)

    def _create_video(
        self, frames_pattern: str, audio_path: Path, output_path: Path, show_progress: bool
    ) -> None:
        """FFmpegを使用して動画を作成"""
        import subprocess

        # FFmpegコマンドを構築
        cmd = [
            "ffmpeg",
            "-y",  # 上書き
            "-framerate",
            str(self.settings.fps),
            "-i",
            frames_pattern,
            "-i",
            str(audio_path),
            "-c:v",
            self.codec_options["codec"],
            "-preset",
            self.settings.preset,
            "-crf",
            str(self.settings.crf),
            "-b:v",
            self.settings.bitrate,
            "-c:a",
            self.settings.audio_codec,
            "-b:a",
            self.settings.audio_bitrate,
            "-pix_fmt",
            "yuv420p",  # 互換性のため
            str(output_path),
        ]

        # プロファイル設定
        if "profile:v" in self.codec_options:
            cmd.extend(["-profile:v", self.codec_options["profile:v"]])

        # 進捗表示設定
        if not show_progress:
            cmd.extend(["-loglevel", "error"])

        # 実行
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg encoding failed: {e}")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg and ensure it's in PATH.")


class StreamingEncoder:
    """ストリーミング用エンコーダー"""

    def __init__(self, settings: VideoSettings):
        """
        Args:
            settings: 動画設定
        """
        self.settings = settings
        self.process: subprocess.Popen | None = None

    def start_encoding(self, output_url: str, audio_sample_rate: int = 44100) -> None:
        """
        ストリーミングエンコードを開始

        Args:
            output_url: 出力URL（RTMPなど）
            audio_sample_rate: 音声サンプルレート
        """
        import subprocess

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.settings.width}x{self.settings.height}",
            "-r",
            str(self.settings.fps),
            "-i",
            "-",  # stdin
            "-f",
            "s16le",
            "-ar",
            str(audio_sample_rate),
            "-ac",
            "2",
            "-i",
            "-",  # stdin
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "zerolatency",
            "-b:v",
            self.settings.bitrate,
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-f",
            "flv",
            output_url,
        ]

        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write_frame(self, frame: NDArray[np.uint8]) -> None:
        """フレームを書き込み"""
        if self.process and self.process.stdin:
            self.process.stdin.write(frame.tobytes())

    def write_audio(self, audio: NDArray[np.float32]) -> None:
        """音声を書き込み"""
        if self.process and self.process.stdin:
            # Float32 to Int16変換
            audio_int16 = (audio * 32767).astype(np.int16)
            self.process.stdin.write(audio_int16.tobytes())

    def stop_encoding(self) -> None:
        """エンコードを停止"""
        if self.process and self.process.stdin:
            self.process.stdin.close()
            self.process.wait()
            self.process = None


class ThumbnailGenerator:
    """サムネイル生成クラス"""

    @staticmethod
    def generate_from_video(
        video_path: str | Path,
        output_path: str | Path,
        time_offset: float = 5.0,
        size: tuple[int, int] | None = None,
    ) -> None:
        """
        動画からサムネイルを生成

        Args:
            video_path: 動画パス
            output_path: 出力パス
            time_offset: 取得時刻（秒）
            size: サムネイルサイズ
        """
        video_path = Path(video_path)
        output_path = Path(output_path)

        # 動画を開く
        cap = cv2.VideoCapture(str(video_path))

        # 指定時刻にシーク
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(fps * time_offset)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # フレームを読み込み
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("Failed to read frame from video")

        # リサイズ
        if size:
            frame = cv2.resize(frame, size)

        # 保存
        cv2.imwrite(str(output_path), frame)

    @staticmethod
    def generate_from_frames(
        frames: list[NDArray[np.uint8]],
        output_path: str | Path,
        frame_indices: list[int] | None = None,
        layout: str = "grid",
    ) -> None:
        """
        フレームからサムネイルを生成

        Args:
            frames: フレームリスト
            output_path: 出力パス
            frame_indices: 使用するフレームのインデックス
            layout: レイアウト ('single', 'grid', 'strip')
        """
        output_path = Path(output_path)

        if frame_indices is None:
            # デフォルト：均等に分割
            n_frames = min(9, len(frames))
            step = len(frames) // n_frames
            frame_indices = [i * step for i in range(n_frames)]

        selected_frames = [frames[i] for i in frame_indices if i < len(frames)]

        if layout == "single":
            # 単一フレーム
            thumbnail = selected_frames[len(selected_frames) // 2]
        elif layout == "strip":
            # 横並び
            thumbnail = np.hstack(selected_frames)
        else:  # grid
            # グリッド配置
            n_cols = int(np.sqrt(len(selected_frames)))
            n_rows = (len(selected_frames) + n_cols - 1) // n_cols

            # パディング
            while len(selected_frames) < n_rows * n_cols:
                selected_frames.append(np.zeros_like(selected_frames[0]))

            rows = []
            for i in range(n_rows):
                row = np.hstack(selected_frames[i * n_cols : (i + 1) * n_cols])
                rows.append(row)
            thumbnail = np.vstack(rows)

        # 保存
        cv2.imwrite(str(output_path), thumbnail)
