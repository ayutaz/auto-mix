"""
音声ファイル読み込みモジュール
"""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


class UnsupportedFormatError(Exception):
    """サポートされていないファイル形式のエラー"""

    pass


@dataclass
class AudioFile:
    """音声ファイルのデータクラス"""

    data: np.ndarray
    sample_rate: int
    duration: float
    channels: int
    file_path: Path | None = None

    def __post_init__(self) -> None:
        """初期化後の処理"""
        if self.data.ndim == 1:
            self.channels = 1
        else:
            self.channels = self.data.shape[1]

        self.duration = len(self.data) / self.sample_rate


class AudioLoader:
    """音声ファイル読み込みクラス"""

    SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}

    def __init__(
        self, target_sample_rate: int | None = None, normalize: bool = False, mono: bool = False
    ):
        """
        Args:
            target_sample_rate: ターゲットサンプルレート（Noneの場合は元のまま）
            normalize: 正規化するかどうか
            mono: モノラルに変換するかどうか
        """
        self.target_sample_rate = target_sample_rate
        self.normalize = normalize
        self.mono = mono

    def load(self, file_path: str | Path) -> AudioFile:
        """
        音声ファイルを読み込む

        Args:
            file_path: 音声ファイルのパス

        Returns:
            AudioFile: 読み込んだ音声データ

        Raises:
            FileNotFoundError: ファイルが存在しない
            UnsupportedFormatError: サポートされていない形式
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise UnsupportedFormatError(
                f"Unsupported format: {file_path.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        # ファイル形式に応じて読み込み
        if file_path.suffix.lower() in {".wav", ".flac"}:
            data, sample_rate = sf.read(file_path, dtype="float32")
        else:
            # MP3, M4A等はlibrosaで読み込み
            data, sample_rate = librosa.load(file_path, sr=self.target_sample_rate, mono=self.mono)

        # モノラル変換
        if self.mono and data.ndim > 1:
            data = np.mean(data, axis=1)

        # リサンプリング
        if self.target_sample_rate and sample_rate != self.target_sample_rate:
            if data.ndim == 1:
                data = librosa.resample(
                    data, orig_sr=sample_rate, target_sr=self.target_sample_rate
                )
            else:
                # ステレオの場合は各チャンネルをリサンプリング
                resampled = []
                for channel in range(data.shape[1]):
                    resampled.append(
                        librosa.resample(
                            data[:, channel], orig_sr=sample_rate, target_sr=self.target_sample_rate
                        )
                    )
                data = np.column_stack(resampled)
            sample_rate = self.target_sample_rate

        # 正規化
        if self.normalize:
            max_val = np.max(np.abs(data))
            if max_val > 0:
                data = data / max_val

        return AudioFile(
            data=data,
            sample_rate=sample_rate,
            duration=len(data) / sample_rate,
            channels=1 if data.ndim == 1 else data.shape[1],
            file_path=file_path,
        )

    def save(
        self,
        audio_file: AudioFile,
        output_path: str | Path,
        format: str | None = None,
        subtype: str | None = None,
    ) -> None:
        """
        音声ファイルを保存する

        Args:
            audio_file: 保存する音声データ
            output_path: 出力パス
            format: ファイル形式（Noneの場合は拡張子から判断）
            subtype: サブタイプ（例: 'PCM_16', 'PCM_24'）
        """
        output_path = Path(output_path)

        # フォーマットの決定
        if format is None:
            format = output_path.suffix[1:].upper()

        # サブタイプのデフォルト設定
        if subtype is None:
            if format == "WAV" or format == "FLAC":
                subtype = "PCM_24"

        # 保存
        sf.write(
            output_path, audio_file.data, audio_file.sample_rate, format=format, subtype=subtype
        )

    def is_supported_format(self, file_path: str | Path) -> bool:
        """
        サポートされているファイル形式かどうかを確認

        Args:
            file_path: ファイルパス

        Returns:
            bool: サポートされているかどうか
        """
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.SUPPORTED_FORMATS

    def get_file_info(self, file_path: str | Path) -> dict:
        """
        音声ファイルの情報を取得（読み込まずに）

        Args:
            file_path: ファイルパス

        Returns:
            dict: ファイル情報
        """
        file_path = Path(file_path)
        info = sf.info(file_path)

        return {
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "subtype": info.subtype,
            "frames": info.frames,
        }
