"""
パフォーマンス最適化モジュール
大きなファイルの効率的な処理のための機能を提供
"""
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Generator, Optional, Tuple

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from ..core.audio_loader import AudioFile


class ChunkedProcessor:
    """大きな音声ファイルをチャンクに分けて処理"""

    def __init__(
        self,
        chunk_duration: float = 30.0,  # 秒
        overlap_duration: float = 0.5,  # 秒
        max_workers: Optional[int] = None,
    ):
        """
        Args:
            chunk_duration: 各チャンクの長さ（秒）
            overlap_duration: チャンク間のオーバーラップ（秒）
            max_workers: 並列処理のワーカー数（Noneの場合はCPU数）
        """
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.max_workers = max_workers or mp.cpu_count()

    def read_chunks(
        self, file_path: Path, sample_rate: int = 44100
    ) -> Generator[Tuple[NDArray[np.float32], int, int], None, None]:
        """ファイルをチャンクごとに読み込む"""
        info = sf.info(str(file_path))
        total_frames = info.frames
        chunk_frames = int(self.chunk_duration * sample_rate)
        overlap_frames = int(self.overlap_duration * sample_rate)

        start = 0
        while start < total_frames:
            end = min(start + chunk_frames, total_frames)
            chunk, _ = sf.read(
                str(file_path), start=start, stop=end, dtype="float32", always_2d=False
            )
            yield chunk, start, end
            start = end - overlap_frames

    def process_file(
        self,
        file_path: Path,
        process_func: Callable[[NDArray[np.float32]], NDArray[np.float32]],
        output_path: Path,
        sample_rate: int = 44100,
    ) -> None:
        """大きなファイルをチャンク単位で処理"""
        info = sf.info(str(file_path))
        total_frames = info.frames

        # 出力ファイルを準備
        with sf.SoundFile(
            str(output_path),
            mode="w",
            samplerate=sample_rate,
            channels=info.channels,
            format=info.format,
        ) as output_file:
            overlap_buffer = None

            for chunk, start, end in self.read_chunks(file_path, sample_rate):
                # チャンクを処理
                processed_chunk = process_func(chunk)

                # オーバーラップ部分をクロスフェード
                if overlap_buffer is not None and start > 0:
                    overlap_frames = len(overlap_buffer)
                    fade_in = np.linspace(0, 1, overlap_frames)
                    fade_out = np.linspace(1, 0, overlap_frames)

                    processed_chunk[:overlap_frames] = (
                        processed_chunk[:overlap_frames] * fade_in + overlap_buffer * fade_out
                    )

                # 次のオーバーラップ用にバッファを保存
                if end < total_frames:
                    overlap_frames = int(self.overlap_duration * sample_rate)
                    overlap_buffer = processed_chunk[-overlap_frames:].copy()
                    # オーバーラップ部分を除いて書き込む
                    output_file.write(processed_chunk[:-overlap_frames])
                else:
                    # 最後のチャンク
                    output_file.write(processed_chunk)

                # メモリを解放
                del chunk
                gc.collect()


class StreamingProcessor:
    """ストリーミング処理によるメモリ効率的な処理"""

    def __init__(self, buffer_size: int = 4096):
        """
        Args:
            buffer_size: バッファサイズ（フレーム数）
        """
        self.buffer_size = buffer_size

    def process_stream(
        self,
        input_path: Path,
        output_path: Path,
        process_func: Callable[[NDArray[np.float32]], NDArray[np.float32]],
        sample_rate: int = 44100,
    ) -> None:
        """ストリーミング処理でファイルを処理"""
        with sf.SoundFile(str(input_path), "r") as input_file:
            channels = input_file.channels
            with sf.SoundFile(
                str(output_path),
                "w",
                samplerate=sample_rate,
                channels=channels,
                format="WAV",
            ) as output_file:
                while True:
                    # バッファサイズ分読み込む
                    data = input_file.read(self.buffer_size, dtype="float32")
                    if len(data) == 0:
                        break

                    # 処理
                    processed = process_func(data)

                    # 書き込み
                    output_file.write(processed)


class ParallelMixProcessor:
    """並列処理によるミックス処理の高速化"""

    def __init__(self, num_workers: Optional[int] = None):
        """
        Args:
            num_workers: ワーカー数（Noneの場合はCPU数）
        """
        self.num_workers = num_workers or mp.cpu_count()

    def parallel_mix(
        self,
        tracks: list[NDArray[np.float32]],
        gains: list[float],
        chunk_size: int = 44100,
    ) -> NDArray[np.float32]:
        """複数トラックを並列でミックス"""
        if not tracks:
            raise ValueError("No tracks to mix")

        # 最も長いトラックの長さを取得
        max_length = max(len(track) for track in tracks)

        # 全トラックを同じ長さにパディング
        padded_tracks = []
        for track in tracks:
            if len(track) < max_length:
                padded = np.pad(track, (0, max_length - len(track)), mode="constant")
                padded_tracks.append(padded)
            else:
                padded_tracks.append(track)

        # チャンクごとに並列処理
        num_chunks = (max_length + chunk_size - 1) // chunk_size
        mixed_chunks = []

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, max_length)

                # 各チャンクのデータを準備
                chunk_data = [(track[start:end], gain) for track, gain in zip(padded_tracks, gains)]
                future = executor.submit(self._mix_chunk, chunk_data)
                futures.append(future)

            # 結果を収集
            for future in futures:
                mixed_chunks.append(future.result())

        # チャンクを結合
        return np.concatenate(mixed_chunks)

    @staticmethod
    def _mix_chunk(chunk_data: list[Tuple[NDArray[np.float32], float]]) -> NDArray[np.float32]:
        """チャンクをミックス"""
        mixed = np.zeros_like(chunk_data[0][0])
        for chunk, gain in chunk_data:
            mixed += chunk * gain
        return mixed


class CacheManager:
    """処理結果のキャッシュ管理"""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Args:
            cache_dir: キャッシュディレクトリ（Noneの場合は一時ディレクトリ）
        """
        if cache_dir is None:
            import tempfile

            self.cache_dir = Path(tempfile.mkdtemp(prefix="automix_cache_"))
        else:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(exist_ok=True)

    def get_cache_path(self, key: str, suffix: str = ".npy") -> Path:
        """キャッシュファイルのパスを取得"""
        import hashlib

        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}{suffix}"

    def exists(self, key: str) -> bool:
        """キャッシュが存在するか確認"""
        return self.get_cache_path(key).exists()

    def save(self, key: str, data: NDArray[np.float32]) -> None:
        """データをキャッシュに保存"""
        cache_path = self.get_cache_path(key)
        np.save(cache_path, data)

    def load(self, key: str) -> Optional[NDArray[np.float32]]:
        """キャッシュからデータを読み込む"""
        cache_path = self.get_cache_path(key)
        if cache_path.exists():
            return np.load(cache_path)
        return None

    def clear(self) -> None:
        """キャッシュをクリア"""
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir()


class MemoryOptimizedProcessor:
    """メモリ使用量を最適化した処理"""

    @staticmethod
    def process_in_place(
        audio: NDArray[np.float32], process_func: Callable[[NDArray[np.float32]], None]
    ) -> NDArray[np.float32]:
        """インプレース処理でメモリ使用量を削減"""
        # ビューを作成して処理
        view = audio.view()
        process_func(view)
        return audio

    @staticmethod
    def downsample_for_preview(
        audio: NDArray[np.float32], target_duration: float = 30.0, sample_rate: int = 44100
    ) -> NDArray[np.float32]:
        """プレビュー用に音声をダウンサンプル"""
        target_samples = int(target_duration * sample_rate)
        if len(audio) <= target_samples:
            return audio

        # 中央部分を抽出
        start = (len(audio) - target_samples) // 2
        return audio[start : start + target_samples]

    @staticmethod
    def estimate_memory_usage(
        duration: float, sample_rate: int = 44100, channels: int = 2, bit_depth: int = 32
    ) -> float:
        """必要なメモリ使用量を推定（MB）"""
        samples = duration * sample_rate * channels
        bytes_per_sample = bit_depth // 8
        memory_mb = (samples * bytes_per_sample) / (1024 * 1024)
        # 処理用のバッファも考慮（約3倍）
        return memory_mb * 3


def optimize_for_gpu() -> bool:
    """GPU最適化が可能か確認し、有効化を試みる"""
    try:
        import cupy as cp

        # CuPyが利用可能
        print("GPU acceleration available via CuPy")
        return True
    except ImportError:
        print("GPU acceleration not available (CuPy not installed)")
        return False