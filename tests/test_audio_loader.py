"""
音声ファイル読み込み機能のテスト
"""
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from automix.core.audio_loader import AudioFile, AudioLoader, UnsupportedFormatError


class TestAudioLoader:
    """AudioLoaderクラスのテスト"""

    @pytest.fixture
    def sample_audio_data(self):
        """テスト用のサンプル音声データを生成"""
        sample_rate = 44100
        duration = 1.0  # 1秒
        frequency = 440.0  # A4
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
        return audio_data, sample_rate

    @pytest.fixture
    def temp_audio_file(self, sample_audio_data):
        """一時的な音声ファイルを作成"""
        audio_data, sample_rate = sample_audio_data
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_data, sample_rate)
            yield Path(f.name)
        # クリーンアップ
        Path(f.name).unlink(missing_ok=True)

    def test_load_wav_file(self, temp_audio_file, sample_audio_data):
        """WAVファイルの読み込みテスト"""
        loader = AudioLoader()
        audio_file = loader.load(temp_audio_file)

        assert isinstance(audio_file, AudioFile)
        assert audio_file.sample_rate == sample_audio_data[1]
        assert len(audio_file.data) == len(sample_audio_data[0])
        assert audio_file.duration > 0
        assert audio_file.channels == 1

    def test_load_stereo_file(self, sample_audio_data):
        """ステレオファイルの読み込みテスト"""
        audio_data, sample_rate = sample_audio_data
        stereo_data = np.column_stack([audio_data, audio_data])

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, stereo_data, sample_rate)
            temp_file = Path(f.name)

        try:
            loader = AudioLoader()
            audio_file = loader.load(temp_file)

            assert audio_file.channels == 2
            assert audio_file.data.shape == (len(audio_data), 2)
        finally:
            temp_file.unlink(missing_ok=True)

    def test_load_mp3_file(self):
        """MP3ファイルの読み込みテスト（モック使用）"""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_file = Path(f.name)

        try:
            with patch("automix.core.audio_loader.librosa.load") as mock_load:
                mock_load.return_value = (np.zeros(44100), 44100)

                loader = AudioLoader()
                audio_file = loader.load(temp_file)

                assert isinstance(audio_file, AudioFile)
                mock_load.assert_called_once()
        finally:
            temp_file.unlink(missing_ok=True)

    def test_load_nonexistent_file(self):
        """存在しないファイルの読み込みテスト"""
        loader = AudioLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(Path("nonexistent.wav"))

    def test_load_unsupported_format(self):
        """サポートされていない形式のテスト"""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            temp_file = Path(f.name)

        try:
            loader = AudioLoader()
            with pytest.raises(UnsupportedFormatError):
                loader.load(temp_file)
        finally:
            temp_file.unlink(missing_ok=True)

    def test_resample_audio(self, temp_audio_file):
        """リサンプリング機能のテスト"""
        loader = AudioLoader(target_sample_rate=48000)
        audio_file = loader.load(temp_audio_file)

        assert audio_file.sample_rate == 48000

    def test_normalize_audio(self, temp_audio_file):
        """正規化機能のテスト"""
        loader = AudioLoader(normalize=True)
        audio_file = loader.load(temp_audio_file)

        # 正規化後の最大値は1.0以下
        assert np.max(np.abs(audio_file.data)) <= 1.0

    def test_load_multiple_formats(self):
        """複数フォーマットの読み込みテスト"""
        formats = [".wav", ".mp3", ".flac", ".m4a"]
        loader = AudioLoader()

        for fmt in formats:
            assert loader.is_supported_format(Path(f"test{fmt}"))

    @pytest.mark.parametrize(
        "sample_rate,expected",
        [
            (44100, 44100),
            (48000, 48000),
            (96000, 96000),
        ],
    )
    def test_various_sample_rates(self, sample_rate, expected):
        """様々なサンプルレートのテスト"""
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_data, sample_rate)
            temp_file = Path(f.name)

        try:
            loader = AudioLoader()
            audio_file = loader.load(temp_file)
            assert audio_file.sample_rate == expected
        finally:
            temp_file.unlink(missing_ok=True)

    def test_concurrent_loading(self, temp_audio_file):
        """並行読み込みのテスト"""
        import concurrent.futures

        loader = AudioLoader()

        def load_file():
            return loader.load(temp_audio_file)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(load_file) for _ in range(4)]
            results = [f.result() for f in futures]

        assert all(isinstance(r, AudioFile) for r in results)
        assert len(results) == 4
