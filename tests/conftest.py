"""
pytest設定とフィクスチャ
"""

import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# ログ設定
logging.basicConfig(level=logging.INFO)


@pytest.fixture(scope="session")
def test_data_dir():
    """テストデータディレクトリ"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_rate():
    """標準サンプルレート"""
    return 44100


@pytest.fixture
def create_sine_wave():
    """正弦波生成フィクスチャ"""

    def _create(frequency=440, duration=1.0, sample_rate=44100, amplitude=0.5):
        t = np.linspace(0, duration, int(sample_rate * duration))
        return amplitude * np.sin(2 * np.pi * frequency * t)

    return _create


@pytest.fixture
def create_audio_file(test_data_dir, sample_rate):
    """音声ファイル作成フィクスチャ"""
    created_files = []

    def _create(filename="test.wav", duration=1.0, frequency=440, amplitude=0.5, stereo=False):
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = amplitude * np.sin(2 * np.pi * frequency * t)

        if stereo:
            # ステレオの場合は左右で少し違う周波数
            audio_r = amplitude * np.sin(2 * np.pi * (frequency * 1.01) * t)
            audio = np.column_stack([audio, audio_r])

        filepath = test_data_dir / filename
        sf.write(filepath, audio, sample_rate)
        created_files.append(filepath)

        return filepath

    yield _create

    # クリーンアップ
    for filepath in created_files:
        filepath.unlink(missing_ok=True)


@pytest.fixture
def mock_audio_data():
    """モック音声データ生成"""

    def _create(duration=1.0, sample_rate=44100, num_channels=1, frequency_components=None):
        if frequency_components is None:
            frequency_components = [(440, 0.5), (880, 0.3), (1320, 0.2)]

        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.zeros_like(t)

        for freq, amp in frequency_components:
            audio += amp * np.sin(2 * np.pi * freq * t)

        if num_channels == 2:
            audio = np.column_stack([audio, audio * 0.9])

        return audio, sample_rate

    return _create


@pytest.fixture(autouse=True)
def reset_matplotlib():
    """matplotlibの状態をリセット"""
    import matplotlib.pyplot as plt

    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def performance_timer():
    """パフォーマンス計測用タイマー"""
    import time

    class Timer:
        def __init__(self):
            self.times = {}

        def start(self, name):
            self.times[name] = {"start": time.time()}

        def stop(self, name):
            if name in self.times and "start" in self.times[name]:
                self.times[name]["duration"] = time.time() - self.times[name]["start"]
                return self.times[name]["duration"]
            return None

        def get_duration(self, name):
            if name in self.times and "duration" in self.times[name]:
                return self.times[name]["duration"]
            return None

        def report(self):
            for name, data in self.times.items():
                if "duration" in data:
                    print(f"{name}: {data['duration']:.3f}s")

    return Timer()


@pytest.fixture
def mock_video_frames():
    """モック動画フレーム生成"""

    def _create(width=1280, height=720, num_frames=30, fps=30):
        frames = []
        for i in range(num_frames):
            # グラデーションフレーム
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = np.linspace(0, 255, width)  # R
            frame[:, :, 1] = np.linspace(0, 255, height).reshape(-1, 1)  # G
            frame[:, :, 2] = i * 255 // num_frames  # B (時間変化)
            frames.append(frame)

        return frames, fps

    return _create


# カスタムマーカー
def pytest_configure(config):
    """カスタムマーカーの登録"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")


# テスト実行時の設定
def pytest_collection_modifyitems(config, items):
    """テスト実行時の動的な設定"""
    # CI環境では遅いテストをスキップ
    if config.getoption("--ci"):
        skip_slow = pytest.mark.skip(reason="Skipping slow tests in CI")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def pytest_addoption(parser):
    """コマンドラインオプションの追加"""
    parser.addoption(
        "--ci", action="store_true", default=False, help="Run in CI mode (skip slow tests)"
    )
    parser.addoption(
        "--performance", action="store_true", default=False, help="Run performance tests"
    )
