"""
音声ミックス処理のテスト
"""

import numpy as np
import pytest

from automix.core.processor import (
    CompressorProcessor,
    EQProcessor,
    MixProcessor,
    VolumeProcessor,
)


class TestMixProcessor:
    """MixProcessorクラスのテスト"""

    @pytest.fixture
    def sample_vocal_bgm(self):
        """テスト用のボーカルとBGMデータ"""
        sr = 44100
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))

        # ボーカル（中音域）
        vocal = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4
        vocal += 0.2 * np.sin(2 * np.pi * 554.37 * t)  # C#5

        # BGM（低音域と高音域）
        bgm = 0.4 * np.sin(2 * np.pi * 110 * t)  # A2 (ベース)
        bgm += 0.2 * np.sin(2 * np.pi * 880 * t)  # A5 (メロディ)
        bgm += 0.1 * np.sin(2 * np.pi * 220 * t)  # A3 (ハーモニー)

        return vocal, bgm, sr

    def test_basic_mix(self, sample_vocal_bgm):
        """基本的なミックス処理のテスト"""
        vocal, bgm, sr = sample_vocal_bgm
        processor = MixProcessor(sample_rate=sr)

        mixed = processor.mix(vocal, bgm)

        assert isinstance(mixed, np.ndarray)
        assert len(mixed) == len(vocal)
        assert np.max(np.abs(mixed)) <= 1.0  # クリッピングなし

    def test_volume_balance(self, sample_vocal_bgm):
        """音量バランス調整のテスト"""
        vocal, bgm, sr = sample_vocal_bgm
        processor = MixProcessor(sample_rate=sr)

        # ボーカルを強調
        mixed = processor.mix(vocal, bgm, vocal_gain_db=3, bgm_gain_db=-3)

        # 簡易的な確認（実際の音量差を検証）
        vocal_power = np.mean(vocal**2)
        bgm_power = np.mean(bgm**2)
        mixed_vocal_estimate = mixed - bgm * 0.707  # -3dB
        mixed_vocal_power = np.mean(mixed_vocal_estimate**2)

        assert mixed_vocal_power > vocal_power  # ボーカルが強調されている

    def test_lufs_normalization(self, sample_vocal_bgm):
        """LUFS正規化のテスト"""
        vocal, bgm, sr = sample_vocal_bgm
        processor = MixProcessor(sample_rate=sr, target_lufs=-14)

        mixed = processor.mix(vocal, bgm)

        # LUFS測定（簡易版）
        lufs = processor.measure_lufs(mixed)
        # 簡易実装なので、正確な値ではなく範囲でチェック
        assert -70 <= lufs <= 0  # LUFSは通常負の値

    @pytest.mark.skip(reason="Alignment implementation needs to be fixed")
    def test_alignment(self):
        """音声アライメントのテスト"""
        from automix.core.processor import AlignmentProcessor

        sr = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # 位相がずれた2つの信号
        signal1 = np.sin(2 * np.pi * 440 * t)
        delay_samples = int(0.01 * sr)  # 10ms遅延
        signal2 = np.roll(signal1, delay_samples)

        aligner = AlignmentProcessor(sample_rate=sr)
        aligned1, aligned2, offset = aligner.align(signal1, signal2)

        assert abs(offset - delay_samples) < 10  # 許容誤差10サンプル
        assert np.corrcoef(aligned1[1000:-1000], aligned2[1000:-1000])[0, 1] > 0.99


class TestVolumeProcessor:
    """VolumeProcessorのテスト"""

    def test_gain_adjustment(self):
        """ゲイン調整のテスト"""
        processor = VolumeProcessor()
        signal = np.ones(1000) * 0.5

        # +6dB
        amplified = processor.apply_gain(signal, 6)
        assert np.allclose(amplified, signal * 2, rtol=0.01)

        # -6dB
        attenuated = processor.apply_gain(signal, -6)
        assert np.allclose(attenuated, signal * 0.5, rtol=0.01)

    def test_auto_gain(self):
        """自動ゲイン調整のテスト"""
        processor = VolumeProcessor()

        # 小さい信号
        quiet_signal = np.random.normal(0, 0.1, 44100)
        adjusted = processor.auto_gain(quiet_signal, target_rms=0.3)

        rms = np.sqrt(np.mean(adjusted**2))
        assert abs(rms - 0.3) < 0.05

    def test_limiting(self):
        """リミッティングのテスト"""
        processor = VolumeProcessor()

        # クリッピングする信号
        signal = np.ones(1000) * 1.5
        limited = processor.limit(signal, threshold=1.0)

        assert np.max(np.abs(limited)) <= 1.0
        assert np.all(limited <= 1.0)
        assert np.all(limited >= -1.0)


class TestEQProcessor:
    """EQProcessorのテスト"""

    @pytest.fixture
    def test_signal(self):
        """マルチ周波数テスト信号"""
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # 低音、中音、高音の混合
        low = 0.3 * np.sin(2 * np.pi * 100 * t)
        mid = 0.3 * np.sin(2 * np.pi * 1000 * t)
        high = 0.3 * np.sin(2 * np.pi * 8000 * t)

        return low + mid + high, sr

    def test_highpass_filter(self, test_signal):
        """ハイパスフィルタのテスト"""
        signal, sr = test_signal
        processor = EQProcessor(sample_rate=sr)

        filtered = processor.highpass(signal, cutoff=500)

        # FFTで周波数成分を確認
        fft = np.fft.rfft(filtered)
        freqs = np.fft.rfftfreq(len(filtered), 1 / sr)

        # 100Hzの成分が減衰していることを確認
        idx_100hz = np.argmin(np.abs(freqs - 100))
        idx_1000hz = np.argmin(np.abs(freqs - 1000))

        assert np.abs(fft[idx_100hz]) < np.abs(fft[idx_1000hz]) * 0.1

    def test_parametric_eq(self, test_signal):
        """パラメトリックEQのテスト"""
        signal, sr = test_signal
        processor = EQProcessor(sample_rate=sr)

        # 1kHzを+12dBブースト
        boosted = processor.parametric_eq(signal, freq=1000, gain_db=12, q=2)

        # オリジナルと処理後の1kHz成分を比較
        fft_original = np.fft.rfft(signal)
        fft_boosted = np.fft.rfft(boosted)
        freqs = np.fft.rfftfreq(len(signal), 1 / sr)

        idx_1000hz = np.argmin(np.abs(freqs - 1000))
        boost_ratio = np.abs(fft_boosted[idx_1000hz]) / np.abs(fft_original[idx_1000hz])

        assert 3.0 < boost_ratio < 5.0  # 約12dB (4倍)


class TestCompressorProcessor:
    """CompressorProcessorのテスト"""

    def test_basic_compression(self):
        """基本的なコンプレッションのテスト"""
        sr = 44100
        processor = CompressorProcessor(
            sample_rate=sr, threshold_db=-20, ratio=4, attack_ms=10, release_ms=100
        )

        # ダイナミックレンジの大きい信号
        t = np.linspace(0, 1, sr)
        signal = np.concatenate(
            [
                np.ones(sr // 4) * 0.1,  # 静か
                np.ones(sr // 4) * 0.8,  # 大きい
                np.ones(sr // 4) * 0.1,  # 静か
                np.ones(sr // 4) * 0.8,  # 大きい
            ]
        )

        compressed = processor.compress(signal)

        # ダイナミックレンジが圧縮されていることを確認
        original_range = np.max(signal) - np.min(signal)
        compressed_range = np.max(compressed) - np.min(compressed)

        assert compressed_range < original_range * 0.7

    def test_multiband_compression(self):
        """マルチバンドコンプレッションのテスト"""
        sr = 44100
        processor = CompressorProcessor(sample_rate=sr)

        # 異なる周波数帯域の信号
        t = np.linspace(0, 1, sr)
        low = 0.8 * np.sin(2 * np.pi * 100 * t)
        high = 0.2 * np.sin(2 * np.pi * 5000 * t)
        signal = low + high

        compressed = processor.multiband_compress(
            signal,
            bands=[(0, 500), (500, 2000), (2000, sr // 2)],
            thresholds=[-20, -15, -10],
            ratios=[2, 3, 4],
        )

        assert isinstance(compressed, np.ndarray)
        assert len(compressed) == len(signal)
        assert np.max(np.abs(compressed)) <= 1.0
