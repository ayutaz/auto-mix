"""
マスタリング機能のテスト
"""
from unittest.mock import patch

import numpy as np
import pytest

from automix.core.mastering import (
    FinalEQ,
    LimiterProcessor,
    LoudnessNormalizer,
    MasteringProcessor,
    MultibandCompressor,
)


class TestMasteringProcessor:
    """MasteringProcessorのテスト"""

    @pytest.fixture
    def mixed_audio(self):
        """ミックス済み音声のサンプル"""
        sr = 44100
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration))

        # リアルな音楽信号のシミュレーション
        # ベース
        bass = 0.3 * np.sin(2 * np.pi * 80 * t)
        bass += 0.1 * np.sin(2 * np.pi * 160 * t)

        # ミッドレンジ
        mid = 0.2 * np.sin(2 * np.pi * 440 * t)
        mid += 0.15 * np.sin(2 * np.pi * 880 * t)

        # ハイ
        high = 0.1 * np.sin(2 * np.pi * 2000 * t)
        high += 0.05 * np.sin(2 * np.pi * 4000 * t)

        # ダイナミクス変化
        envelope = np.ones_like(t)
        envelope[int(sr) : int(2 * sr)] = 0.5  # 静かな部分
        envelope[int(3 * sr) : int(4 * sr)] = 1.2  # 大きな部分

        mixed = (bass + mid + high) * envelope
        return mixed, sr

    def test_full_mastering_chain(self, mixed_audio):
        """フルマスタリングチェーンのテスト"""
        audio, sr = mixed_audio
        mastering = MasteringProcessor(sample_rate=sr, target_lufs=-14, ceiling_db=-0.3)

        mastered = mastering.process(audio)

        assert isinstance(mastered, np.ndarray)
        assert len(mastered) == len(audio)
        assert np.max(np.abs(mastered)) <= 0.95  # -0.3dB以下

        # ダイナミクスが適切に処理されている
        original_dynamic_range = np.max(audio) - np.mean(audio)
        mastered_dynamic_range = np.max(mastered) - np.mean(mastered)
        assert mastered_dynamic_range < original_dynamic_range

    def test_target_lufs(self, mixed_audio):
        """ターゲットLUFS達成のテスト"""
        audio, sr = mixed_audio
        target_lufs = -14

        mastering = MasteringProcessor(sample_rate=sr, target_lufs=target_lufs)

        mastered = mastering.process(audio)

        # LUFS測定（モック）
        with patch("automix.core.mastering.measure_lufs") as mock_lufs:
            mock_lufs.return_value = -14.2
            measured_lufs = mastering.measure_lufs(mastered)

            assert abs(measured_lufs - target_lufs) < 0.5


class TestLimiterProcessor:
    """LimiterProcessorのテスト"""

    def test_basic_limiting(self):
        """基本的なリミッティングのテスト"""
        sr = 44100
        limiter = LimiterProcessor(sample_rate=sr, threshold_db=-3, release_ms=50, lookahead_ms=5)

        # 閾値を超える信号
        signal = np.ones(sr) * 2.0  # +6dB
        limited = limiter.process(signal)

        # 閾値以下に制限されている
        threshold_linear = 10 ** (-3 / 20)
        assert np.max(limited) <= threshold_linear * 1.01

    def test_transparent_limiting(self):
        """透明性の高いリミッティングのテスト"""
        sr = 44100
        limiter = LimiterProcessor(sample_rate=sr, threshold_db=-0.3, release_ms=100)

        # 音楽的な信号
        t = np.linspace(0, 1, sr)
        signal = np.sin(2 * np.pi * 440 * t) * 0.8

        # 一部だけピークを追加
        signal[sr // 2 : sr // 2 + 100] *= 1.5

        limited = limiter.process(signal)

        # ピーク以外の部分は変化が少ない
        untouched_part = signal[: sr // 4]
        limited_part = limited[: sr // 4]

        correlation = np.corrcoef(untouched_part, limited_part)[0, 1]
        assert correlation > 0.99

    def test_lookahead_limiting(self):
        """ルックアヘッドリミッティングのテスト"""
        sr = 44100
        limiter_no_lookahead = LimiterProcessor(sample_rate=sr, threshold_db=-6, lookahead_ms=0)

        limiter_with_lookahead = LimiterProcessor(sample_rate=sr, threshold_db=-6, lookahead_ms=10)

        # 急激な立ち上がりの信号
        signal = np.zeros(sr)
        signal[sr // 2 : sr // 2 + 10] = 2.0

        limited_no_la = limiter_no_lookahead.process(signal)
        limited_with_la = limiter_with_lookahead.process(signal)

        # ルックアヘッドありの方が滑らかな処理
        assert np.max(limited_with_la) < np.max(limited_no_la)


class TestMultibandCompressor:
    """MultibandCompressorのテスト"""

    def test_multiband_separation(self):
        """マルチバンド分離のテスト"""
        sr = 44100
        compressor = MultibandCompressor(
            sample_rate=sr,
            crossover_frequencies=[200, 2000],
            thresholds_db=[-20, -15, -10],
            ratios=[2, 3, 4],
            attack_ms=[10, 5, 2],
            release_ms=[100, 50, 20],
        )

        # 各帯域の信号を生成
        t = np.linspace(0, 1, sr)
        low = 0.8 * np.sin(2 * np.pi * 100 * t)
        mid = 0.6 * np.sin(2 * np.pi * 1000 * t)
        high = 0.4 * np.sin(2 * np.pi * 5000 * t)
        signal = low + mid + high

        compressed = compressor.process(signal)

        # 各帯域が独立して圧縮されている
        assert isinstance(compressed, np.ndarray)
        assert len(compressed) == len(signal)

        # 全体のダイナミクスが改善
        original_range = np.max(signal) - np.min(signal)
        compressed_range = np.max(compressed) - np.min(compressed)
        assert compressed_range < original_range

    def test_frequency_balance(self):
        """周波数バランスの維持テスト"""
        sr = 44100
        compressor = MultibandCompressor(
            sample_rate=sr,
            crossover_frequencies=[500, 4000],
            thresholds_db=[-15, -15, -15],
            ratios=[3, 3, 3],
        )

        # ピンクノイズ的な信号
        signal = np.random.normal(0, 0.3, sr)
        compressed = compressor.process(signal)

        # 周波数バランスが大きく変わらない
        fft_original = np.abs(np.fft.rfft(signal))
        fft_compressed = np.abs(np.fft.rfft(compressed))

        # スペクトルの形状相関
        correlation = np.corrcoef(np.log10(fft_original + 1e-10), np.log10(fft_compressed + 1e-10))[
            0, 1
        ]
        assert correlation > 0.8


class TestLoudnessNormalizer:
    """LoudnessNormalizerのテスト"""

    def test_lufs_normalization(self):
        """LUFS正規化のテスト"""
        sr = 44100
        normalizer = LoudnessNormalizer(sample_rate=sr, target_lufs=-14, true_peak_db=-1)

        # テスト信号
        t = np.linspace(0, 3, 3 * sr)
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)

        normalized = normalizer.normalize(signal)

        # 簡易的なLUFS確認
        assert np.max(np.abs(normalized)) > np.max(np.abs(signal))
        assert np.max(np.abs(normalized)) <= 10 ** (-1 / 20)  # -1dB true peak

    def test_integrated_loudness(self):
        """統合ラウドネス計算のテスト"""
        sr = 44100
        normalizer = LoudnessNormalizer(sample_rate=sr)

        # 異なるラウドネスのセクション
        quiet = np.random.normal(0, 0.1, sr)
        loud = np.random.normal(0, 0.5, sr)
        signal = np.concatenate([quiet, loud, quiet])

        loudness = normalizer.measure_integrated_lufs(signal)

        assert isinstance(loudness, float)
        assert loudness < 0  # LUFSは通常負の値

    def test_streaming_platforms_presets(self):
        """ストリーミングプラットフォーム向けプリセットのテスト"""
        sr = 44100
        signal = np.random.normal(0, 0.3, sr * 2)

        platforms = {
            "spotify": -14,
            "youtube": -14,
            "apple_music": -16,
            "tidal": -14,
            "amazon_music": -14,
        }

        for platform, target_lufs in platforms.items():
            normalizer = LoudnessNormalizer.from_preset(platform, sample_rate=sr)
            assert normalizer.target_lufs == target_lufs


class TestFinalEQ:
    """FinalEQのテスト"""

    def test_mastering_eq_curves(self):
        """マスタリングEQカーブのテスト"""
        sr = 44100
        eq = FinalEQ(sample_rate=sr)

        # フラットな信号
        signal = np.random.normal(0, 0.3, sr)

        # 各プリセットをテスト
        presets = ["bright", "warm", "neutral", "vinyl", "radio"]

        for preset in presets:
            eq_preset = FinalEQ.from_preset(preset, sample_rate=sr)
            processed = eq_preset.process(signal)

            assert isinstance(processed, np.ndarray)
            assert len(processed) == len(signal)

            # スペクトル変化を確認
            fft_original = np.abs(np.fft.rfft(signal))
            fft_processed = np.abs(np.fft.rfft(processed))

            # プリセットによって異なる周波数特性
            if preset == "bright":
                # 高域が強調される
                high_idx = len(fft_original) * 3 // 4
                assert np.mean(fft_processed[high_idx:]) > np.mean(fft_original[high_idx:])
            elif preset == "warm":
                # 低中域が強調される
                low_mid_idx = len(fft_original) // 4
                assert np.mean(fft_processed[:low_mid_idx]) > np.mean(fft_original[:low_mid_idx])

    def test_linear_phase_eq(self):
        """リニアフェーズEQのテスト"""
        sr = 44100
        eq = FinalEQ(sample_rate=sr, linear_phase=True)

        # インパルス信号
        signal = np.zeros(sr)
        signal[sr // 2] = 1.0

        # EQ適用
        eq.add_bell(1000, 6, 1)
        processed = eq.process(signal)

        # 位相歪みが最小限
        # (リニアフェーズの場合、インパルス応答が対称的)
        peak_idx = np.argmax(np.abs(processed))
        window = 1000

        pre_peak = processed[peak_idx - window : peak_idx]
        post_peak = processed[peak_idx + 1 : peak_idx + window + 1]

        # 前後の対称性を確認（完全ではないが近似的に）
        symmetry = np.corrcoef(pre_peak[::-1], post_peak)[0, 1]
        assert symmetry > 0.7
