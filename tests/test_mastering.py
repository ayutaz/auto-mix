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


class TestMasteringProcessorEdgeCases:
    """MasteringProcessorのエッジケーステスト"""
    
    def test_empty_audio(self):
        """空の音声データのテスト"""
        mastering = MasteringProcessor()
        empty = np.array([])
        processed = mastering.process(empty)
        assert len(processed) == 0
        
    def test_silence_processing(self):
        """無音処理のテスト"""
        sr = 44100
        mastering = MasteringProcessor(sample_rate=sr, target_lufs=-14)
        silence = np.zeros(sr)
        
        processed = mastering.process(silence)
        # 無音は無音のまま
        assert np.allclose(processed, 0)
        
    def test_clipping_prevention(self):
        """クリッピング防止のテスト"""
        sr = 44100
        mastering = MasteringProcessor(sample_rate=sr, ceiling_db=-0.1)
        
        # 非常に大きな信号
        loud_signal = np.ones(sr) * 10.0
        processed = mastering.process(loud_signal)
        
        # ceiling_db以下に制限される
        ceiling_linear = 10 ** (-0.1 / 20)
        assert np.max(processed) <= ceiling_linear
        
    def test_custom_settings(self):
        """カスタム設定のテスト"""
        from automix.core.mastering import MasteringSettings
        
        sr = 44100
        mastering = MasteringProcessor(sample_rate=sr)
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr))
        
        # EQのみ適用
        settings = MasteringSettings(
            use_eq=True,
            use_multiband_comp=False,
            use_limiter=False,
            eq_preset="bright"
        )
        
        processed = mastering.process(signal, settings)
        assert len(processed) == len(signal)
        
    def test_extreme_target_lufs(self):
        """極端なターゲットLUFSのテスト"""
        sr = 44100
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)) * 0.5
        
        # 非常に高いLUFS
        mastering_loud = MasteringProcessor(sample_rate=sr, target_lufs=-6)
        loud = mastering_loud.process(signal)
        
        # 非常に低いLUFS
        mastering_quiet = MasteringProcessor(sample_rate=sr, target_lufs=-23)
        quiet = mastering_quiet.process(signal)
        
        # 音量の違いを確認
        assert np.mean(np.abs(loud)) > np.mean(np.abs(quiet))


class TestLimiterProcessorEdgeCases:
    """LimiterProcessorのエッジケーステスト"""
    
    def test_zero_threshold(self):
        """ゼロdB閾値のテスト"""
        limiter = LimiterProcessor(threshold_db=0)
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        limited = limiter.process(signal)
        
        # 1.0を超えない
        assert np.max(np.abs(limited)) <= 1.0
        
    def test_negative_signal(self):
        """負の信号のテスト"""
        limiter = LimiterProcessor(threshold_db=-6)
        signal = -np.ones(44100) * 2.0
        limited = limiter.process(signal)
        
        threshold = 10 ** (-6 / 20)
        assert np.min(limited) >= -threshold * 1.01
        
    def test_instantaneous_attack(self):
        """瞬時アタックのテスト"""
        sr = 44100
        limiter = LimiterProcessor(sample_rate=sr, lookahead_ms=0, release_ms=1)
        
        # ステップ信号
        signal = np.zeros(sr)
        signal[sr//2:] = 2.0
        
        limited = limiter.process(signal)
        # 瞬時に制限される
        assert np.max(limited[sr//2:]) <= limiter.threshold * 1.1
        
    def test_very_long_release(self):
        """非常に長いリリースのテスト"""
        sr = 44100
        limiter = LimiterProcessor(sample_rate=sr, release_ms=5000)
        
        # 短いピーク
        signal = np.zeros(sr * 2)
        signal[1000:1100] = 2.0
        
        limited = limiter.process(signal)
        # リリースが遅いため、ゲインリダクションが長く続く
        assert limited[1500] < signal[1500] * 0.9  # まだ圧縮されている
        
    def test_complex_transients(self):
        """複雑なトランジェントのテスト"""
        sr = 44100
        limiter = LimiterProcessor(sample_rate=sr, lookahead_ms=5)
        
        # ドラムヒットのシミュレーション
        t = np.linspace(0, 0.1, int(sr * 0.1))
        hit = np.exp(-t * 100) * np.sin(2 * np.pi * 60 * t) * 3.0
        signal = np.zeros(sr)
        signal[1000:1000+len(hit)] = hit
        
        limited = limiter.process(signal)
        # トランジェントが保持されつつ制限される
        assert np.max(limited) <= limiter.threshold * 1.05


class TestMultibandCompressorEdgeCases:
    """MultibandCompressorのエッジケーステスト"""
    
    def test_single_band(self):
        """シングルバンドのテスト"""
        compressor = MultibandCompressor(
            crossover_frequencies=[],  # クロスオーバーなし
            thresholds_db=[-20],
            ratios=[3],
            attack_ms=[10],
            release_ms=[100]
        )
        
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        compressed = compressor.process(signal)
        assert len(compressed) == len(signal)
        
    def test_many_bands(self):
        """多数のバンドのテスト"""
        # 5バンドコンプレッサー
        compressor = MultibandCompressor(
            crossover_frequencies=[100, 500, 2000, 8000],
            thresholds_db=[-20] * 5,
            ratios=[2] * 5,
            attack_ms=[10] * 5,
            release_ms=[100] * 5
        )
        
        signal = np.random.normal(0, 0.3, 44100)
        compressed = compressor.process(signal)
        assert not np.any(np.isnan(compressed))
        
    def test_extreme_crossover_frequencies(self):
        """極端なクロスオーバー周波数のテスト"""
        sr = 44100
        # 非常に低い/高いクロスオーバー
        compressor = MultibandCompressor(
            sample_rate=sr,
            crossover_frequencies=[20, 20000],
            thresholds_db=[-15, -15, -15],
            ratios=[3, 3, 3]
        )
        
        signal = np.random.normal(0, 0.3, sr)
        compressed = compressor.process(signal)
        assert len(compressed) == len(signal)
        
    def test_unity_ratio(self):
        """圧縮比1:1のテスト"""
        compressor = MultibandCompressor(
            crossover_frequencies=[1000],
            thresholds_db=[-20, -20],
            ratios=[1, 1],  # 圧縮なし
            attack_ms=[10, 10],
            release_ms=[100, 100]
        )
        
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        compressed = compressor.process(signal)
        # ほぼ変化なし（フィルタリングの影響のみ）
        assert np.allclose(compressed, signal, rtol=0.1)
        
    def test_extreme_attack_release(self):
        """極端なアタック/リリースのテスト"""
        compressor = MultibandCompressor(
            crossover_frequencies=[1000],
            thresholds_db=[-20, -20],
            ratios=[4, 4],
            attack_ms=[0.1, 0.1],  # 非常に速い
            release_ms=[10000, 10000]  # 非常に遅い
        )
        
        signal = np.ones(44100) * 0.8
        # 瞬間的なピーク
        signal[1000:1010] = 1.5
        
        compressed = compressor.process(signal)
        assert np.max(compressed) < np.max(signal)


class TestLoudnessNormalizerEdgeCases:
    """LoudnessNormalizerのエッジケーステスト"""
    
    def test_already_normalized(self):
        """既に正規化済みの信号のテスト"""
        normalizer = LoudnessNormalizer(target_lufs=-14)
        
        # 既に-14 LUFS相当の信号（モック）
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 3, 44100 * 3)) * 0.5
        
        with patch.object(normalizer, 'measure_integrated_lufs', return_value=-14.0):
            normalized = normalizer.normalize(signal)
            # ほぼ変化なし
            assert np.allclose(normalized, signal, rtol=0.01)
            
    def test_very_quiet_signal(self):
        """非常に静かな信号のテスト"""
        normalizer = LoudnessNormalizer(target_lufs=-14, true_peak_db=-1)
        
        # 非常に静かな信号
        quiet_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)) * 0.001
        
        normalized = normalizer.normalize(quiet_signal)
        # 大幅に増幅される
        assert np.max(np.abs(normalized)) > np.max(np.abs(quiet_signal)) * 10
        # しかしtrue peakは超えない
        assert np.max(np.abs(normalized)) <= 10 ** (-1 / 20) * 1.01
        
    def test_invalid_preset(self):
        """無効なプリセットのテスト"""
        with pytest.raises(ValueError):
            LoudnessNormalizer.from_preset("invalid_platform")
            
    def test_k_weighting_filter(self):
        """K-weightingフィルタのテスト"""
        sr = 44100
        normalizer = LoudnessNormalizer(sample_rate=sr)
        
        # 異なる周波数の純音
        t = np.linspace(0, 1, sr)
        low_freq = np.sin(2 * np.pi * 100 * t)
        high_freq = np.sin(2 * np.pi * 3000 * t)
        
        # 同じ振幅でもLUFSは異なる
        lufs_low = normalizer.measure_integrated_lufs(low_freq)
        lufs_high = normalizer.measure_integrated_lufs(high_freq)
        
        # K-weightingにより高域の方が高いLUFS
        assert lufs_high > lufs_low
        
    def test_short_signal(self):
        """短い信号のテスト"""
        normalizer = LoudnessNormalizer()
        
        # 0.1秒の短い信号
        short_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410))
        
        lufs = normalizer.measure_integrated_lufs(short_signal)
        # 有効な値が返される
        assert isinstance(lufs, float)
        assert lufs < 0


class TestFinalEQEdgeCases:
    """FinalEQのエッジケーステスト"""
    
    def test_no_bands(self):
        """バンドなしのテスト"""
        eq = FinalEQ()
        signal = np.random.normal(0, 0.3, 44100)
        
        # バンドを追加せずに処理
        processed = eq.process(signal)
        assert np.array_equal(processed, signal)
        
    def test_extreme_gain(self):
        """極端なゲインのテスト"""
        eq = FinalEQ()
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        
        # +24dBブースト
        eq.add_bell(440, 24, 1)
        processed = eq.process(signal)
        
        # 信号が大幅に増幅される
        assert np.max(np.abs(processed)) > np.max(np.abs(signal)) * 10
        # しかし無限大にはならない
        assert not np.any(np.isinf(processed))
        
    def test_narrow_q(self):
        """狭いQのテスト"""
        sr = 44100
        eq = FinalEQ(sample_rate=sr)
        
        # 複数の周波数を含む信号
        t = np.linspace(0, 1, sr)
        signal = np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 445 * t)
        
        # 非常に狭いQ（440Hzのみカット）
        eq.add_bell(440, -12, 50)
        processed = eq.process(signal)
        
        # 440Hzは減衰、445Hzは維持
        fft_original = np.abs(np.fft.rfft(signal))
        fft_processed = np.abs(np.fft.rfft(processed))
        freqs = np.fft.rfftfreq(len(signal), 1/sr)
        
        idx_440 = np.argmin(np.abs(freqs - 440))
        idx_445 = np.argmin(np.abs(freqs - 445))
        
        assert fft_processed[idx_440] < fft_original[idx_440] * 0.5
        assert fft_processed[idx_445] > fft_original[idx_445] * 0.8
        
    def test_multiple_overlapping_bands(self):
        """重複する複数バンドのテスト"""
        eq = FinalEQ()
        signal = np.random.normal(0, 0.3, 44100)
        
        # 同じ周波数に複数のEQ
        eq.add_bell(1000, 6, 1)
        eq.add_bell(1000, -3, 2)
        eq.add_bell(1000, 2, 0.5)
        
        processed = eq.process(signal)
        # 処理が正常に完了
        assert len(processed) == len(signal)
        assert not np.any(np.isnan(processed))
        
    def test_invalid_preset_eq(self):
        """無効なEQプリセットのテスト"""
        with pytest.raises(ValueError):
            FinalEQ.from_preset("invalid_preset")
            
    def test_shelf_edge_cases(self):
        """シェルフEQのエッジケーステスト"""
        sr = 44100
        eq = FinalEQ(sample_rate=sr)
        signal = np.random.normal(0, 0.3, sr)
        
        # 非常に低い/高い周波数でのシェルフ
        eq.add_shelf(10, 6, "low_shelf")  # 10Hz
        eq.add_shelf(20000, 6, "high_shelf")  # 20kHz
        
        processed = eq.process(signal)
        assert not np.any(np.isnan(processed))
        
    def test_linear_vs_minimum_phase(self):
        """リニアフェーズ vs ミニマムフェーズのテスト"""
        sr = 44100
        signal = np.random.normal(0, 0.3, sr)
        
        eq_linear = FinalEQ(sample_rate=sr, linear_phase=True)
        eq_minimum = FinalEQ(sample_rate=sr, linear_phase=False)
        
        # 同じEQ設定
        for eq in [eq_linear, eq_minimum]:
            eq.add_bell(1000, 6, 1)
            eq.add_shelf(100, 3, "low_shelf")
            
        linear_result = eq_linear.process(signal)
        minimum_result = eq_minimum.process(signal)
        
        # 両方とも有効な出力
        assert len(linear_result) == len(signal)
        assert len(minimum_result) == len(signal)
        
        # 周波数応答は似ているが位相特性が異なる
        fft_linear = np.abs(np.fft.rfft(linear_result))
        fft_minimum = np.abs(np.fft.rfft(minimum_result))
        
        correlation = np.corrcoef(fft_linear, fft_minimum)[0, 1]
        assert correlation > 0.9  # 周波数応答は似ている


class TestIntegrationMastering:
    """マスタリングの統合テスト"""
    
    def test_full_mastering_pipeline(self):
        """完全なマスタリングパイプラインのテスト"""
        sr = 44100
        duration = 10
        
        # リアルな音楽信号のシミュレーション
        t = np.linspace(0, duration, sr * duration)
        
        # ドラム（キック）
        kick_pattern = np.zeros_like(t)
        for i in range(0, len(t), sr // 2):  # 120 BPM
            kick_pattern[i:i+1000] = np.exp(-np.linspace(0, 1, 1000) * 10) * np.sin(2 * np.pi * 60 * np.linspace(0, 1, 1000))
            
        # ベース
        bass = 0.3 * np.sin(2 * np.pi * 80 * t)
        
        # リード
        lead = 0.2 * np.sin(2 * np.pi * 440 * t) * (1 + 0.3 * np.sin(2 * np.pi * 5 * t))  # ビブラート
        
        # ミックス
        mix = kick_pattern + bass + lead
        
        # ダイナミクスの変化
        mix[:sr*2] *= 0.5  # イントロは静か
        mix[sr*8:] *= 1.2  # 最後は盛り上がる
        
        # マスタリング
        mastering = MasteringProcessor(sample_rate=sr, target_lufs=-14, ceiling_db=-0.3)
        mastered = mastering.process(mix)
        
        # 検証
        assert len(mastered) == len(mix)
        assert np.max(np.abs(mastered)) <= 10 ** (-0.3 / 20) * 1.01
        
        # 周波数バランスが改善
        fft_mix = np.abs(np.fft.rfft(mix))
        fft_mastered = np.abs(np.fft.rfft(mastered))
        
        # 低域、中域、高域のバランスを確認
        low_end = len(fft_mix) // 10
        mid_end = len(fft_mix) // 2
        
        low_ratio_mix = np.mean(fft_mix[:low_end]) / np.mean(fft_mix[low_end:mid_end])
        low_ratio_mastered = np.mean(fft_mastered[:low_end]) / np.mean(fft_mastered[low_end:mid_end])
        
        # マスタリング後の方がバランスが良い（極端な比率が改善）
        assert abs(low_ratio_mastered - 1) < abs(low_ratio_mix - 1)
        
    def test_genre_specific_mastering(self):
        """ジャンル別マスタリングのテスト"""
        from automix.core.mastering import MasteringSettings
        
        sr = 44100
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 3, sr * 3))
        
        # EDM向け設定（ラウド、明るい）
        edm_settings = MasteringSettings(
            target_lufs=-9,
            ceiling_db=-0.1,
            use_multiband_comp=True,
            eq_preset="bright"
        )
        
        # ジャズ向け設定（ダイナミック、暖かい）
        jazz_settings = MasteringSettings(
            target_lufs=-16,
            ceiling_db=-1.0,
            use_multiband_comp=False,
            eq_preset="warm"
        )
        
        mastering = MasteringProcessor(sample_rate=sr)
        
        edm_master = mastering.process(signal, edm_settings)
        jazz_master = mastering.process(signal, jazz_settings)
        
        # EDMの方がラウド
        assert np.mean(np.abs(edm_master)) > np.mean(np.abs(jazz_master))
