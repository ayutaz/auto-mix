"""
エフェクト処理のテスト
"""

import numpy as np
import pytest

from automix.core.effects import (
    ChorusProcessor,
    DeesserProcessor,
    DelayProcessor,
    ReverbProcessor,
    StereoProcessor,
)


class TestReverbProcessor:
    """ReverbProcessorのテスト"""

    @pytest.fixture
    def impulse_signal(self):
        """インパルス信号（リバーブテスト用）"""
        signal = np.zeros(44100)
        signal[1000] = 1.0  # インパルス
        return signal, 44100

    def test_basic_reverb(self, impulse_signal):
        """基本的なリバーブ処理のテスト"""
        signal, sr = impulse_signal
        reverb = ReverbProcessor(sample_rate=sr, room_size=0.5, damping=0.5, wet_dry_mix=0.3)

        processed = reverb.process(signal)

        # リバーブによる残響があることを確認
        assert len(processed) == len(signal)
        assert np.sum(np.abs(processed[2000:10000])) > 0.1  # 残響成分
        assert np.max(np.abs(processed)) <= 1.0

    def test_reverb_presets(self):
        """リバーブプリセットのテスト"""
        signal = np.random.normal(0, 0.3, 44100)
        sr = 44100

        presets = ["hall", "room", "plate", "spring"]
        for preset in presets:
            reverb = ReverbProcessor.from_preset(preset, sample_rate=sr)
            processed = reverb.process(signal)

            assert isinstance(processed, np.ndarray)
            assert len(processed) == len(signal)

    def test_wet_dry_mix(self):
        """Wet/Dryミックスのテスト"""
        signal = np.random.normal(0, 0.3, 44100)
        sr = 44100

        # 100% dry
        reverb_dry = ReverbProcessor(sample_rate=sr, wet_dry_mix=0.0)
        processed_dry = reverb_dry.process(signal)
        assert np.allclose(processed_dry, signal, rtol=0.01)

        # 100% wet
        reverb_wet = ReverbProcessor(sample_rate=sr, wet_dry_mix=1.0)
        processed_wet = reverb_wet.process(signal)
        assert not np.allclose(processed_wet, signal, rtol=0.1)


class TestDelayProcessor:
    """DelayProcessorのテスト"""

    def test_simple_delay(self):
        """シンプルなディレイのテスト"""
        sr = 44100
        delay_ms = 100
        delay = DelayProcessor(sample_rate=sr, delay_time_ms=delay_ms, feedback=0.5, mix=0.5)

        # インパルス信号
        signal = np.zeros(sr)
        signal[1000] = 1.0

        processed = delay.process(signal)

        # ディレイサンプル数
        delay_samples = int(delay_ms * sr / 1000)

        # 元の位置とディレイ位置にピークがあることを確認
        assert processed[1000] > 0.5  # 元の信号
        assert processed[1000 + delay_samples] > 0.2  # ディレイ
        assert processed[1000 + 2 * delay_samples] > 0.1  # フィードバック

    def test_ping_pong_delay(self):
        """ピンポンディレイのテスト（ステレオ）"""
        sr = 44100
        delay = DelayProcessor(
            sample_rate=sr, delay_time_ms=200, feedback=0.6, mix=0.4, ping_pong=True
        )

        # モノラル信号
        signal = np.zeros(sr)
        signal[1000] = 1.0

        # ステレオ出力
        processed = delay.process_stereo(signal)

        assert processed.shape[1] == 2  # ステレオ
        # 左右のチャンネルでディレイが交互に現れる
        assert not np.array_equal(processed[:, 0], processed[:, 1])

    def test_tempo_sync_delay(self):
        """テンポ同期ディレイのテスト"""
        sr = 44100
        bpm = 120

        # 1/4音符のディレイ
        delay = DelayProcessor.from_tempo(
            sample_rate=sr, bpm=bpm, note_division="1/4", feedback=0.5, mix=0.5
        )

        expected_delay_ms = 60000 / bpm  # 500ms
        assert abs(delay.delay_time_ms - expected_delay_ms) < 1


class TestChorusProcessor:
    """ChorusProcessorのテスト"""

    def test_basic_chorus(self):
        """基本的なコーラス効果のテスト"""
        sr = 44100
        chorus = ChorusProcessor(sample_rate=sr, rate_hz=1.5, depth=0.3, mix=0.5)

        # 単純な正弦波
        t = np.linspace(0, 1, sr)
        signal = np.sin(2 * np.pi * 440 * t)

        processed = chorus.process(signal)

        # コーラス効果により信号が変化
        assert not np.allclose(processed, signal)
        assert len(processed) == len(signal)

        # 周波数スペクトルが広がっていることを確認
        fft_original = np.abs(np.fft.rfft(signal))
        fft_processed = np.abs(np.fft.rfft(processed))

        # 440Hz周辺のピークが広がっている
        peak_idx = np.argmax(fft_original)
        spread = np.sum(fft_processed[peak_idx - 10 : peak_idx + 10])
        original_spread = np.sum(fft_original[peak_idx - 10 : peak_idx + 10])

        assert spread > original_spread * 0.9


class TestDeesserProcessor:
    """DeesserProcessorのテスト"""

    def test_deessing(self):
        """歯擦音除去のテスト"""
        sr = 44100
        deesser = DeesserProcessor(sample_rate=sr, frequency=6000, threshold_db=-30, ratio=4)

        # 歯擦音を含む信号（高周波成分）
        t = np.linspace(0, 0.1, int(sr * 0.1))
        sibilant = np.sin(2 * np.pi * 7000 * t)  # 7kHzの歯擦音
        voice = np.sin(2 * np.pi * 200 * t)  # 200Hzの音声
        signal = voice + sibilant

        processed = deesser.process(signal)

        # 高周波成分が減衰していることを確認
        fft_original = np.abs(np.fft.rfft(signal))
        fft_processed = np.abs(np.fft.rfft(processed))
        freqs = np.fft.rfftfreq(len(signal), 1 / sr)

        idx_7k = np.argmin(np.abs(freqs - 7000))
        idx_200 = np.argmin(np.abs(freqs - 200))

        # 7kHzは減衰、200Hzは維持
        assert fft_processed[idx_7k] < fft_original[idx_7k] * 0.7
        assert fft_processed[idx_200] > fft_original[idx_200] * 0.9


class TestStereoProcessor:
    """StereoProcessorのテスト"""

    def test_stereo_widening(self):
        """ステレオ拡張のテスト"""
        sr = 44100
        processor = StereoProcessor(sample_rate=sr)

        # モノラルに近いステレオ信号
        t = np.linspace(0, 1, sr)
        left = np.sin(2 * np.pi * 440 * t)
        right = 0.9 * left + 0.1 * np.sin(2 * np.pi * 445 * t)
        stereo = np.column_stack([left, right])

        # ステレオ幅を拡張
        widened = processor.widen(stereo, width=1.5)

        # 左右の差が大きくなっていることを確認
        original_diff = np.mean(np.abs(stereo[:, 0] - stereo[:, 1]))
        widened_diff = np.mean(np.abs(widened[:, 0] - widened[:, 1]))

        assert widened_diff > original_diff

    def test_mid_side_processing(self):
        """Mid/Side処理のテスト"""
        sr = 44100
        processor = StereoProcessor(sample_rate=sr)

        # ステレオ信号
        t = np.linspace(0, 0.1, int(sr * 0.1))
        left = np.sin(2 * np.pi * 440 * t)
        right = np.sin(2 * np.pi * 554 * t)
        stereo = np.column_stack([left, right])

        # Mid/Sideに変換して戻す
        mid, side = processor.to_mid_side(stereo)
        back_stereo = processor.from_mid_side(mid, side)

        # 元に戻ることを確認
        assert np.allclose(stereo, back_stereo, rtol=0.001)

    def test_auto_pan(self):
        """オートパンのテスト"""
        sr = 44100
        processor = StereoProcessor(sample_rate=sr)

        # モノラル信号
        signal = np.ones(sr) * 0.5

        # オートパン適用
        panned = processor.auto_pan(signal, rate_hz=2.0, depth=0.8)

        assert panned.shape == (sr, 2)  # ステレオ出力

        # 左右のチャンネルが逆位相で動いている
        correlation = np.corrcoef(panned[:, 0], panned[:, 1])[0, 1]
        assert correlation < -0.5  # 負の相関
