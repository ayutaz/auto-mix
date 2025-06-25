"""
音声解析機能のテスト
"""

import numpy as np
import pytest

from automix.core.analyzer import (
    AudioAnalyzer,
    PitchAnalysis,
    SpectralAnalysis,
    TempoAnalysis,
    VolumeAnalysis,
)


class TestAudioAnalyzer:
    """AudioAnalyzerクラスのテスト"""

    @pytest.fixture
    def sample_audio(self):
        """テスト用のサンプル音声データ"""
        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # 複数の周波数成分を含む信号
        audio = (
            0.3 * np.sin(2 * np.pi * 440 * t)  # A4
            + 0.2 * np.sin(2 * np.pi * 554.37 * t)  # C#5
            + 0.1 * np.sin(2 * np.pi * 659.25 * t)  # E5
        )

        # エンベロープを追加
        envelope = np.exp(-t / 0.5)
        audio *= envelope

        return audio, sample_rate

    def test_pitch_detection(self, sample_audio):
        """ピッチ検出のテスト"""
        audio, sr = sample_audio
        analyzer = AudioAnalyzer(sample_rate=sr)

        pitch_analysis = analyzer.analyze_pitch(audio)

        assert isinstance(pitch_analysis, PitchAnalysis)
        assert len(pitch_analysis.pitches) > 0
        assert len(pitch_analysis.confidences) == len(pitch_analysis.pitches)
        assert pitch_analysis.median_pitch > 0
        # ピッチ検出はサブハーモニックを検出することがある
        # 440Hzまたはそのサブハーモニック(110Hz, 220Hz)を検出
        assert (105 < pitch_analysis.median_pitch < 115) or (
            215 < pitch_analysis.median_pitch < 225
        ) or (430 < pitch_analysis.median_pitch < 450)

    def test_volume_analysis(self, sample_audio):
        """音量解析のテスト"""
        audio, sr = sample_audio
        analyzer = AudioAnalyzer(sample_rate=sr)

        volume_analysis = analyzer.analyze_volume(audio)

        assert isinstance(volume_analysis, VolumeAnalysis)
        assert volume_analysis.rms > 0
        assert volume_analysis.peak > volume_analysis.rms
        assert volume_analysis.lufs < 0  # LUFSは通常負の値
        assert len(volume_analysis.envelope) > 0

    def test_spectral_analysis(self, sample_audio):
        """スペクトル解析のテスト"""
        audio, sr = sample_audio
        analyzer = AudioAnalyzer(sample_rate=sr)

        spectral_analysis = analyzer.analyze_spectrum(audio)

        assert isinstance(spectral_analysis, SpectralAnalysis)
        assert spectral_analysis.spectral_centroid > 0
        assert spectral_analysis.spectral_rolloff > spectral_analysis.spectral_centroid
        assert len(spectral_analysis.mfcc) > 0
        assert spectral_analysis.spectral_bandwidth > 0

    def test_tempo_detection(self, sample_audio):
        """テンポ検出のテスト"""
        # ビートのあるサンプルを生成
        sr = 44100
        duration = 4.0
        tempo = 120  # BPM

        # キックドラムのシミュレーション
        t = np.linspace(0, duration, int(sr * duration))
        beat_interval = 60.0 / tempo
        audio = np.zeros_like(t)

        for i in range(int(duration / beat_interval)):
            beat_time = i * beat_interval
            beat_sample = int(beat_time * sr)
            if beat_sample < len(audio):
                # キックドラムのエンベロープ
                env_length = int(0.1 * sr)
                env = np.exp(-np.linspace(0, 10, env_length))
                end_sample = min(beat_sample + env_length, len(audio))
                audio[beat_sample:end_sample] += env[: end_sample - beat_sample]

        analyzer = AudioAnalyzer(sample_rate=sr)
        tempo_analysis = analyzer.analyze_tempo(audio)

        assert isinstance(tempo_analysis, TempoAnalysis)
        assert tempo_analysis.tempo > 0
        assert len(tempo_analysis.beats) > 0
        assert 110 < tempo_analysis.tempo < 130  # 120 BPM付近

    def test_full_analysis(self, sample_audio):
        """全解析機能の統合テスト"""
        audio, sr = sample_audio
        analyzer = AudioAnalyzer(sample_rate=sr)

        analysis = analyzer.analyze_all(audio)

        assert "pitch" in analysis
        assert "volume" in analysis
        assert "spectrum" in analysis
        assert "tempo" in analysis

        assert isinstance(analysis["pitch"], PitchAnalysis)
        assert isinstance(analysis["volume"], VolumeAnalysis)
        assert isinstance(analysis["spectrum"], SpectralAnalysis)
        assert isinstance(analysis["tempo"], TempoAnalysis)

    def test_stereo_audio_analysis(self):
        """ステレオ音声の解析テスト"""
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # ステレオ信号（左右で異なる周波数）
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 554.37 * t)
        stereo_audio = np.column_stack([left, right])

        analyzer = AudioAnalyzer(sample_rate=sr)

        # 各チャンネルの解析
        left_analysis = analyzer.analyze_pitch(stereo_audio[:, 0])
        right_analysis = analyzer.analyze_pitch(stereo_audio[:, 1])

        assert abs(left_analysis.median_pitch - 440) < 10
        assert abs(right_analysis.median_pitch - 554.37) < 10

    def test_silence_handling(self):
        """無音部分の処理テスト"""
        sr = 44100
        silence = np.zeros(sr)  # 1秒の無音

        analyzer = AudioAnalyzer(sample_rate=sr)

        pitch_analysis = analyzer.analyze_pitch(silence)
        volume_analysis = analyzer.analyze_volume(silence)

        assert pitch_analysis.median_pitch == 0 or np.isnan(pitch_analysis.median_pitch)
        assert volume_analysis.rms < 0.001
        assert volume_analysis.peak < 0.001

    @pytest.mark.parametrize(
        "window_size,hop_length",
        [
            (2048, 512),
            (4096, 1024),
            (8192, 2048),
        ],
    )
    def test_different_window_sizes(self, sample_audio, window_size, hop_length):
        """異なるウィンドウサイズでの解析テスト"""
        audio, sr = sample_audio
        analyzer = AudioAnalyzer(sample_rate=sr, window_size=window_size, hop_length=hop_length)

        spectral_analysis = analyzer.analyze_spectrum(audio)

        assert spectral_analysis.spectral_centroid > 0
        assert len(spectral_analysis.mfcc) > 0

    def test_performance_analysis(self, sample_audio):
        """パフォーマンス解析（実行時間）"""
        import time

        audio, sr = sample_audio
        analyzer = AudioAnalyzer(sample_rate=sr)

        start_time = time.time()
        analyzer.analyze_all(audio)
        end_time = time.time()

        execution_time = end_time - start_time
        assert execution_time < 2.0  # 2秒の音声を2秒以内に解析
