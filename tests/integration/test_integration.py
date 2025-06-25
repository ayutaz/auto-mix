"""
統合テスト - エンドツーエンドのワークフローテスト
"""
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

try:
    import cv2
except ImportError:
    cv2 = None
import soundfile as sf

from automix.core.analyzer import AudioAnalyzer
from automix.core.audio_loader import AudioLoader
from automix.core.effects import ReverbProcessor
from automix.core.mastering import MasteringProcessor
from automix.core.processor import CompressorProcessor, MixProcessor
from automix.video.encoder import VideoEncoder, VideoSettings
from automix.video.visualizer import VisualizerComposite


class TestEndToEndWorkflow:
    """エンドツーエンドワークフローのテスト"""

    @pytest.fixture
    def create_test_audio(self):
        """テスト用音声ファイルを生成"""

        def _create(duration=5.0, filename="test.wav"):
            sr = 44100
            t = np.linspace(0, duration, int(sr * duration))

            # 音楽的な音声を生成
            # ボーカル風（中音域）
            vocal = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4
            vocal += 0.2 * np.sin(2 * np.pi * 554.37 * t)  # C#5
            vocal *= np.exp(-t / 3)  # フェードアウト

            # BGM風（複数の音域）
            bgm = 0.2 * np.sin(2 * np.pi * 110 * t)  # ベース
            bgm += 0.15 * np.sin(2 * np.pi * 220 * t)  # 低音
            bgm += 0.1 * np.sin(2 * np.pi * 880 * t)  # 高音

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, vocal if "vocal" in filename else bgm, sr)
                return Path(f.name), sr

        return _create

    @pytest.mark.slow
    def test_complete_audio_processing_pipeline(self, create_test_audio):
        """完全な音声処理パイプラインのテスト"""
        # テスト音声を作成
        vocal_path, sr = create_test_audio(duration=3.0, filename="vocal.wav")
        bgm_path, _ = create_test_audio(duration=3.0, filename="bgm.wav")

        try:
            # 1. 音声読み込み
            loader = AudioLoader(target_sample_rate=sr, normalize=True)
            vocal_audio = loader.load(vocal_path)
            bgm_audio = loader.load(bgm_path)

            assert vocal_audio.sample_rate == sr
            assert bgm_audio.sample_rate == sr

            # 2. 音声解析
            analyzer = AudioAnalyzer(sample_rate=sr)
            vocal_analysis = analyzer.analyze_all(vocal_audio.data)
            bgm_analysis = analyzer.analyze_all(bgm_audio.data)

            assert "pitch" in vocal_analysis
            assert "volume" in vocal_analysis

            # 3. ミックス処理
            processor = MixProcessor(sample_rate=sr)
            mixed = processor.mix(vocal_audio.data, bgm_audio.data, vocal_gain_db=0, bgm_gain_db=-3)

            assert len(mixed) == len(vocal_audio.data)
            assert np.max(np.abs(mixed)) <= 1.0

            # 4. エフェクト適用
            reverb = ReverbProcessor(sample_rate=sr, room_size=0.3, wet_dry_mix=0.2)
            mixed_with_reverb = reverb.process(mixed)

            compressor = CompressorProcessor(sample_rate=sr, threshold_db=-20, ratio=4)
            compressed = compressor.compress(mixed_with_reverb)

            # 5. マスタリング
            mastering = MasteringProcessor(sample_rate=sr, target_lufs=-14)
            mastered = mastering.process(compressed)

            # 最終確認
            assert isinstance(mastered, np.ndarray)
            assert len(mastered) == len(vocal_audio.data)
            assert np.max(np.abs(mastered)) <= 1.0

            # 出力ファイルに保存
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, mastered, sr)
                output_path = Path(f.name)
                assert output_path.exists()
                output_path.unlink()

        finally:
            # クリーンアップ
            vocal_path.unlink(missing_ok=True)
            bgm_path.unlink(missing_ok=True)

    @pytest.mark.slow
    @pytest.mark.skipif(cv2 is None, reason="OpenCV (cv2) not installed")
    @patch("automix.video.encoder.mpe")
    def test_complete_video_generation_pipeline(self, mock_mpe, create_test_audio):
        """完全な動画生成パイプラインのテスト"""
        # テスト音声を作成
        audio_path, sr = create_test_audio(duration=2.0)

        try:
            # 音声読み込み
            loader = AudioLoader()
            audio_file = loader.load(audio_path)

            # ビジュアライザー設定
            visualizer = VisualizerComposite(width=1280, height=720, sample_rate=sr)

            # 動画設定
            video_settings = VideoSettings(width=1280, height=720, fps=30)

            encoder = VideoEncoder(video_settings)

            # フレーム生成（簡易版）
            frames = []
            frame_duration = 1 / video_settings.fps
            samples_per_frame = int(sr * frame_duration)

            for i in range(0, len(audio_file.data), samples_per_frame):
                audio_chunk = audio_file.data[i : i + samples_per_frame]
                if len(audio_chunk) < samples_per_frame:
                    break

                frame = visualizer.render_composite_frame(audio_chunk)
                frames.append(frame)

            assert len(frames) > 0
            assert frames[0].shape == (720, 1280, 3)

            # エンコード（モック）
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                output_path = Path(f.name)

            mock_mpe.VideoFileClip.return_value.write_videofile = lambda *args, **kwargs: None
            encoder.encode(frames, audio_file.data, output_path, sample_rate=sr)

            output_path.unlink(missing_ok=True)

        finally:
            audio_path.unlink(missing_ok=True)

    @pytest.mark.slow
    def test_performance_benchmark(self, create_test_audio):
        """パフォーマンスベンチマークテスト"""
        # 各処理ステップの実行時間を計測
        durations = [1.0, 5.0, 10.0]  # 秒
        results = {}

        for duration in durations:
            vocal_path, sr = create_test_audio(duration=duration, filename="vocal.wav")
            bgm_path, _ = create_test_audio(duration=duration, filename="bgm.wav")

            try:
                # 読み込み時間
                start = time.time()
                loader = AudioLoader()
                vocal = loader.load(vocal_path)
                bgm = loader.load(bgm_path)
                load_time = time.time() - start

                # 解析時間
                start = time.time()
                analyzer = AudioAnalyzer(sample_rate=sr)
                analyzer.analyze_all(vocal.data)
                analyze_time = time.time() - start

                # ミックス時間
                start = time.time()
                processor = MixProcessor(sample_rate=sr)
                mixed = processor.mix(vocal.data, bgm.data)
                mix_time = time.time() - start

                # マスタリング時間
                start = time.time()
                mastering = MasteringProcessor(sample_rate=sr)
                mastered = mastering.process(mixed)
                master_time = time.time() - start

                results[duration] = {
                    "load": load_time,
                    "analyze": analyze_time,
                    "mix": mix_time,
                    "master": master_time,
                    "total": load_time + analyze_time + mix_time + master_time,
                }

                # パフォーマンス基準: リアルタイムの2倍以内
                assert results[duration]["total"] < duration * 2

            finally:
                vocal_path.unlink(missing_ok=True)
                bgm_path.unlink(missing_ok=True)

        # 結果を出力（デバッグ用）
        for duration, times in results.items():
            print(f"\n{duration}秒の音声:")
            for step, time_val in times.items():
                print(f"  {step}: {time_val:.3f}秒")

    def test_error_handling_pipeline(self, create_test_audio):
        """エラーハンドリングのテスト"""
        # 不正な入力でのエラーハンドリング

        # 1. 存在しないファイル
        loader = AudioLoader()
        with pytest.raises(Exception):
            loader.load(Path("nonexistent_file.wav"))

        # 2. サンプルレート不一致
        vocal_path, _ = create_test_audio(duration=1.0)

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                # 異なるサンプルレート
                sf.write(f.name, np.zeros(22050), 22050)
                bgm_path = Path(f.name)

            loader1 = AudioLoader(target_sample_rate=44100)
            loader2 = AudioLoader(target_sample_rate=44100)

            vocal = loader1.load(vocal_path)
            bgm = loader2.load(bgm_path)

            # リサンプリングにより一致するはず
            assert vocal.sample_rate == bgm.sample_rate

            bgm_path.unlink(missing_ok=True)
        finally:
            vocal_path.unlink(missing_ok=True)

    def test_memory_efficiency(self, create_test_audio):
        """メモリ効率のテスト"""
        import gc

        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not installed")

        # 初期メモリ使用量
        process = psutil.Process()
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 大きめのファイルを処理
        audio_path, sr = create_test_audio(duration=30.0)

        try:
            loader = AudioLoader()
            audio = loader.load(audio_path)

            # 処理中のメモリ使用量
            processor = MixProcessor(sample_rate=sr)
            mixed = processor.mix(audio.data, audio.data)

            gc.collect()
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB

            # メモリ使用量の増加が妥当な範囲内
            memory_increase = peak_memory - initial_memory
            assert memory_increase < 500  # 500MB以内

            # オブジェクトを削除してメモリが解放されることを確認
            del audio, mixed
            gc.collect()

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            assert final_memory < peak_memory

        finally:
            audio_path.unlink(missing_ok=True)
