"""
動画生成機能のテスト
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# OpenCVのインポート
try:
    import cv2

    HAS_CV2 = True
    print(f"DEBUG: cv2 imported successfully in test_video.py, version: {cv2.__version__}")
except ImportError as e:
    print(f"DEBUG: Failed to import cv2 in test_video.py: {e}")
    cv2 = None
    HAS_CV2 = False

from automix.video.encoder import (  # noqa: E402
    CodecOptions,
    StreamingEncoder,
    ThumbnailGenerator,
    VideoEncoder,
    VideoSettings,
)
from automix.video.text_overlay import (  # noqa: E402
    LyricsRenderer,
    MetadataDisplay,
    TextOverlay,
    TextStyle,
)
from automix.video.visualizer import (  # noqa: E402
    ParticleVisualizer,
    SpectrumVisualizer,
    VisualizerComposite,
    WaveformVisualizer,
)


class TestWaveformVisualizer:
    """WaveformVisualizerのテスト"""

    @pytest.fixture
    def sample_waveform(self):
        """サンプル波形データ"""
        sr = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # 変化のある波形
        waveform = np.sin(2 * np.pi * 5 * t) * np.sin(2 * np.pi * 0.5 * t)
        return waveform, sr

    def test_waveform_rendering(self, sample_waveform):
        """波形レンダリングのテスト"""
        waveform, sr = sample_waveform

        visualizer = WaveformVisualizer(
            width=1920, height=1080, sample_rate=sr, color=(0, 255, 0), style="filled"
        )

        # 1フレーム分のデータ
        frame_duration = 1 / 30  # 30fps
        samples_per_frame = int(sr * frame_duration)
        frame_data = waveform[:samples_per_frame]

        frame = visualizer.render_frame(frame_data)

        assert frame.shape == (1080, 1920, 3)
        assert frame.dtype == np.uint8
        # デバッグ情報
        print(f"Frame data shape: {frame_data.shape}")
        print(f"Frame data range: [{np.min(frame_data)}, {np.max(frame_data)}]")
        print(f"Frame max value: {np.max(frame)}")
        print(f"cv2 available: {cv2 is not None}")
        assert np.any(frame > 0)  # 何か描画されている

    def test_waveform_styles(self, sample_waveform):
        """異なる波形スタイルのテスト"""
        waveform, sr = sample_waveform

        styles = ["line", "filled", "mirror", "circular"]

        for style in styles:
            visualizer = WaveformVisualizer(width=800, height=600, sample_rate=sr, style=style)

            frame = visualizer.render_frame(waveform[:1000])
            assert frame.shape == (600, 800, 3)

    def test_stereo_waveform(self):
        """ステレオ波形のテスト"""
        sr = 44100
        duration = 0.1
        t = np.linspace(0, duration, int(sr * duration))

        # ステレオ信号
        left = np.sin(2 * np.pi * 440 * t)
        right = np.sin(2 * np.pi * 554 * t)
        stereo = np.column_stack([left, right])

        visualizer = WaveformVisualizer(width=1280, height=720, sample_rate=sr, stereo=True)

        frame = visualizer.render_frame(stereo)

        # 上下に分かれて表示される
        assert frame.shape == (720, 1280, 3)
        # 上半分と下半分で異なるパターン
        top_half = frame[:360]
        bottom_half = frame[360:]
        assert not np.array_equal(top_half, bottom_half)


class TestSpectrumVisualizer:
    """SpectrumVisualizerのテスト"""

    def test_spectrum_rendering(self):
        """スペクトラムレンダリングのテスト"""
        sr = 44100

        # 複数の周波数成分を持つ信号
        t = np.linspace(0, 0.1, int(sr * 0.1))
        signal = (
            np.sin(2 * np.pi * 440 * t)
            + 0.5 * np.sin(2 * np.pi * 880 * t)
            + 0.3 * np.sin(2 * np.pi * 1760 * t)
        )

        visualizer = SpectrumVisualizer(
            width=1920,
            height=1080,
            sample_rate=sr,
            fft_size=2048,
            bar_count=64,
            color_scheme="rainbow",
        )

        frame = visualizer.render_frame(signal)

        assert frame.shape == (1080, 1920, 3)
        assert np.any(frame > 0)

    def test_frequency_scales(self):
        """周波数スケールのテスト"""
        sr = 44100
        signal = np.random.normal(0, 0.3, 4096)

        scales = ["linear", "log", "mel"]

        for scale in scales:
            visualizer = SpectrumVisualizer(
                width=1024, height=768, sample_rate=sr, frequency_scale=scale
            )

            frame = visualizer.render_frame(signal)
            assert frame.shape == (768, 1024, 3)

    def test_spectrum_smoothing(self):
        """スペクトラムスムージングのテスト"""
        sr = 44100
        visualizer = SpectrumVisualizer(width=800, height=600, sample_rate=sr, smoothing_factor=0.8)

        # ノイズ信号でテスト
        frame1 = visualizer.render_frame(np.random.normal(0, 0.3, 2048))
        frame2 = visualizer.render_frame(np.random.normal(0, 0.3, 2048))

        # スムージングにより連続性がある
        assert not np.array_equal(frame1, frame2)


class TestParticleVisualizer:
    """ParticleVisualizerのテスト"""

    def test_particle_generation(self):
        """パーティクル生成のテスト"""
        visualizer = ParticleVisualizer(
            width=1280, height=720, max_particles=1000, particle_size=5, audio_reactive=True
        )

        # 音声データ
        audio_level = 0.8
        frequency_data = np.random.rand(32) * 0.5

        frame = visualizer.render_frame(audio_level, frequency_data)

        assert frame.shape == (720, 1280, 3)
        assert np.any(frame > 0)  # パーティクルが描画されている

    def test_particle_physics(self):
        """パーティクル物理演算のテスト"""
        visualizer = ParticleVisualizer(
            width=800, height=600, gravity=(0, 0.5), wind=(0.1, 0), turbulence=0.1
        )

        # 複数フレームでパーティクルの動きを確認
        frames = []
        for i in range(10):
            audio_level = 0.5 + 0.3 * np.sin(i * 0.5)
            frame = visualizer.render_frame(audio_level)
            frames.append(frame)

        # フレーム間で変化がある
        for i in range(1, len(frames)):
            assert not np.array_equal(frames[i - 1], frames[i])


class TestTextOverlay:
    """TextOverlayのテスト"""

    def test_basic_text_rendering(self):
        """基本的なテキストレンダリングのテスト"""
        overlay = TextOverlay(
            width=1920,
            height=1080,
            font_size=48,
            font_color=(255, 255, 255),
            background_color=(0, 0, 0, 128),
        )

        # 背景画像
        background = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # テキストを追加
        frame = overlay.add_text(background, "Test Song Title", position=(100, 100))

        assert frame.shape == background.shape
        assert not np.array_equal(frame, background)

    def test_text_effects(self):
        """テキストエフェクトのテスト"""
        overlay = TextOverlay(width=1280, height=720)

        background = np.zeros((720, 1280, 3), dtype=np.uint8)

        # フェードイン効果
        for alpha in [0.0, 0.5, 1.0]:
            frame = overlay.add_text(background, "Fade Test", position=(100, 100), alpha=alpha)
            # アルファ値によって見え方が変わる
            assert isinstance(frame, np.ndarray)

    def test_multi_line_text(self):
        """複数行テキストのテスト"""
        overlay = TextOverlay(width=800, height=600, line_spacing=1.5)

        background = np.zeros((600, 800, 3), dtype=np.uint8)

        multi_line_text = "Line 1\nLine 2\nLine 3"
        frame = overlay.add_text(background, multi_line_text, position=(50, 50))

        assert frame.shape == background.shape


class TestLyricsRenderer:
    """LyricsRendererのテスト"""

    def test_srt_parsing(self):
        """SRTファイル解析のテスト"""
        srt_content = """1
00:00:00,000 --> 00:00:02,000
First line of lyrics

2
00:00:02,000 --> 00:00:04,000
Second line of lyrics
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(srt_content)
            srt_path = Path(f.name)

        try:
            renderer = LyricsRenderer(srt_file=srt_path, width=1920, height=1080)

            # 時刻0.5秒の歌詞
            lyrics = renderer.get_lyrics_at_time(0.5)
            assert lyrics == "First line of lyrics"

            # 時刻3秒の歌詞
            lyrics = renderer.get_lyrics_at_time(3.0)
            assert lyrics == "Second line of lyrics"
        finally:
            srt_path.unlink()

    def test_karaoke_highlight(self):
        """カラオケ風ハイライトのテスト"""
        renderer = LyricsRenderer(
            width=1280, height=720, karaoke_mode=True, highlight_color=(255, 255, 0)
        )

        background = np.zeros((720, 1280, 3), dtype=np.uint8)

        # 歌詞と進行度
        lyrics = "This is a test line"
        progress = 0.5  # 50%進行

        frame = renderer.render_karaoke(background, lyrics, progress, position=(100, 400))

        assert frame.shape == background.shape
        assert np.any(frame[:, :, 2] > 0)  # 黄色のハイライト


class TestVideoEncoder:
    """VideoEncoderのテスト"""

    def test_video_settings(self):
        """ビデオ設定のテスト"""
        settings = VideoSettings(
            width=1920, height=1080, fps=30, bitrate="8M", codec="h264", preset="medium"
        )

        assert settings.width == 1920
        assert settings.height == 1080
        assert settings.fps == 30

    @pytest.mark.skip(reason="FFmpeg not available in CI environment")
    @patch("automix.video.encoder.moviepy.VideoFileClip")
    def test_encoding_process(self, mock_moviepy):
        """エンコーディングプロセスのテスト"""
        settings = VideoSettings(width=1280, height=720, fps=24)

        encoder = VideoEncoder(settings)

        # テスト用のフレーム
        frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(10)]

        audio = np.random.normal(0, 0.3, 44100)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"

            # モックの設定
            mock_clip = MagicMock()
            mock_moviepy.return_value = mock_clip

            encoder.encode(frames, audio, output_path, sample_rate=44100)

            # エンコード関数が呼ばれたことを確認
            assert mock_moviepy.called

    def test_codec_options(self):
        """コーデックオプションのテスト"""
        codecs = ["h264", "h265", "vp9"]

        for codec in codecs:
            options = CodecOptions.get_options(codec)
            assert isinstance(options, dict)
            assert "codec" in options

    def test_resolution_presets(self):
        """解像度プリセットのテスト"""
        presets = {
            "4k": (3840, 2160),
            "1080p": (1920, 1080),
            "720p": (1280, 720),
            "480p": (854, 480),
        }

        for preset_name, (width, height) in presets.items():
            settings = VideoSettings.from_preset(preset_name)
            assert settings.width == width
            assert settings.height == height


class TestVideoEncoderEdgeCases:
    """VideoEncoderのエッジケーステスト"""

    def test_empty_frames(self):
        """空のフレームリストのテスト"""
        settings = VideoSettings()
        encoder = VideoEncoder(settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"
            audio = np.zeros(44100)

            with pytest.raises(Exception):
                encoder.encode([], audio, output_path)

    def test_single_frame(self):
        """単一フレームのテスト"""
        settings = VideoSettings(width=640, height=480, fps=1)
        encoder = VideoEncoder(settings)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        audio = np.zeros(44100)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"

            with patch("automix.video.encoder.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                encoder.encode([frame], audio, output_path)
                assert mock_run.called

    def test_invalid_codec(self):
        """無効なコーデックのテスト"""
        settings = VideoSettings.from_preset("1080p")
        settings.codec = "invalid_codec"

        encoder = VideoEncoder(settings)
        # デフォルトのh264にフォールバック
        assert encoder.codec_options["codec"] == "libx264"

    def test_extreme_bitrate(self):
        """極端なビットレートのテスト"""
        # 非常に低いビットレート
        settings_low = VideoSettings(bitrate="100k")
        encoder_low = VideoEncoder(settings_low)
        assert encoder_low.settings.bitrate == "100k"

        # 非常に高いビットレート
        settings_high = VideoSettings(bitrate="100M")
        encoder_high = VideoEncoder(settings_high)
        assert encoder_high.settings.bitrate == "100M"

    @patch("automix.video.encoder.cv2.imwrite")
    def test_frame_saving_error(self, mock_imwrite):
        """フレーム保存エラーのテスト"""
        mock_imwrite.return_value = False  # 保存失敗

        settings = VideoSettings()
        encoder = VideoEncoder(settings)

        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        audio = np.zeros(44100)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"

            # フレーム保存に失敗してもエラーハンドリングされる
            with pytest.raises(Exception):
                encoder.encode(frames, audio, output_path)

    def test_audio_normalization(self):
        """音声正規化のテスト"""
        encoder = VideoEncoder(VideoSettings())

        # クリッピングする音声
        audio = np.ones(44100) * 2.0

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "audio.wav"

            with patch("soundfile.write") as mock_write:
                encoder._save_audio(audio, audio_path, 44100)

                # 正規化された音声が保存される
                saved_audio = mock_write.call_args[0][1]
                assert np.max(np.abs(saved_audio)) <= 0.95

    def test_ffmpeg_not_found(self):
        """FFmpegが見つからない場合のテスト"""
        encoder = VideoEncoder(VideoSettings())

        with patch("automix.video.encoder.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            with (
                tempfile.TemporaryDirectory() as tmpdir,
                pytest.raises(RuntimeError, match="FFmpeg not found"),
            ):
                encoder._create_video(
                    "frame_%06d.png", Path(tmpdir) / "audio.wav", Path(tmpdir) / "output.mp4", False
                )

    def test_encoding_failure(self):
        """エンコーディング失敗のテスト"""
        import subprocess

        encoder = VideoEncoder(VideoSettings())

        with patch("automix.video.encoder.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg")

            with (
                tempfile.TemporaryDirectory() as tmpdir,
                pytest.raises(RuntimeError, match="FFmpeg encoding failed"),
            ):
                encoder._create_video(
                    "frame_%06d.png", Path(tmpdir) / "audio.wav", Path(tmpdir) / "output.mp4", False
                )


class TestStreamingEncoder:
    """StreamingEncoderのテスト"""

    def test_streaming_initialization(self):
        """ストリーミング初期化のテスト"""
        settings = VideoSettings(width=1280, height=720, fps=30)
        encoder = StreamingEncoder(settings)

        assert encoder.settings == settings
        assert encoder.process is None

    @patch("subprocess.Popen")
    def test_start_streaming(self, mock_popen):
        """ストリーミング開始のテスト"""
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        encoder = StreamingEncoder(VideoSettings())
        encoder.start_encoding("rtmp://localhost/live/stream")

        assert mock_popen.called
        assert encoder.process == mock_process

        # FFmpegコマンドの確認
        cmd = mock_popen.call_args[0][0]
        assert "ffmpeg" in cmd
        assert "rtmp://localhost/live/stream" in cmd

    def test_write_frame_without_process(self):
        """プロセスなしでのフレーム書き込みテスト"""
        encoder = StreamingEncoder(VideoSettings())
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # プロセスが開始されていない場合は何も起きない
        encoder.write_frame(frame)

    def test_write_audio_conversion(self):
        """音声変換のテスト"""
        encoder = StreamingEncoder(VideoSettings())

        # モックプロセス
        encoder.process = MagicMock()
        encoder.process.stdin = MagicMock()

        # Float32音声
        audio = np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32)
        encoder.write_audio(audio)

        # Int16に変換されて書き込まれる
        encoder.process.stdin.write.assert_called_once()
        written_data = encoder.process.stdin.write.call_args[0][0]

        # バイト列から復元して確認
        audio_int16 = np.frombuffer(written_data, dtype=np.int16)
        assert len(audio_int16) == len(audio)
        assert np.max(np.abs(audio_int16)) <= 32767

    def test_stop_encoding(self):
        """エンコーディング停止のテスト"""
        encoder = StreamingEncoder(VideoSettings())

        # モックプロセス
        mock_process = MagicMock()
        encoder.process = mock_process

        encoder.stop_encoding()

        mock_process.stdin.close.assert_called_once()
        mock_process.wait.assert_called_once()
        assert encoder.process is None


class TestThumbnailGenerator:
    """ThumbnailGeneratorのテスト"""

    @patch("cv2.VideoCapture")
    @patch("cv2.imwrite")
    def test_generate_from_video(self, mock_imwrite, mock_capture):
        """動画からのサムネイル生成テスト"""
        # モックビデオキャプチャ
        mock_cap = MagicMock()
        mock_cap.get.return_value = 30.0  # FPS
        mock_cap.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
        mock_capture.return_value = mock_cap
        mock_imwrite.return_value = True

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "video.mp4"
            output_path = Path(tmpdir) / "thumb.jpg"

            ThumbnailGenerator.generate_from_video(
                video_path, output_path, time_offset=5.0, size=(640, 360)
            )

            # 正しいフレームにシークされた
            mock_cap.set.assert_called_with(cv2.CAP_PROP_POS_FRAMES, 150)  # 5秒 * 30fps
            mock_imwrite.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_video_read_failure(self, mock_capture):
        """動画読み込み失敗のテスト"""
        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)
        mock_capture.return_value = mock_cap

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "video.mp4"
            output_path = Path(tmpdir) / "thumb.jpg"

            with pytest.raises(RuntimeError, match="Failed to read frame"):
                ThumbnailGenerator.generate_from_video(video_path, output_path)

    @patch("cv2.imwrite")
    def test_generate_from_frames(self, mock_imwrite):
        """フレームからのサムネイル生成テスト"""
        mock_imwrite.return_value = True

        # テストフレーム
        frames = [np.full((480, 640, 3), i * 50, dtype=np.uint8) for i in range(5)]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "thumb.jpg"

            # シングルレイアウト
            ThumbnailGenerator.generate_from_frames(frames, output_path, layout="single")
            assert mock_imwrite.called

            # グリッドレイアウト
            mock_imwrite.reset_mock()
            ThumbnailGenerator.generate_from_frames(frames, output_path, layout="grid")
            assert mock_imwrite.called

            # ストリップレイアウト
            mock_imwrite.reset_mock()
            ThumbnailGenerator.generate_from_frames(frames, output_path, layout="strip")
            assert mock_imwrite.called

    def test_frame_indices_selection(self):
        """フレームインデックス選択のテスト"""
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(100)]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "thumb.jpg"

            with patch("cv2.imwrite") as mock_imwrite:
                mock_imwrite.return_value = True

                # カスタムインデックス
                ThumbnailGenerator.generate_from_frames(
                    frames, output_path, frame_indices=[0, 25, 50, 75, 99]
                )

                # デフォルトインデックス（均等分割）
                ThumbnailGenerator.generate_from_frames(frames, output_path, frame_indices=None)


class TestVisualizerEdgeCases:
    """ビジュアライザーのエッジケーステスト"""

    def test_waveform_empty_audio(self):
        """空の音声データのテスト"""
        visualizer = WaveformVisualizer()

        empty_audio = np.array([])
        frame = visualizer.render_frame(empty_audio)

        # 黒い画像が返される
        assert frame.shape == (1080, 1920, 3)
        assert np.all(frame == 0)

    def test_waveform_extreme_values(self):
        """極端な値のテスト"""
        visualizer = WaveformVisualizer(width=800, height=600)

        # クリッピングする音声
        clipping_audio = np.ones(1000) * 10.0
        frame = visualizer.render_frame(clipping_audio)

        # 描画されるが画面内に収まる
        assert frame.shape == (600, 800, 3)
        assert np.any(frame > 0)

    def test_spectrum_short_audio(self):
        """短い音声のスペクトラムテスト"""
        visualizer = SpectrumVisualizer(fft_size=2048)

        # FFTサイズより短い音声
        short_audio = np.random.normal(0, 0.3, 1000)
        frame = visualizer.render_frame(short_audio)

        assert frame.shape == (1080, 1920, 3)

    def test_spectrum_dc_offset(self):
        """DCオフセットのあるスペクトラムテスト"""
        visualizer = SpectrumVisualizer()

        # DCオフセット付き信号
        audio = np.ones(2048) * 0.5 + np.sin(2 * np.pi * 440 * np.linspace(0, 1, 2048)) * 0.3
        frame = visualizer.render_frame(audio)

        assert np.any(frame > 0)

    def test_particle_maximum_limit(self):
        """パーティクル最大数制限のテスト"""
        visualizer = ParticleVisualizer(max_particles=10)

        # 大量のパーティクルを生成しようとする
        for _ in range(20):
            frame = visualizer.render_frame(1.0, np.ones(32))

        # 最大数を超えない
        assert len(visualizer.particles) <= 10

    def test_particle_boundary_conditions(self):
        """パーティクル境界条件のテスト"""
        visualizer = ParticleVisualizer(
            width=800,
            height=600,
            gravity=(0, 0),  # 重力なし
            wind=(10, 0),  # 強い横風
        )

        # 複数フレームで境界外に出るパーティクルをテスト
        for _ in range(100):
            frame = visualizer.render_frame(0.5)

        # 画面内のパーティクルのみ残る
        for p in visualizer.particles:
            assert 0 <= p["x"] <= 800
            assert 0 <= p["y"] <= 600

    def test_composite_visualizer(self):
        """複合ビジュアライザーのテスト"""
        composite = VisualizerComposite(width=1920, height=1080)

        # 各種ブレンドモード
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410))

        for blend_mode in ["overlay", "screen", "add"]:
            frame = composite.render_composite_frame(audio, blend_mode)
            assert frame.shape == (1080, 1920, 3)
            assert frame.dtype == np.uint8
            assert np.any(frame > 0)


class TestTextOverlayEdgeCases:
    """テキストオーバーレイのエッジケーステスト"""

    def test_empty_text(self):
        """空のテキストのテスト"""
        overlay = TextOverlay()
        background = np.zeros((1080, 1920, 3), dtype=np.uint8)

        frame = overlay.add_text(background, "", (100, 100))
        # 背景と同じ
        assert np.array_equal(frame, background)

    def test_text_outside_bounds(self):
        """画面外のテキストのテスト"""
        overlay = TextOverlay(width=800, height=600)
        background = np.zeros((600, 800, 3), dtype=np.uint8)

        # 画面外の位置
        frame = overlay.add_text(background, "Outside", (900, 700))
        # エラーにならない
        assert frame.shape == background.shape

    def test_text_wrapping(self):
        """テキスト折り返しのテスト"""
        overlay = TextOverlay()
        background = np.zeros((720, 1280, 3), dtype=np.uint8)

        long_text = "This is a very long text that should be wrapped to multiple lines when the maximum width is specified"

        frame = overlay.add_multiline_text(background, long_text, (50, 50), max_width=300)

        assert frame.shape == background.shape
        assert np.any(frame > 0)

    def test_text_alignment(self):
        """テキスト配置のテスト"""
        overlay = TextOverlay()
        background = np.zeros((600, 800, 3), dtype=np.uint8)

        text = "Aligned Text"

        for align in ["left", "center", "right"]:
            frame = overlay.add_multiline_text(background, text, (400, 300), align=align)
            assert not np.array_equal(frame, background)

    def test_text_style_combination(self):
        """テキストスタイルの組み合わせテスト"""
        overlay = TextOverlay()
        background = np.zeros((480, 640, 3), dtype=np.uint8)

        style = TextStyle(
            font_size=72,
            font_color=(255, 255, 255),
            background_color=(0, 0, 0, 200),
            outline_color=(255, 0, 0),
            outline_thickness=3,
            shadow_offset=(5, 5),
            shadow_color=(128, 128, 128),
        )

        frame = overlay.add_text(background, "Styled Text", (50, 100), style=style)

        assert np.any(frame > 0)

    def test_lyrics_no_subtitles(self):
        """字幕なしの歌詞レンダラーテスト"""
        renderer = LyricsRenderer()  # SRTファイルなし
        background = np.zeros((720, 1280, 3), dtype=np.uint8)

        frame = renderer.render_lyrics_frame(background, 5.0)
        # 背景と同じ
        assert np.array_equal(frame, background)

    def test_lyrics_invalid_time(self):
        """無効な時刻での歌詞取得テスト"""
        renderer = LyricsRenderer()

        # 負の時刻
        lyrics = renderer.get_lyrics_at_time(-1.0)
        assert lyrics is None

        # 非常に大きな時刻
        lyrics = renderer.get_lyrics_at_time(999999.0)
        assert lyrics is None

    def test_metadata_display(self):
        """メタデータ表示のテスト"""
        display = MetadataDisplay()
        background = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # 各種位置でのメタデータ表示
        positions = ["top-left", "top-right", "bottom-left", "bottom-right"]

        for pos in positions:
            frame = display.render_metadata(
                background,
                title="Test Song",
                artist="Test Artist",
                additional_info={"Album": "Test Album", "Year": "2024"},
                position=pos,
                fade_alpha=0.8,
            )
            assert not np.array_equal(frame, background)

    def test_metadata_fade_effect(self):
        """メタデータフェード効果のテスト"""
        display = MetadataDisplay()
        background = np.zeros((720, 1280, 3), dtype=np.uint8)

        # フェードアニメーション
        frames = []
        for alpha in np.linspace(0, 1, 10):
            frame = display.render_metadata(
                background, "Fading Title", "Fading Artist", fade_alpha=alpha
            )
            frames.append(frame)

        # フェードによって変化している
        for i in range(1, len(frames)):
            assert not np.array_equal(frames[i], frames[i - 1])


class TestIntegrationVideo:
    """ビデオ機能の統合テスト"""

    def test_full_video_pipeline(self):
        """完全なビデオパイプラインのテスト"""
        # 設定
        width, height = 640, 480
        fps = 10
        duration = 2.0
        sample_rate = 44100

        # 音声生成
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.5

        # ビジュアライザー
        waveform_viz = WaveformVisualizer(width, height, sample_rate)
        spectrum_viz = SpectrumVisualizer(width, height, sample_rate)

        # テキストオーバーレイ
        text_overlay = TextOverlay(width, height)

        # フレーム生成
        frames = []
        samples_per_frame = int(sample_rate / fps)

        for i in range(int(duration * fps)):
            # 音声チャンク
            start = i * samples_per_frame
            end = start + samples_per_frame
            audio_chunk = audio[start:end] if end <= len(audio) else audio[start:]

            # ビジュアライズ
            wave_frame = waveform_viz.render_frame(audio_chunk)
            spec_frame = spectrum_viz.render_frame(audio_chunk)

            # 合成
            if cv2 is not None:
                frame = cv2.addWeighted(wave_frame, 0.5, spec_frame, 0.5, 0)
            else:
                # cv2が使えない場合は簡易的な合成
                frame = (wave_frame * 0.5 + spec_frame * 0.5).astype(np.uint8)

            # テキスト追加
            frame = text_overlay.add_text(
                frame, f"Frame {i+1}/{int(duration * fps)}", (10, 30), alpha=0.8
            )

            frames.append(frame)

        assert len(frames) == int(duration * fps)
        assert all(f.shape == (height, width, 3) for f in frames)

    @patch("subprocess.run")
    @patch("cv2.imwrite")
    @patch("soundfile.write")
    def test_encoder_integration(self, mock_sf_write, mock_cv_write, mock_subprocess):
        """エンコーダー統合テスト"""
        mock_cv_write.return_value = True
        mock_subprocess.return_value.returncode = 0

        # ビデオ設定
        settings = VideoSettings(width=1280, height=720, fps=24, codec="h264", preset="fast")

        encoder = VideoEncoder(settings)

        # テストデータ
        frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(24)]
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_video.mp4"

            encoder.encode(frames, audio, output_path, sample_rate=44100)

            # 各コンポーネントが呼ばれた
            assert mock_cv_write.call_count == len(frames)
            assert mock_sf_write.called
            assert mock_subprocess.called

            # FFmpegコマンドの確認
            ffmpeg_cmd = mock_subprocess.call_args[0][0]
            assert "-c:v" in ffmpeg_cmd
            assert "libx264" in ffmpeg_cmd
            assert "-preset" in ffmpeg_cmd
            assert "fast" in ffmpeg_cmd
