"""
動画生成機能のテスト
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch, MagicMock

from automix.video.visualizer import (
    WaveformVisualizer,
    SpectrumVisualizer,
    ParticleVisualizer,
    VisualizerComposite
)
from automix.video.text_overlay import (
    TextOverlay,
    LyricsRenderer,
    MetadataDisplay
)
from automix.video.encoder import (
    VideoEncoder,
    VideoSettings,
    CodecOptions
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
            width=1920,
            height=1080,
            sample_rate=sr,
            color=(0, 255, 0),
            style='filled'
        )
        
        # 1フレーム分のデータ
        frame_duration = 1/30  # 30fps
        samples_per_frame = int(sr * frame_duration)
        frame_data = waveform[:samples_per_frame]
        
        frame = visualizer.render_frame(frame_data)
        
        assert frame.shape == (1080, 1920, 3)
        assert frame.dtype == np.uint8
        assert np.any(frame > 0)  # 何か描画されている
    
    def test_waveform_styles(self, sample_waveform):
        """異なる波形スタイルのテスト"""
        waveform, sr = sample_waveform
        
        styles = ['line', 'filled', 'mirror', 'circular']
        
        for style in styles:
            visualizer = WaveformVisualizer(
                width=800,
                height=600,
                sample_rate=sr,
                style=style
            )
            
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
        
        visualizer = WaveformVisualizer(
            width=1280,
            height=720,
            sample_rate=sr,
            stereo=True
        )
        
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
            np.sin(2 * np.pi * 440 * t) +
            0.5 * np.sin(2 * np.pi * 880 * t) +
            0.3 * np.sin(2 * np.pi * 1760 * t)
        )
        
        visualizer = SpectrumVisualizer(
            width=1920,
            height=1080,
            sample_rate=sr,
            fft_size=2048,
            bar_count=64,
            color_scheme='rainbow'
        )
        
        frame = visualizer.render_frame(signal)
        
        assert frame.shape == (1080, 1920, 3)
        assert np.any(frame > 0)
    
    def test_frequency_scales(self):
        """周波数スケールのテスト"""
        sr = 44100
        signal = np.random.normal(0, 0.3, 4096)
        
        scales = ['linear', 'log', 'mel']
        
        for scale in scales:
            visualizer = SpectrumVisualizer(
                width=1024,
                height=768,
                sample_rate=sr,
                frequency_scale=scale
            )
            
            frame = visualizer.render_frame(signal)
            assert frame.shape == (768, 1024, 3)
    
    def test_spectrum_smoothing(self):
        """スペクトラムスムージングのテスト"""
        sr = 44100
        visualizer = SpectrumVisualizer(
            width=800,
            height=600,
            sample_rate=sr,
            smoothing_factor=0.8
        )
        
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
            width=1280,
            height=720,
            max_particles=1000,
            particle_size=5,
            audio_reactive=True
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
            width=800,
            height=600,
            gravity=(0, 0.5),
            wind=(0.1, 0),
            turbulence=0.1
        )
        
        # 複数フレームでパーティクルの動きを確認
        frames = []
        for i in range(10):
            audio_level = 0.5 + 0.3 * np.sin(i * 0.5)
            frame = visualizer.render_frame(audio_level)
            frames.append(frame)
        
        # フレーム間で変化がある
        for i in range(1, len(frames)):
            assert not np.array_equal(frames[i-1], frames[i])


class TestTextOverlay:
    """TextOverlayのテスト"""
    
    def test_basic_text_rendering(self):
        """基本的なテキストレンダリングのテスト"""
        overlay = TextOverlay(
            width=1920,
            height=1080,
            font_size=48,
            font_color=(255, 255, 255),
            background_color=(0, 0, 0, 128)
        )
        
        # 背景画像
        background = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # テキストを追加
        frame = overlay.add_text(
            background,
            "Test Song Title",
            position=(100, 100)
        )
        
        assert frame.shape == background.shape
        assert not np.array_equal(frame, background)
    
    def test_text_effects(self):
        """テキストエフェクトのテスト"""
        overlay = TextOverlay(
            width=1280,
            height=720
        )
        
        background = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # フェードイン効果
        for alpha in [0.0, 0.5, 1.0]:
            frame = overlay.add_text(
                background,
                "Fade Test",
                position=(100, 100),
                alpha=alpha
            )
            # アルファ値によって見え方が変わる
            assert isinstance(frame, np.ndarray)
    
    def test_multi_line_text(self):
        """複数行テキストのテスト"""
        overlay = TextOverlay(
            width=800,
            height=600,
            line_spacing=1.5
        )
        
        background = np.zeros((600, 800, 3), dtype=np.uint8)
        
        multi_line_text = "Line 1\nLine 2\nLine 3"
        frame = overlay.add_text(
            background,
            multi_line_text,
            position=(50, 50)
        )
        
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
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            f.write(srt_content)
            srt_path = Path(f.name)
        
        try:
            renderer = LyricsRenderer(
                srt_file=srt_path,
                width=1920,
                height=1080
            )
            
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
            width=1280,
            height=720,
            karaoke_mode=True,
            highlight_color=(255, 255, 0)
        )
        
        background = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # 歌詞と進行度
        lyrics = "This is a test line"
        progress = 0.5  # 50%進行
        
        frame = renderer.render_karaoke(
            background,
            lyrics,
            progress,
            position=(100, 400)
        )
        
        assert frame.shape == background.shape
        assert np.any(frame[:, :, 2] > 0)  # 黄色のハイライト


class TestVideoEncoder:
    """VideoEncoderのテスト"""
    
    def test_video_settings(self):
        """ビデオ設定のテスト"""
        settings = VideoSettings(
            width=1920,
            height=1080,
            fps=30,
            bitrate='8M',
            codec='h264',
            preset='medium'
        )
        
        assert settings.width == 1920
        assert settings.height == 1080
        assert settings.fps == 30
    
    @patch('automix.video.encoder.moviepy.VideoFileClip')
    def test_encoding_process(self, mock_moviepy):
        """エンコーディングプロセスのテスト"""
        settings = VideoSettings(
            width=1280,
            height=720,
            fps=24
        )
        
        encoder = VideoEncoder(settings)
        
        # テスト用のフレーム
        frames = [
            np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        
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
        codecs = ['h264', 'h265', 'vp9']
        
        for codec in codecs:
            options = CodecOptions.get_options(codec)
            assert isinstance(options, dict)
            assert 'codec' in options
    
    def test_resolution_presets(self):
        """解像度プリセットのテスト"""
        presets = {
            '4k': (3840, 2160),
            '1080p': (1920, 1080),
            '720p': (1280, 720),
            '480p': (854, 480)
        }
        
        for preset_name, (width, height) in presets.items():
            settings = VideoSettings.from_preset(preset_name)
            assert settings.width == width
            assert settings.height == height