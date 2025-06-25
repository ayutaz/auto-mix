"""
音声ビジュアライザーモジュール
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class VisualizerConfig:
    """ビジュアライザー設定"""

    width: int = 1920
    height: int = 1080
    fps: int = 30
    background_color: tuple[int, int, int] = (0, 0, 0)
    foreground_color: tuple[int, int, int] = (0, 255, 0)


class WaveformVisualizer:
    """波形ビジュアライザー"""

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        sample_rate: int = 44100,
        color: tuple[int, int, int] = (0, 255, 0),
        style: str = "filled",
        stereo: bool = False,
    ):
        """
        Args:
            width: 画像幅
            height: 画像高さ
            sample_rate: サンプルレート
            color: 波形の色 (B, G, R)
            style: 描画スタイル ('line', 'filled', 'mirror', 'circular')
            stereo: ステレオ表示
        """
        self.width = width
        self.height = height
        self.sample_rate = sample_rate
        self.color = color
        self.style = style
        self.stereo = stereo

    def render_frame(self, audio_data: NDArray[np.float32]) -> NDArray[np.uint8]:
        """
        1フレーム分の波形を描画

        Args:
            audio_data: 音声データ

        Returns:
            NDArray[np.uint8]: 描画された画像 (H, W, 3)
        """
        # 背景を作成
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if cv2 is None:
            return frame

        if self.stereo and audio_data.ndim == 2:
            # ステレオ表示
            mid_y = self.height // 2
            self._draw_waveform(frame, audio_data[:, 0], 0, mid_y)
            self._draw_waveform(frame, audio_data[:, 1], mid_y, self.height)
        else:
            # モノラル表示
            if audio_data.ndim == 2:
                audio_data = np.mean(audio_data, axis=1)
            self._draw_waveform(frame, audio_data, 0, self.height)

        return frame

    def _draw_waveform(
        self, frame: NDArray[np.uint8], audio: NDArray[np.float32], y_start: int, y_end: int
    ) -> None:
        """波形を描画"""
        height = y_end - y_start
        mid_y = y_start + height // 2

        # 空の音声データの場合は何も描画しない
        if len(audio) == 0:
            return

        # サンプル数をフレーム幅に合わせる
        if len(audio) > self.width:
            # ダウンサンプリング
            indices = np.linspace(0, len(audio) - 1, self.width).astype(int)
            audio_resampled = audio[indices]
        else:
            # アップサンプリング
            x_old = np.linspace(0, 1, len(audio))
            x_new = np.linspace(0, 1, self.width)
            audio_resampled = np.interp(x_new, x_old, audio)

        # 正規化
        max_val = np.max(np.abs(audio_resampled))
        if max_val > 0:
            audio_normalized = audio_resampled / max_val
        else:
            audio_normalized = audio_resampled

        if self.style == "line":
            self._draw_line_waveform(frame, audio_normalized, mid_y, height)
        elif self.style == "filled":
            self._draw_filled_waveform(frame, audio_normalized, mid_y, height)
        elif self.style == "mirror":
            self._draw_mirror_waveform(frame, audio_normalized, mid_y, height)
        elif self.style == "circular":
            self._draw_circular_waveform(frame, audio_normalized, mid_y, height)

    def _draw_line_waveform(
        self, frame: NDArray[np.uint8], audio: NDArray[np.float32], mid_y: int, height: int
    ) -> None:
        """線形波形を描画"""
        points = []
        for i in range(len(audio)):
            y = int(mid_y - audio[i] * height * 0.4)
            points.append([i * self.width // len(audio), y])

        points_array = np.array(points, np.int32)
        cv2.polylines(frame, [points_array], False, self.color, 2)

    def _draw_filled_waveform(
        self, frame: NDArray[np.uint8], audio: NDArray[np.float32], mid_y: int, height: int
    ) -> None:
        """塗りつぶし波形を描画"""
        for i in range(len(audio)):
            x = i * self.width // len(audio)
            y_offset = int(abs(audio[i]) * height * 0.4)
            cv2.line(frame, (x, mid_y - y_offset), (x, mid_y + y_offset), self.color, 1)

    def _draw_mirror_waveform(
        self, frame: NDArray[np.uint8], audio: NDArray[np.float32], mid_y: int, height: int
    ) -> None:
        """ミラー波形を描画"""
        # 上部
        points_top = []
        # 下部
        points_bottom = []

        for i in range(len(audio)):
            x = i * self.width // len(audio)
            y_offset = int(audio[i] * height * 0.4)
            points_top.append([x, mid_y - abs(y_offset)])
            points_bottom.append([x, mid_y + abs(y_offset)])

        points_top_array = np.array(points_top, np.int32)
        points_bottom_array = np.array(points_bottom, np.int32)

        cv2.polylines(frame, [points_top_array], False, self.color, 2)
        cv2.polylines(frame, [points_bottom_array], False, self.color, 2)

    def _draw_circular_waveform(
        self, frame: NDArray[np.uint8], audio: NDArray[np.float32], mid_y: int, height: int
    ) -> None:
        """円形波形を描画"""
        center_x = self.width // 2
        center_y = mid_y
        radius = min(self.width, height) // 3

        points_list = []
        for i in range(len(audio)):
            angle = 2 * np.pi * i / len(audio)
            r = radius + audio[i] * radius * 0.5
            x = int(center_x + r * np.cos(angle))
            y = int(center_y + r * np.sin(angle))
            points_list.append([x, y])

        points_array = np.array(points_list, np.int32)
        cv2.polylines(frame, [points_array], True, self.color, 2)


class SpectrumVisualizer:
    """スペクトラムビジュアライザー"""

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        sample_rate: int = 44100,
        fft_size: int = 2048,
        bar_count: int = 64,
        color_scheme: str = "rainbow",
        frequency_scale: str = "log",
        smoothing_factor: float = 0.8,
    ):
        """
        Args:
            width: 画像幅
            height: 画像高さ
            sample_rate: サンプルレート
            fft_size: FFTサイズ
            bar_count: バーの数
            color_scheme: カラースキーム
            frequency_scale: 周波数スケール ('linear', 'log', 'mel')
            smoothing_factor: スムージング係数
        """
        self.width = width
        self.height = height
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.bar_count = bar_count
        self.color_scheme = color_scheme
        self.frequency_scale = frequency_scale
        self.smoothing_factor = smoothing_factor

        # 前フレームのスペクトラム（スムージング用）
        self.prev_spectrum = np.zeros(bar_count)

    def render_frame(self, audio_data: NDArray[np.float32]) -> NDArray[np.uint8]:
        """
        1フレーム分のスペクトラムを描画

        Args:
            audio_data: 音声データ

        Returns:
            NDArray[np.uint8]: 描画された画像 (H, W, 3)
        """
        # FFT計算
        if len(audio_data) < self.fft_size:
            audio_padded = np.pad(audio_data, (0, self.fft_size - len(audio_data)))
        else:
            audio_padded = audio_data[: self.fft_size]

        # ウィンドウ関数
        window = np.hanning(len(audio_padded))
        windowed = audio_padded * window

        # FFT
        fft = np.fft.rfft(windowed)
        magnitude = np.abs(fft)

        # 周波数ビンをバーにマッピング
        spectrum = self._map_to_bars(magnitude)

        # スムージング
        spectrum = (
            self.smoothing_factor * self.prev_spectrum + (1 - self.smoothing_factor) * spectrum
        )
        self.prev_spectrum = spectrum.astype(np.float64)

        # 描画
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if cv2 is None:
            return frame

        self._draw_bars(frame, spectrum)

        return frame

    def _map_to_bars(self, magnitude: NDArray[np.float32]) -> NDArray[np.float32]:
        """FFT結果をバーにマッピング"""
        freqs = np.fft.rfftfreq(self.fft_size, 1 / self.sample_rate)

        if self.frequency_scale == "log":
            # 対数スケール
            min_freq = 20
            max_freq = self.sample_rate / 2
            bar_freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), self.bar_count + 1)
        elif self.frequency_scale == "mel":
            # メルスケール
            min_mel = 2595 * np.log10(1 + 20 / 700)
            max_mel = 2595 * np.log10(1 + self.sample_rate / 2 / 700)
            mel_points = np.linspace(min_mel, max_mel, self.bar_count + 1)
            bar_freqs = 700 * (10 ** (mel_points / 2595) - 1)
        else:
            # リニアスケール
            bar_freqs = np.linspace(0, self.sample_rate / 2, self.bar_count + 1)

        # 各バーの値を計算
        bars = np.zeros(self.bar_count)
        for i in range(self.bar_count):
            freq_mask = (freqs >= bar_freqs[i]) & (freqs < bar_freqs[i + 1])
            if np.any(freq_mask):
                bars[i] = np.mean(magnitude[freq_mask])

        # 正規化（dBスケール）
        bars_db = 20 * np.log10(bars + 1e-10)
        bars_normalized = (bars_db + 60) / 60  # -60dB to 0dB
        bars_normalized = np.clip(bars_normalized, 0, 1)

        return bars_normalized.astype(np.float32)

    def _draw_bars(self, frame: NDArray[np.uint8], spectrum: NDArray[np.float32]) -> None:
        """バーを描画"""
        bar_width = self.width // self.bar_count
        bar_spacing = 2

        for i in range(self.bar_count):
            x = i * bar_width + bar_spacing
            bar_height = int(spectrum[i] * self.height * 0.8)
            y = self.height - bar_height

            # 色を計算
            if self.color_scheme == "rainbow":
                hue = int(i * 180 / self.bar_count)
                color = cv2.cvtColor(
                    np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
                )[0, 0].tolist()
            elif self.color_scheme == "gradient":
                intensity = int(spectrum[i] * 255)
                color = (intensity, intensity // 2, 255 - intensity)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x + bar_width - bar_spacing * 2, self.height), color, -1)


class ParticleVisualizer:
    """パーティクルビジュアライザー"""

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        max_particles: int = 1000,
        particle_size: int = 3,
        audio_reactive: bool = True,
        gravity: tuple[float, float] = (0, 0.1),
        wind: tuple[float, float] = (0, 0),
        turbulence: float = 0.0,
    ):
        """
        Args:
            width: 画像幅
            height: 画像高さ
            max_particles: 最大パーティクル数
            particle_size: パーティクルサイズ
            audio_reactive: 音声反応モード
            gravity: 重力
            wind: 風力
            turbulence: 乱流強度
        """
        self.width = width
        self.height = height
        self.max_particles = max_particles
        self.particle_size = particle_size
        self.audio_reactive = audio_reactive
        self.gravity = gravity
        self.wind = wind
        self.turbulence = turbulence

        # パーティクルの初期化
        self.particles: list[dict[str, Any]] = []
        self._init_particles()

    def _init_particles(self) -> None:
        """パーティクルを初期化"""
        for _ in range(self.max_particles // 10):  # 初期は少なめ
            self.particles.append(
                {
                    "x": np.random.uniform(0, self.width),
                    "y": np.random.uniform(0, self.height),
                    "vx": np.random.uniform(-1, 1),
                    "vy": np.random.uniform(-1, 1),
                    "life": np.random.uniform(0.5, 1.0),
                    "size": self.particle_size,
                    "color": [255, 255, 255],
                }
            )

    def render_frame(
        self, audio_level: float, frequency_data: NDArray[np.float32] | None = None
    ) -> NDArray[np.uint8]:
        """
        1フレーム分のパーティクルを描画

        Args:
            audio_level: 音声レベル (0.0-1.0)
            frequency_data: 周波数データ

        Returns:
            NDArray[np.uint8]: 描画された画像 (H, W, 3)
        """
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if cv2 is None:
            return frame

        # 音声に応じてパーティクルを生成
        if self.audio_reactive and audio_level > 0.5:
            self._spawn_particles(audio_level, frequency_data)

        # パーティクルを更新・描画
        self._update_particles()
        self._draw_particles(frame)

        return frame

    def _spawn_particles(
        self, audio_level: float, frequency_data: NDArray[np.float32] | None
    ) -> None:
        """新しいパーティクルを生成"""
        spawn_count = int(audio_level * 10)

        for _ in range(spawn_count):
            if len(self.particles) >= self.max_particles:
                break

            # 生成位置（下部中央）
            x = self.width / 2 + np.random.normal(0, 50)
            y = self.height - 10

            # 初速度（上向き）
            vx = np.random.normal(0, 2)
            vy = -audio_level * 20 - np.random.uniform(5, 10)

            # 色（周波数データに基づく）
            if frequency_data is not None and len(frequency_data) > 0:
                freq_idx = int(np.random.uniform(0, len(frequency_data)))
                hue = int(frequency_data[freq_idx] * 180)
                color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
                color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
                color = color_bgr.tolist()
            else:
                color = [255, 255, 255]

            self.particles.append(
                {
                    "x": x,
                    "y": y,
                    "vx": vx,
                    "vy": vy,
                    "life": 1.0,
                    "size": self.particle_size + np.random.randint(-1, 2),
                    "color": color,
                }
            )

    def _update_particles(self) -> None:
        """パーティクルを更新"""
        alive_particles = []

        for p in self.particles:
            # 物理演算
            p["vx"] += self.gravity[0] + self.wind[0]
            p["vy"] += self.gravity[1] + self.wind[1]

            # 乱流
            if self.turbulence > 0:
                p["vx"] += np.random.normal(0, self.turbulence)
                p["vy"] += np.random.normal(0, self.turbulence)

            # 位置更新
            p["x"] += p["vx"]
            p["y"] += p["vy"]

            # 寿命減少
            p["life"] -= 0.01

            # 画面内かつ生きているパーティクルのみ残す
            if 0 <= p["x"] <= self.width and 0 <= p["y"] <= self.height and p["life"] > 0:
                alive_particles.append(p)

        self.particles = alive_particles

    def _draw_particles(self, frame: NDArray[np.uint8]) -> None:
        """パーティクルを描画"""
        for p in self.particles:
            alpha = p["life"]
            color = [int(c * alpha) for c in p["color"]]

            cv2.circle(frame, (int(p["x"]), int(p["y"])), p["size"], color, -1)


class VisualizerComposite:
    """複合ビジュアライザー"""

    def __init__(self, width: int = 1920, height: int = 1080, sample_rate: int = 44100):
        """
        Args:
            width: 画像幅
            height: 画像高さ
            sample_rate: サンプルレート
        """
        self.width = width
        self.height = height
        self.sample_rate = sample_rate

        # 各ビジュアライザーを初期化
        self.waveform = WaveformVisualizer(width, height, sample_rate, style="mirror")
        self.spectrum = SpectrumVisualizer(width, height, sample_rate)
        self.particles = ParticleVisualizer(width, height)

    def render_composite_frame(
        self, audio_data: NDArray[np.float32], blend_mode: str = "overlay"
    ) -> NDArray[np.uint8]:
        """
        複合フレームを描画

        Args:
            audio_data: 音声データ
            blend_mode: ブレンドモード

        Returns:
            NDArray[np.uint8]: 描画された画像 (H, W, 3)
        """
        # cv2が使えない場合は空のフレームを返す
        if cv2 is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 各要素を描画
        waveform_frame = self.waveform.render_frame(audio_data)
        spectrum_frame = self.spectrum.render_frame(audio_data)

        # 音声レベルを計算
        audio_level = np.sqrt(np.mean(audio_data**2))
        particles_frame = self.particles.render_frame(audio_level)

        # ブレンド
        if blend_mode == "overlay":
            composite = cv2.addWeighted(waveform_frame, 0.5, spectrum_frame, 0.5, 0)
            composite = cv2.add(composite, particles_frame)
        elif blend_mode == "screen":
            composite = 255 - ((255 - waveform_frame) * (255 - spectrum_frame) // 255)
            composite = cv2.add(composite, particles_frame)
        else:
            composite = waveform_frame + spectrum_frame + particles_frame

        # クリッピング
        composite = np.clip(composite, 0, 255).astype(np.uint8)

        return composite.astype(np.uint8)
