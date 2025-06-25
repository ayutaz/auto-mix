"""
カスタムエフェクトプラグインの例
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import AudioEffectPlugin


class VintageWarmthPlugin(AudioEffectPlugin):
    """ビンテージウォームスプラグイン - アナログ機器の温かみを追加"""

    def __init__(self) -> None:
        super().__init__("vintage_warmth", "1.0.0")
        self.warmth = 0.5
        self.saturation = 0.3

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info["description"] = "Adds vintage analog warmth to audio"
        info["author"] = "AutoMix Plugins"
        return info

    def get_parameters(self) -> dict[str, Any]:
        return {
            "warmth": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5,
                "description": "Amount of warmth to add",
            },
            "saturation": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.3,
                "description": "Analog saturation amount",
            },
        }

    def process(self, audio: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        if not self.enabled:
            return audio

        # ローパスフィルタで高域を少し削る
        from scipy import signal

        # カットオフ周波数を温かさに応じて調整
        cutoff = 8000 + (1 - self.warmth) * 8000  # 8kHz - 16kHz
        nyquist = sample_rate / 2
        normal_cutoff = cutoff / nyquist

        # バターワースフィルタ
        b, a = signal.butter(2, normal_cutoff, btype="low")
        filtered = signal.filtfilt(b, a, audio)

        # サチュレーション（ソフトクリッピング）
        if self.saturation > 0:
            # tanh関数でソフトクリッピング
            drive = 1 + self.saturation * 4
            saturated = np.tanh(filtered * drive) / drive
            filtered = saturated

        # 原音とブレンド
        result = audio * (1 - self.warmth * 0.5) + filtered * (self.warmth * 0.5 + 0.5)
        return result.astype(np.float32)


class VocalEnhancerPlugin(AudioEffectPlugin):
    """ボーカルエンハンサープラグイン - ボーカルの存在感を強調"""

    def __init__(self) -> None:
        super().__init__("vocal_enhancer", "1.0.0")
        self.brightness = 0.5
        self.presence = 0.5
        self.air = 0.3

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info["description"] = "Enhances vocal clarity and presence"
        info["author"] = "AutoMix Plugins"
        return info

    def get_parameters(self) -> dict[str, Any]:
        return {
            "brightness": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5,
                "description": "Vocal brightness (2-4kHz boost)",
            },
            "presence": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5,
                "description": "Vocal presence (5-8kHz boost)",
            },
            "air": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.3,
                "description": "Air frequencies (10kHz+ boost)",
            },
        }

    def process(self, audio: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        if not self.enabled:
            return audio

        from scipy import signal

        result = audio.copy()

        # ブライトネス (2-4kHz)
        if self.brightness > 0:
            freq = 3000
            q = 2
            gain_db = self.brightness * 6  # 最大6dB
            b, a = self._design_peaking_eq(freq, q, gain_db, sample_rate)
            result = signal.filtfilt(b, a, result)

        # プレゼンス (5-8kHz)
        if self.presence > 0:
            freq = 6500
            q = 2
            gain_db = self.presence * 4  # 最大4dB
            b, a = self._design_peaking_eq(freq, q, gain_db, sample_rate)
            result = signal.filtfilt(b, a, result)

        # エア (10kHz+)
        if self.air > 0:
            freq = 12000
            q = 0.7
            gain_db = self.air * 3  # 最大3dB
            b, a = self._design_high_shelf(freq, q, gain_db, sample_rate)
            result = signal.filtfilt(b, a, result)

        return result

    def _design_peaking_eq(self, freq: float, q: float, gain_db: float, fs: float):
        """ピーキングEQフィルタを設計"""
        w0 = 2 * np.pi * freq / fs
        A = 10 ** (gain_db / 40)
        alpha = np.sin(w0) / (2 * q)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        return [b0 / a0, b1 / a0, b2 / a0], [1, a1 / a0, a2 / a0]

    def _design_high_shelf(self, freq: float, q: float, gain_db: float, fs: float):
        """ハイシェルフフィルタを設計"""
        w0 = 2 * np.pi * freq / fs
        A = 10 ** (gain_db / 40)
        alpha = np.sin(w0) / (2 * q)

        b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
        a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

        return [b0 / a0, b1 / a0, b2 / a0], [1, a1 / a0, a2 / a0]


class StereoEnhancerPlugin(AudioEffectPlugin):
    """ステレオエンハンサープラグイン - ステレオイメージを拡張"""

    def __init__(self) -> None:
        super().__init__("stereo_enhancer", "1.0.0")
        self.width = 0.5
        self.bass_mono = True
        self.bass_freq = 120.0

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info["description"] = "Enhances stereo width and imaging"
        info["author"] = "AutoMix Plugins"
        return info

    def get_parameters(self) -> dict[str, Any]:
        return {
            "width": {
                "type": "float",
                "min": 0.0,
                "max": 2.0,
                "default": 0.5,
                "description": "Stereo width (0=mono, 1=normal, 2=wide)",
            },
            "bass_mono": {
                "type": "bool",
                "default": True,
                "description": "Keep bass frequencies in mono",
            },
            "bass_freq": {
                "type": "float",
                "min": 80.0,
                "max": 200.0,
                "default": 120.0,
                "description": "Bass mono frequency cutoff",
            },
        }

    def process(self, audio: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        if not self.enabled:
            return audio

        # モノラル音声の場合はそのまま返す
        if audio.ndim == 1:
            return audio

        # ステレオの場合のみ処理
        if audio.shape[1] == 2:
            from scipy import signal

            left = audio[:, 0]
            right = audio[:, 1]

            # Mid/Side処理
            mid = (left + right) * 0.5
            side = (left - right) * 0.5

            # サイド信号を調整してステレオ幅を変更
            side = side * self.width

            # Bass Mono処理
            if self.bass_mono and self.bass_freq > 0:
                # ローパスフィルタでベース成分を抽出
                nyquist = sample_rate / 2
                normal_cutoff = self.bass_freq / nyquist
                b, a = signal.butter(2, normal_cutoff, btype="low")

                # ベース成分をモノラル化
                bass_left = signal.filtfilt(b, a, left)
                bass_right = signal.filtfilt(b, a, right)
                bass_mono = (bass_left + bass_right) * 0.5

                # 高域成分は元のステレオ幅を維持
                high_left = left - bass_left
                high_right = right - bass_right

                # 再構成
                left = bass_mono + high_left
                right = bass_mono + high_right

                # Mid/Side再計算
                mid = (left + right) * 0.5
                side = (left - right) * 0.5 * self.width

            # L/Rに戻す
            new_left = mid + side
            new_right = mid - side

            # 結果を結合
            result = np.column_stack((new_left, new_right))

            # クリッピング防止
            max_val = np.max(np.abs(result))
            if max_val > 1.0:
                result = result / max_val * 0.95

            return result
        else:
            return audio


class HarmonicExciterPlugin(AudioEffectPlugin):
    """ハーモニックエキサイタープラグイン - 倍音を追加して音に輝きを与える"""

    def __init__(self) -> None:
        super().__init__("harmonic_exciter", "1.0.0")
        self.amount = 0.3
        self.frequency = 3000.0
        self.mix = 0.2

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info["description"] = "Adds harmonic excitement and brilliance"
        info["author"] = "AutoMix Plugins"
        return info

    def get_parameters(self) -> dict[str, Any]:
        return {
            "amount": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.3,
                "description": "Amount of harmonics to generate",
            },
            "frequency": {
                "type": "float",
                "min": 1000.0,
                "max": 10000.0,
                "default": 3000.0,
                "description": "Frequency above which to excite",
            },
            "mix": {
                "type": "float",
                "min": 0.0,
                "max": 0.5,
                "default": 0.2,
                "description": "Mix amount with original signal",
            },
        }

    def process(self, audio: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        if not self.enabled or self.amount == 0:
            return audio

        from scipy import signal

        # ハイパスフィルタで高域成分を抽出
        nyquist = sample_rate / 2
        normal_cutoff = self.frequency / nyquist

        if normal_cutoff >= 1.0:
            normal_cutoff = 0.99

        b, a = signal.butter(2, normal_cutoff, btype="high")
        high_freq = signal.filtfilt(b, a, audio)

        # 非線形処理で倍音を生成
        # ソフトクリッピングとチューブサチュレーションのシミュレーション
        drive = 1 + self.amount * 10
        excited = np.tanh(high_freq * drive)

        # 偶数次倍音を追加（暖かさ）
        even_harmonics = excited**2 * np.sign(excited) * 0.5

        # 奇数次倍音を追加（明瞭さ）
        odd_harmonics = excited**3 * 0.3

        # 倍音をミックス
        harmonics = even_harmonics + odd_harmonics

        # 元の信号とブレンド
        result = audio + harmonics * self.mix

        # クリッピング防止
        max_val = np.max(np.abs(result))
        if max_val > 1.0:
            result = result / max_val * 0.95

        return result
