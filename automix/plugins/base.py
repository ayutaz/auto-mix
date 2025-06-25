"""
プラグインベースクラスと管理システム
"""

import importlib
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class PluginInterface(ABC):
    """プラグインのベースインターフェース"""

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.enabled = True
        self.config: dict[str, Any] = {}

    @abstractmethod
    def get_info(self) -> dict[str, Any]:
        """プラグイン情報を取得"""
        return {
            "name": self.name,
            "version": self.version,
            "type": self.__class__.__name__,
            "enabled": self.enabled,
        }

    def configure(self, config: dict[str, Any]) -> None:
        """プラグインを設定"""
        self.config.update(config)

    def enable(self) -> None:
        """プラグインを有効化"""
        self.enabled = True

    def disable(self) -> None:
        """プラグインを無効化"""
        self.enabled = False


class AudioEffectPlugin(PluginInterface):
    """音声エフェクトプラグインのベースクラス"""

    @abstractmethod
    def process(self, audio: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        """音声を処理"""
        pass

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """パラメータ定義を取得"""
        pass

    def set_parameter(self, name: str, value: Any) -> None:
        """パラメータを設定"""
        if name in self.get_parameters():
            setattr(self, name, value)
        else:
            raise ValueError(f"Unknown parameter: {name}")


class VisualizerPlugin(PluginInterface):
    """ビジュアライザープラグインのベースクラス"""

    @abstractmethod
    def render(self, audio_data: NDArray[np.float32], width: int, height: int) -> NDArray[np.uint8]:
        """フレームをレンダリング"""
        pass

    @abstractmethod
    def get_style_options(self) -> list[str]:
        """利用可能なスタイルオプションを取得"""
        pass


class AnalyzerPlugin(PluginInterface):
    """音声解析プラグインのベースクラス"""

    @abstractmethod
    def analyze(self, audio: NDArray[np.float32], sample_rate: int) -> dict[str, Any]:
        """音声を解析"""
        pass

    @abstractmethod
    def get_metrics(self) -> list[str]:
        """解析可能なメトリクスを取得"""
        pass


class PluginManager:
    """プラグインマネージャー"""

    def __init__(self, plugin_dirs: list[Path] | None = None):
        self.plugins: dict[str, PluginInterface] = {}
        self.plugin_dirs = plugin_dirs or []
        # 具象クラスのみを登録
        self._plugin_types: dict[str, type[PluginInterface]] = {}

    def register_plugin(self, plugin: PluginInterface) -> None:
        """プラグインを登録"""
        if plugin.name in self.plugins:
            raise ValueError(f"Plugin '{plugin.name}' already registered")
        self.plugins[plugin.name] = plugin

    def unregister_plugin(self, name: str) -> None:
        """プラグインを登録解除"""
        if name in self.plugins:
            del self.plugins[name]

    def get_plugin(self, name: str) -> PluginInterface | None:
        """プラグインを取得"""
        return self.plugins.get(name)

    def list_plugins(self, plugin_type: str | None = None) -> list[dict[str, Any]]:
        """プラグインリストを取得"""
        plugins = []
        for plugin in self.plugins.values():
            info = plugin.get_info()
            if plugin_type is None or info["type"] == plugin_type:
                plugins.append(info)
        return plugins

    def load_plugin_from_file(self, file_path: Path) -> None:
        """ファイルからプラグインを読み込む"""
        spec = importlib.util.spec_from_file_location("plugin", file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # プラグインクラスを探す
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, PluginInterface)
                    and obj != PluginInterface
                    and obj not in self._plugin_types.values()
                ):
                    # プラグインインスタンスを作成して登録
                    try:
                        plugin = obj()
                        self.register_plugin(plugin)
                    except TypeError:
                        # 抽象クラスの場合はスキップ
                        pass

    def load_plugins_from_directory(self, directory: Path) -> None:
        """ディレクトリからプラグインを読み込む"""
        if not directory.exists():
            return

        for file_path in directory.glob("*.py"):
            if file_path.name.startswith("_"):
                continue
            try:
                self.load_plugin_from_file(file_path)
            except Exception as e:
                print(f"Failed to load plugin from {file_path}: {e}")

    def apply_audio_effect_chain(
        self, audio: NDArray[np.float32], sample_rate: int, plugin_names: list[str]
    ) -> NDArray[np.float32]:
        """複数の音声エフェクトプラグインを連続適用"""
        result = audio.copy()
        for name in plugin_names:
            plugin = self.get_plugin(name)
            if plugin and isinstance(plugin, AudioEffectPlugin) and plugin.enabled:
                result = plugin.process(result, sample_rate)
        return result


# 組み込みプラグインの例
class PitchShiftPlugin(AudioEffectPlugin):
    """ピッチシフトプラグイン"""

    def __init__(self) -> None:
        super().__init__("pitch_shift", "1.0.0")
        self.semitones = 0.0

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info["description"] = "Shifts the pitch of audio"
        return info

    def get_parameters(self) -> dict[str, Any]:
        return {
            "semitones": {
                "type": "float",
                "min": -12.0,
                "max": 12.0,
                "default": 0.0,
                "description": "Pitch shift in semitones",
            }
        }

    def process(self, audio: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        if self.semitones == 0:
            return audio

        try:
            import librosa

            return librosa.effects.pitch_shift(
                audio, sr=sample_rate, n_steps=self.semitones
            ).astype(np.float32)
        except ImportError:
            print("librosa not available, returning original audio")
            return audio


class NoiseGatePlugin(AudioEffectPlugin):
    """ノイズゲートプラグイン"""

    def __init__(self) -> None:
        super().__init__("noise_gate", "1.0.0")
        self.threshold_db = -40.0
        self.attack_ms = 1.0
        self.release_ms = 100.0

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info["description"] = "Reduces noise by gating quiet signals"
        return info

    def get_parameters(self) -> dict[str, Any]:
        return {
            "threshold_db": {
                "type": "float",
                "min": -60.0,
                "max": 0.0,
                "default": -40.0,
                "description": "Gate threshold in dB",
            },
            "attack_ms": {
                "type": "float",
                "min": 0.1,
                "max": 10.0,
                "default": 1.0,
                "description": "Attack time in milliseconds",
            },
            "release_ms": {
                "type": "float",
                "min": 10.0,
                "max": 1000.0,
                "default": 100.0,
                "description": "Release time in milliseconds",
            },
        }

    def process(self, audio: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        # 簡易的なノイズゲート実装
        threshold = 10 ** (self.threshold_db / 20)
        envelope = np.abs(audio)

        # スムージング
        attack_samples = int(self.attack_ms * sample_rate / 1000)
        release_samples = int(self.release_ms * sample_rate / 1000)

        gate = np.zeros_like(audio)
        for i in range(len(audio)):
            if envelope[i] > threshold:
                gate[i] = 1.0
            elif i > 0:
                gate[i] = gate[i - 1] * (1 - 1 / release_samples)

        # アタック処理
        for i in range(1, len(gate)):
            if gate[i] > gate[i - 1]:
                gate[i] = gate[i - 1] + (1 - gate[i - 1]) / attack_samples

        return audio * gate


class SpectrumVisualizerPlugin(VisualizerPlugin):
    """スペクトラムビジュアライザープラグイン"""

    def __init__(self) -> None:
        super().__init__("spectrum_visualizer", "1.0.0")
        self.style = "bars"
        self.color_scheme = "rainbow"

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info["description"] = "Displays audio spectrum"
        return info

    def get_style_options(self) -> list[str]:
        return ["bars", "line", "circular"]

    def render(self, audio_data: NDArray[np.float32], width: int, height: int) -> NDArray[np.uint8]:
        # 簡易的なスペクトラム表示
        try:
            import cv2
        except ImportError:
            # cv2が利用できない場合は空のフレームを返す
            return np.zeros((height, width, 3), dtype=np.uint8)

        # FFT
        fft = np.fft.rfft(audio_data)
        magnitude = np.abs(fft)

        # 周波数ビンを画面幅に合わせる
        bins = np.linspace(0, len(magnitude), width, endpoint=False).astype(int)
        spectrum = magnitude[bins]

        # 正規化
        spectrum = spectrum / (np.max(spectrum) + 1e-10)

        # 描画
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        if self.style == "bars":
            bar_width = max(1, width // len(spectrum))
            for i, value in enumerate(spectrum):
                bar_height = int(value * height * 0.8)
                x = i * bar_width
                color = self._get_color(i / len(spectrum))
                cv2.rectangle(
                    frame, (x, height - bar_height), (x + bar_width - 1, height), color, -1
                )

        return frame

    def _get_color(self, position: float) -> tuple:
        """位置に基づいて色を取得"""
        if self.color_scheme == "rainbow":
            try:
                import cv2

                # HSVからRGBに変換
                hue = int(position * 180)
                color = cv2.cvtColor(
                    np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
                )[0, 0]
                return tuple(int(c) for c in color)
            except ImportError:
                # cv2が利用できない場合はデフォルト色
                return (255, 255, 255)
        else:
            # デフォルトは白
            return (255, 255, 255)
