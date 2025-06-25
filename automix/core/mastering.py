"""
マスタリング処理モジュール
"""

from dataclasses import dataclass

import numpy as np
import scipy.signal
from numpy.typing import NDArray


@dataclass
class MasteringSettings:
    """マスタリング設定"""

    target_lufs: float = -14.0
    ceiling_db: float = -0.3
    use_limiter: bool = True
    use_multiband_comp: bool = True
    use_eq: bool = True
    eq_preset: str = "neutral"


class MasteringProcessor:
    """マスタリング処理クラス"""

    def __init__(
        self, sample_rate: int = 44100, target_lufs: float = -14.0, ceiling_db: float = -0.3
    ):
        """
        Args:
            sample_rate: サンプルレート
            target_lufs: ターゲットLUFS
            ceiling_db: 最大レベル（dB）
        """
        self.sample_rate = sample_rate
        self.target_lufs = target_lufs
        self.ceiling_db = ceiling_db

        # サブプロセッサーの初期化
        self.eq = FinalEQ(sample_rate)
        self.multiband = MultibandCompressor(sample_rate)
        self.limiter = LimiterProcessor(sample_rate, ceiling_db)
        self.normalizer = LoudnessNormalizer(sample_rate, target_lufs)

    def process(
        self, audio: NDArray[np.float32], settings: MasteringSettings | None = None
    ) -> NDArray[np.float32]:
        """
        フルマスタリングチェーンを適用

        Args:
            audio: 音声データ
            settings: マスタリング設定

        Returns:
            NDArray[np.float32]: マスタリング後の音声
        """
        if settings is None:
            settings = MasteringSettings()

        processed = audio.copy()

        # 1. EQ
        if settings.use_eq:
            processed = self.eq.process(processed, preset=settings.eq_preset)

        # 2. マルチバンドコンプレッション
        if settings.use_multiband_comp:
            processed = self.multiband.process(processed)

        # 3. ラウドネス正規化
        processed = self.normalizer.normalize(processed)

        # 4. リミッティング
        if settings.use_limiter:
            processed = self.limiter.process(processed)

        return processed

    def measure_lufs(self, audio: NDArray[np.float32]) -> float:
        """LUFS測定"""
        return self.normalizer.measure_integrated_lufs(audio)


class LimiterProcessor:
    """リミッター処理クラス"""

    def __init__(
        self,
        sample_rate: int = 44100,
        threshold_db: float = -0.3,
        release_ms: float = 50.0,
        lookahead_ms: float = 5.0,
    ):
        """
        Args:
            sample_rate: サンプルレート
            threshold_db: 閾値（dB）
            release_ms: リリースタイム（ms）
            lookahead_ms: ルックアヘッドタイム（ms）
        """
        self.sample_rate = sample_rate
        self.threshold_db = threshold_db
        self.threshold = 10 ** (threshold_db / 20)
        self.release_ms = release_ms
        self.lookahead_ms = lookahead_ms

        # ルックアヘッドバッファ
        self.lookahead_samples = int(lookahead_ms * sample_rate / 1000)
        self.delay_buffer: NDArray[np.float32] | None = None

        # リリース係数
        self.release_coef = np.exp(-1 / (sample_rate * release_ms / 1000))

    def process(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        リミッティング処理

        Args:
            audio: 音声データ

        Returns:
            NDArray[np.float32]: リミッティング後の音声
        """
        # ディレイバッファの初期化
        if self.delay_buffer is None:
            self.delay_buffer = np.zeros(max(1, self.lookahead_samples))

        output = np.zeros_like(audio)
        envelope = 0.0

        # ルックアヘッド処理
        padded_audio = np.concatenate([audio, np.zeros(self.lookahead_samples)])

        for i in range(len(audio)):
            # ルックアヘッド区間の最大値
            if self.lookahead_samples > 0:
                lookahead_max = np.max(np.abs(padded_audio[i : i + self.lookahead_samples]))
            else:
                lookahead_max = np.abs(padded_audio[i])

            # エンベロープ追従
            if lookahead_max > envelope:
                envelope = lookahead_max
            else:
                envelope = lookahead_max + self.release_coef * (envelope - lookahead_max)

            # ゲイン計算
            if envelope > self.threshold:
                gain = self.threshold / envelope
            else:
                gain = 1.0

            # ディレイバッファから出力
            if self.lookahead_samples > 0:
                output[i] = self.delay_buffer[0] * gain
                # ディレイバッファ更新
                self.delay_buffer = np.roll(self.delay_buffer, -1)
                self.delay_buffer[-1] = audio[i]
            else:
                output[i] = audio[i] * gain

        return output


class MultibandCompressor:
    """マルチバンドコンプレッサークラス"""

    def __init__(
        self,
        sample_rate: int = 44100,
        crossover_frequencies: list[float] = [200, 2000],
        thresholds_db: list[float] = [-20, -15, -10],
        ratios: list[float] = [2, 3, 4],
        attack_ms: list[float] = [10, 5, 2],
        release_ms: list[float] = [100, 50, 20],
    ):
        """
        Args:
            sample_rate: サンプルレート
            crossover_frequencies: クロスオーバー周波数
            thresholds_db: 各帯域の閾値
            ratios: 各帯域の圧縮比
            attack_ms: 各帯域のアタックタイム
            release_ms: 各帯域のリリースタイム
        """
        self.sample_rate = sample_rate
        self.crossover_frequencies = crossover_frequencies
        self.num_bands = len(crossover_frequencies) + 1

        # 各帯域のコンプレッサー設定
        self.thresholds = [10 ** (db / 20) for db in thresholds_db]
        self.ratios = ratios
        self.attack_coefs = [np.exp(-1 / (sample_rate * ms / 1000)) for ms in attack_ms]
        self.release_coefs = [np.exp(-1 / (sample_rate * ms / 1000)) for ms in release_ms]

    def process(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        マルチバンドコンプレッション処理

        Args:
            audio: 音声データ

        Returns:
            NDArray[np.float32]: 処理後の音声
        """
        # 帯域分割
        bands = self._split_bands(audio)

        # 各帯域をコンプレッション
        compressed_bands = []
        for i, band in enumerate(bands):
            compressed = self._compress_band(
                band,
                i,
                self.thresholds[i],
                self.ratios[i],
                self.attack_coefs[i],
                self.release_coefs[i],
            )
            compressed_bands.append(compressed)

        # 帯域を合成
        return np.sum(compressed_bands, axis=0)

    def _split_bands(self, audio: NDArray[np.float32]) -> list[NDArray[np.float32]]:
        """周波数帯域に分割"""
        # 空の音声の場合は空の帯域リストを返す
        if len(audio) == 0:
            return [np.array([]) for _ in range(self.num_bands)]

        bands = []

        # 単一帯域の場合はフィルタリングなし
        if self.num_bands == 1:
            return [audio.copy()]

        # Linkwitz-Riley フィルタで帯域分割
        for i in range(self.num_bands):
            if i == 0:
                # 最低帯域（ローパス）
                sos = scipy.signal.butter(
                    4,
                    self.crossover_frequencies[0],
                    btype="lowpass",
                    fs=self.sample_rate,
                    output="sos",
                )
            elif i == self.num_bands - 1:
                # 最高帯域（ハイパス）
                sos = scipy.signal.butter(
                    4,
                    self.crossover_frequencies[-1],
                    btype="highpass",
                    fs=self.sample_rate,
                    output="sos",
                )
            else:
                # 中間帯域（バンドパス）
                sos = scipy.signal.butter(
                    4,
                    [self.crossover_frequencies[i - 1], self.crossover_frequencies[i]],
                    btype="bandpass",
                    fs=self.sample_rate,
                    output="sos",
                )

            band = scipy.signal.sosfilt(sos, audio)
            bands.append(band)

        return bands

    def _compress_band(
        self,
        band: NDArray[np.float32],
        band_idx: int,
        threshold: float,
        ratio: float,
        attack_coef: float,
        release_coef: float,
    ) -> NDArray[np.float32]:
        """単一帯域のコンプレッション"""
        envelope = 0.0
        output = np.zeros_like(band)

        for i in range(len(band)):
            input_level = abs(band[i])

            # エンベロープ追従
            if input_level > envelope:
                envelope = input_level + attack_coef * (envelope - input_level)
            else:
                envelope = input_level + release_coef * (envelope - input_level)

            # ゲインリダクション
            if envelope > threshold:
                over_threshold = envelope / threshold
                gain = threshold * (over_threshold ** (1 / ratio - 1)) / envelope
            else:
                gain = 1.0

            output[i] = band[i] * gain

        return output


class LoudnessNormalizer:
    """ラウドネス正規化クラス"""

    def __init__(
        self, sample_rate: int = 44100, target_lufs: float = -14.0, true_peak_db: float = -1.0
    ):
        """
        Args:
            sample_rate: サンプルレート
            target_lufs: ターゲットLUFS
            true_peak_db: トゥルーピーク制限（dB）
        """
        self.sample_rate = sample_rate
        self.target_lufs = target_lufs
        self.true_peak_db = true_peak_db
        self.true_peak_linear = 10 ** (true_peak_db / 20)

    def normalize(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        ラウドネス正規化

        Args:
            audio: 音声データ

        Returns:
            NDArray[np.float32]: 正規化後の音声
        """
        # 空の音声の場合はそのまま返す
        if len(audio) == 0:
            return audio

        # 現在のLUFSを測定
        current_lufs = self.measure_integrated_lufs(audio)

        # 必要なゲインを計算
        gain_db = self.target_lufs - current_lufs
        gain_linear = 10 ** (gain_db / 20)

        # ゲインを適用
        normalized = audio * gain_linear

        # トゥルーピーク制限
        peak = np.max(np.abs(normalized))
        if peak > self.true_peak_linear:
            normalized = normalized * (self.true_peak_linear / peak)

        return normalized

    def measure_integrated_lufs(self, audio: NDArray[np.float32]) -> float:
        """
        統合ラウドネス（Integrated LUFS）を測定
        簡易版実装

        Args:
            audio: 音声データ

        Returns:
            float: LUFS値
        """
        # 空の音声の場合は-70.0 LUFS（無音）を返す
        if len(audio) == 0:
            return -70.0

        # K-weighting pre-filter
        # High shelf at 1500 Hz, +4 dB
        sos1 = self._high_shelf_filter(1500, 4)
        filtered = scipy.signal.sosfilt(sos1, audio)

        # High-pass at 38 Hz
        sos2 = scipy.signal.butter(2, 38, btype="highpass", fs=self.sample_rate, output="sos")
        filtered = scipy.signal.sosfilt(sos2, filtered)

        # 400ms ブロックでの測定
        block_size = int(0.4 * self.sample_rate)
        hop_size = int(0.1 * self.sample_rate)

        blocks = []
        for i in range(0, len(filtered) - block_size, hop_size):
            block = filtered[i : i + block_size]
            mean_square = np.mean(block**2)
            if mean_square > 0:
                blocks.append(mean_square)

        if not blocks:
            return -70.0

        # 統合ラウドネス計算
        mean_of_means = np.mean(blocks)
        lufs = -0.691 + 10 * np.log10(mean_of_means)

        return float(lufs)

    def _high_shelf_filter(self, freq: float, gain_db: float) -> NDArray[np.float32]:
        """ハイシェルフフィルタの係数計算"""
        w0 = 2 * np.pi * freq / self.sample_rate
        cos_w0 = np.cos(w0)
        A = 10 ** (gain_db / 40)
        S = 1  # Shelf slope

        alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / S - 1) + 2)

        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

        return np.array([[b0 / a0, b1 / a0, b2 / a0, 1, a1 / a0, a2 / a0]])

    @classmethod
    def from_preset(cls, preset: str, sample_rate: int = 44100) -> "LoudnessNormalizer":
        """プリセットから作成"""
        presets = {
            "spotify": {"target_lufs": -14, "true_peak_db": -1},
            "youtube": {"target_lufs": -14, "true_peak_db": -1},
            "apple_music": {"target_lufs": -16, "true_peak_db": -1},
            "tidal": {"target_lufs": -14, "true_peak_db": -1},
            "amazon_music": {"target_lufs": -14, "true_peak_db": -2},
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}")

        return cls(sample_rate=sample_rate, **presets[preset])


class FinalEQ:
    """最終段EQクラス"""

    def __init__(self, sample_rate: int = 44100, linear_phase: bool = False):
        """
        Args:
            sample_rate: サンプルレート
            linear_phase: リニアフェーズモード
        """
        self.sample_rate = sample_rate
        self.linear_phase = linear_phase
        self.bands: list[dict[str, float]] = []

    def add_bell(self, freq: float, gain_db: float, q: float) -> None:
        """ベル型EQを追加"""
        self.bands.append({"type": "bell", "freq": freq, "gain_db": gain_db, "q": q})

    def add_shelf(self, freq: float, gain_db: float, shelf_type: str) -> None:
        """シェルフEQを追加"""
        self.bands.append({"type": shelf_type, "freq": freq, "gain_db": gain_db})

    def process(self, audio: NDArray[np.float32], preset: str | None = None) -> NDArray[np.float32]:
        """
        EQ処理を適用

        Args:
            audio: 音声データ
            preset: プリセット名

        Returns:
            NDArray[np.float32]: EQ適用後の音声
        """
        if preset:
            self._load_preset(preset)

        processed = audio.copy()

        for band in self.bands:
            if band["type"] == "bell":
                processed = self._apply_bell(processed, band["freq"], band["gain_db"], band["q"])
            elif band["type"] in ["low_shelf", "high_shelf"]:
                processed = self._apply_shelf(
                    processed, band["freq"], band["gain_db"], band["type"]
                )

        return processed

    def _apply_bell(
        self, audio: NDArray[np.float32], freq: float, gain_db: float, q: float
    ) -> NDArray[np.float32]:
        """ベル型EQ適用"""
        # Parametric EQ係数
        w0 = 2 * np.pi * freq / self.sample_rate
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        A = 10 ** (gain_db / 40)
        alpha = sin_w0 / (2 * q)

        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A

        b = np.array([b0, b1, b2]) / a0
        a = np.array([a0, a1, a2]) / a0

        if self.linear_phase:
            return scipy.signal.filtfilt(b, a, audio)
        else:
            return scipy.signal.lfilter(b, a, audio)

    def _apply_shelf(
        self, audio: NDArray[np.float32], freq: float, gain_db: float, shelf_type: str
    ) -> NDArray[np.float32]:
        """シェルフEQ適用"""
        w0 = 2 * np.pi * freq / self.sample_rate
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        A = 10 ** (gain_db / 40)
        S = 1
        alpha = sin_w0 / 2 * np.sqrt((A + 1 / A) * (1 / S - 1) + 2)

        if shelf_type == "low_shelf":
            b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
            b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
            a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
            a2 = (A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
        else:  # high_shelf
            b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
            b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
            a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

        b = np.array([b0, b1, b2]) / a0
        a = np.array([a0, a1, a2]) / a0

        if self.linear_phase:
            return scipy.signal.filtfilt(b, a, audio)
        else:
            return scipy.signal.lfilter(b, a, audio)

    def _load_preset(self, preset: str) -> None:
        """プリセットをロード"""
        self.bands.clear()

        presets = {
            "bright": [
                {"type": "high_shelf", "freq": 8000, "gain_db": 3},
                {"type": "bell", "freq": 3000, "gain_db": 2, "q": 0.7},
            ],
            "warm": [
                {"type": "low_shelf", "freq": 200, "gain_db": 2},
                {"type": "bell", "freq": 500, "gain_db": 1.5, "q": 0.8},
                {"type": "high_shelf", "freq": 10000, "gain_db": -2},
            ],
            "neutral": [],
            "vinyl": [
                {"type": "high_shelf", "freq": 15000, "gain_db": -6},
                {"type": "low_shelf", "freq": 50, "gain_db": -3},
                {"type": "bell", "freq": 1000, "gain_db": 1, "q": 0.5},
            ],
            "radio": [
                {"type": "bell", "freq": 2500, "gain_db": 3, "q": 0.6},
                {"type": "high_shelf", "freq": 10000, "gain_db": 2},
            ],
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}")

        self.bands = presets[preset]

    @classmethod
    def from_preset(cls, preset: str, sample_rate: int = 44100) -> "FinalEQ":
        """プリセットからEQを作成"""
        eq = cls(sample_rate=sample_rate)
        eq._load_preset(preset)
        return eq
