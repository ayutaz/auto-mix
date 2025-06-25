"""
音声ミックス処理モジュール
"""

import numpy as np
import scipy.signal
from numpy.typing import NDArray


class MixProcessor:
    """音声ミックス処理クラス"""

    def __init__(self, sample_rate: int = 44100, target_lufs: float = -14.0):
        """
        Args:
            sample_rate: サンプルレート
            target_lufs: ターゲットLUFS値
        """
        self.sample_rate = sample_rate
        self.target_lufs = target_lufs

    def mix(
        self,
        vocal: NDArray[np.float32],
        bgm: NDArray[np.float32],
        vocal_gain_db: float = 0.0,
        bgm_gain_db: float = 0.0,
        auto_balance: bool = True,
    ) -> NDArray[np.float32]:
        """
        ボーカルとBGMをミックスする

        Args:
            vocal: ボーカル音声
            bgm: BGM音声
            vocal_gain_db: ボーカルゲイン（dB）
            bgm_gain_db: BGMゲイン（dB）
            auto_balance: 自動バランス調整

        Returns:
            NDArray[np.float32]: ミックスされた音声
        """
        # 長さを合わせる
        min_length = min(len(vocal), len(bgm))
        vocal = vocal[:min_length]
        bgm = bgm[:min_length]

        # ゲイン適用
        vocal_gain = 10 ** (vocal_gain_db / 20)
        bgm_gain = 10 ** (bgm_gain_db / 20)

        vocal_processed = vocal * vocal_gain
        bgm_processed = bgm * bgm_gain

        if auto_balance:
            # 自動バランス調整
            vocal_rms = np.sqrt(np.mean(vocal_processed**2))
            bgm_rms = np.sqrt(np.mean(bgm_processed**2))

            # ボーカルが聞こえやすいようにバランス調整
            if vocal_rms > 0 and bgm_rms > 0:
                target_ratio = 1.5  # ボーカルをBGMの1.5倍の音量に
                current_ratio = vocal_rms / bgm_rms

                if current_ratio < target_ratio:
                    vocal_processed *= target_ratio / current_ratio
                else:
                    bgm_processed *= current_ratio / target_ratio

        # ミックス
        mixed = vocal_processed + bgm_processed

        # クリッピング防止
        max_val = np.max(np.abs(mixed))
        if max_val > 0.95:
            mixed = mixed * 0.95 / max_val

        return mixed

    def measure_lufs(self, audio: NDArray[np.float32]) -> float:
        """
        LUFS（Loudness Units relative to Full Scale）を測定
        簡易版実装

        Args:
            audio: 音声データ

        Returns:
            float: LUFS値
        """
        # K-weighting フィルタ（簡易版）
        # 実際のITU-R BS.1770規格に準拠した実装はより複雑

        # Pre-filter (shelf filter)
        sos1 = scipy.signal.butter(2, 1500, btype="highpass", fs=self.sample_rate, output="sos")
        audio_filtered = scipy.signal.sosfilt(sos1, audio)

        # RLB-weighting
        sos2 = scipy.signal.butter(2, 38, btype="highpass", fs=self.sample_rate, output="sos")
        audio_weighted = scipy.signal.sosfilt(sos2, audio_filtered)

        # Mean square
        mean_square = np.mean(audio_weighted**2)

        # Convert to LUFS
        if mean_square > 0:
            lufs = -0.691 + 10 * np.log10(mean_square)
        else:
            lufs = -70.0

        return float(lufs)


class VolumeProcessor:
    """音量処理クラス"""

    def apply_gain(self, audio: NDArray[np.float32], gain_db: float) -> NDArray[np.float32]:
        """
        ゲインを適用

        Args:
            audio: 音声データ
            gain_db: ゲイン（dB）

        Returns:
            NDArray[np.float32]: ゲイン適用後の音声
        """
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear

    def auto_gain(self, audio: NDArray[np.float32], target_rms: float = 0.2) -> NDArray[np.float32]:
        """
        自動ゲイン調整

        Args:
            audio: 音声データ
            target_rms: ターゲットRMS値

        Returns:
            NDArray[np.float32]: ゲイン調整後の音声
        """
        current_rms = np.sqrt(np.mean(audio**2))

        if current_rms > 0:
            gain = target_rms / current_rms
            return audio * gain

        return audio

    def limit(self, audio: NDArray[np.float32], threshold: float = 0.95) -> NDArray[np.float32]:
        """
        リミッティング

        Args:
            audio: 音声データ
            threshold: 閾値

        Returns:
            NDArray[np.float32]: リミッティング後の音声
        """
        return np.clip(audio, -threshold, threshold)


class EQProcessor:
    """イコライザー処理クラス"""

    def __init__(self, sample_rate: int = 44100):
        """
        Args:
            sample_rate: サンプルレート
        """
        self.sample_rate = sample_rate

    def highpass(
        self, audio: NDArray[np.float32], cutoff: float, order: int = 4
    ) -> NDArray[np.float32]:
        """
        ハイパスフィルタ

        Args:
            audio: 音声データ
            cutoff: カットオフ周波数
            order: フィルタ次数

        Returns:
            NDArray[np.float32]: フィルタ適用後の音声
        """
        sos = scipy.signal.butter(
            order, cutoff, btype="highpass", fs=self.sample_rate, output="sos"
        )
        return scipy.signal.sosfilt(sos, audio)

    def lowpass(
        self, audio: NDArray[np.float32], cutoff: float, order: int = 4
    ) -> NDArray[np.float32]:
        """
        ローパスフィルタ

        Args:
            audio: 音声データ
            cutoff: カットオフ周波数
            order: フィルタ次数

        Returns:
            NDArray[np.float32]: フィルタ適用後の音声
        """
        sos = scipy.signal.butter(order, cutoff, btype="lowpass", fs=self.sample_rate, output="sos")
        return scipy.signal.sosfilt(sos, audio)

    def parametric_eq(
        self, audio: NDArray[np.float32], freq: float, gain_db: float, q: float = 1.0
    ) -> NDArray[np.float32]:
        """
        パラメトリックEQ

        Args:
            audio: 音声データ
            freq: 中心周波数
            gain_db: ゲイン（dB）
            q: Q値

        Returns:
            NDArray[np.float32]: EQ適用後の音声
        """
        # Peaking EQ filter coefficients
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

        # Normalize
        b = np.array([b0, b1, b2]) / a0
        a = np.array([a0, a1, a2]) / a0

        return scipy.signal.lfilter(b, a, audio)


class CompressorProcessor:
    """コンプレッサー処理クラス"""

    def __init__(
        self,
        sample_rate: int = 44100,
        threshold_db: float = -20.0,
        ratio: float = 4.0,
        attack_ms: float = 10.0,
        release_ms: float = 100.0,
    ):
        """
        Args:
            sample_rate: サンプルレート
            threshold_db: 閾値（dB）
            ratio: 圧縮比
            attack_ms: アタックタイム（ms）
            release_ms: リリースタイム（ms）
        """
        self.sample_rate = sample_rate
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.attack_ms = attack_ms
        self.release_ms = release_ms

        # アタック/リリース係数を計算
        self.attack_coef = np.exp(-1 / (self.sample_rate * self.attack_ms / 1000))
        self.release_coef = np.exp(-1 / (self.sample_rate * self.release_ms / 1000))

    def compress(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        コンプレッション処理

        Args:
            audio: 音声データ

        Returns:
            NDArray[np.float32]: コンプレッション後の音声
        """
        threshold = 10 ** (self.threshold_db / 20)

        # エンベロープフォロワー
        envelope = np.zeros_like(audio)
        for i in range(len(audio)):
            input_level = abs(audio[i])

            if input_level > envelope[i - 1] if i > 0 else 0:
                # Attack
                envelope[i] = input_level + self.attack_coef * (
                    envelope[i - 1] if i > 0 else 0 - input_level
                )
            else:
                # Release
                envelope[i] = input_level + self.release_coef * (
                    envelope[i - 1] if i > 0 else 0 - input_level
                )

        # ゲインリダクション計算
        gain_reduction = np.ones_like(audio)
        mask = envelope > threshold

        if np.any(mask):
            over_threshold = envelope[mask] / threshold
            gain_reduction[mask] = (
                threshold * (over_threshold ** (1 / self.ratio - 1))
            ) / envelope[mask]

        # 適用
        return audio * gain_reduction

    def multiband_compress(
        self,
        audio: NDArray[np.float32],
        bands: list[tuple[float, float]],
        thresholds: list[float],
        ratios: list[float],
    ) -> NDArray[np.float32]:
        """
        マルチバンドコンプレッション

        Args:
            audio: 音声データ
            bands: 周波数帯域のリスト [(low, high), ...]
            thresholds: 各帯域の閾値
            ratios: 各帯域の圧縮比

        Returns:
            NDArray[np.float32]: マルチバンドコンプレッション後の音声
        """
        # TODO: マルチバンド実装
        # 現在は単一バンドのみ
        return self.compress(audio)


class AlignmentProcessor:
    """音声アライメント処理クラス"""

    def __init__(self, sample_rate: int = 44100):
        """
        Args:
            sample_rate: サンプルレート
        """
        self.sample_rate = sample_rate

    def align(
        self, audio1: NDArray[np.float32], audio2: NDArray[np.float32], max_offset_sec: float = 5.0
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], int]:
        """
        2つの音声を同期させる

        Args:
            audio1: 音声1
            audio2: 音声2
            max_offset_sec: 最大オフセット（秒）

        Returns:
            Tuple[NDArray[np.float32], NDArray[np.float32], int]:
                同期された音声1、音声2、オフセット
        """
        max_offset_samples = int(max_offset_sec * self.sample_rate)

        # 相互相関を計算
        correlation = scipy.signal.correlate(audio1, audio2, mode="valid")

        # 最大相関の位置を見つける
        offset = np.argmax(np.abs(correlation))

        # オフセットを制限
        offset = np.clip(offset, -max_offset_samples, max_offset_samples)

        # アライメント
        if offset > 0:
            audio2_aligned = np.pad(audio2, (offset, 0), mode="constant")[: len(audio1)]
            audio1_aligned = audio1
        else:
            audio1_aligned = np.pad(audio1, (-offset, 0), mode="constant")[: len(audio2)]
            audio2_aligned = audio2

        # 長さを合わせる
        min_len = min(len(audio1_aligned), len(audio2_aligned))
        audio1_aligned = audio1_aligned[:min_len]
        audio2_aligned = audio2_aligned[:min_len]

        return audio1_aligned, audio2_aligned, int(offset)
