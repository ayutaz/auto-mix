"""
エフェクト処理モジュール
"""

import numpy as np
import scipy.signal
from numpy.typing import NDArray


class ReverbProcessor:
    """リバーブ処理クラス"""

    def __init__(
        self,
        sample_rate: int = 44100,
        room_size: float = 0.5,
        damping: float = 0.5,
        wet_dry_mix: float = 0.3,
    ):
        """
        Args:
            sample_rate: サンプルレート
            room_size: ルームサイズ (0.0-1.0)
            damping: ダンピング (0.0-1.0)
            wet_dry_mix: ウェット/ドライミックス (0.0-1.0)
        """
        self.sample_rate = sample_rate
        self.room_size = room_size
        self.damping = damping
        self.wet_dry_mix = wet_dry_mix

        # Freeverb アルゴリズムのパラメータ（簡易版）
        self.comb_delays = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116]
        self.allpass_delays = [225, 556, 441, 341]

    def process(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        リバーブを適用

        Args:
            audio: 音声データ

        Returns:
            NDArray[np.float32]: リバーブ適用後の音声
        """
        # シンプルなリバーブ実装
        # 実際のFreeverbアルゴリズムはより複雑

        wet = np.zeros_like(audio)

        # Comb filters
        for delay in self.comb_delays:
            delay_samples = int(delay * self.room_size * self.sample_rate / 44100)
            if delay_samples < len(audio):
                delayed = np.pad(audio, (delay_samples, 0))[: len(audio)]
                feedback = delayed * (1 - self.damping)
                wet += feedback * 0.125  # 8 comb filters

        # All-pass filters
        for delay in self.allpass_delays:
            delay_samples = int(delay * self.sample_rate / 44100)
            if delay_samples < len(audio):
                wet = self._allpass_filter(wet, delay_samples)

        # Mix wet and dry signals
        return audio * (1 - self.wet_dry_mix) + wet * self.wet_dry_mix

    def _allpass_filter(
        self, audio: NDArray[np.float32], delay_samples: int
    ) -> NDArray[np.float32]:
        """All-passフィルタ"""
        if delay_samples >= len(audio):
            return audio

        delayed = np.pad(audio, (delay_samples, 0))[: len(audio)]
        result = -audio + delayed + audio * 0.5
        return result.astype(np.float32)

    @classmethod
    def from_preset(cls, preset: str, sample_rate: int = 44100) -> "ReverbProcessor":
        """
        プリセットからリバーブを作成

        Args:
            preset: プリセット名 ('hall', 'room', 'plate', 'spring')
            sample_rate: サンプルレート

        Returns:
            ReverbProcessor: リバーブプロセッサー
        """
        presets = {
            "hall": {"room_size": 0.8, "damping": 0.3, "wet_dry_mix": 0.3},
            "room": {"room_size": 0.5, "damping": 0.5, "wet_dry_mix": 0.2},
            "plate": {"room_size": 0.6, "damping": 0.6, "wet_dry_mix": 0.25},
            "spring": {"room_size": 0.3, "damping": 0.7, "wet_dry_mix": 0.15},
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}")

        return cls(sample_rate=sample_rate, **presets[preset])


class DelayProcessor:
    """ディレイ処理クラス"""

    def __init__(
        self,
        sample_rate: int = 44100,
        delay_time_ms: float = 250.0,
        feedback: float = 0.5,
        mix: float = 0.3,
        ping_pong: bool = False,
    ):
        """
        Args:
            sample_rate: サンプルレート
            delay_time_ms: ディレイタイム（ミリ秒）
            feedback: フィードバック量 (0.0-1.0)
            mix: ミックス量 (0.0-1.0)
            ping_pong: ピンポンディレイモード
        """
        self.sample_rate = sample_rate
        self.delay_time_ms = delay_time_ms
        self.feedback = np.clip(feedback, 0.0, 0.95)  # 発振防止
        self.mix = mix
        self.ping_pong = ping_pong

        self.delay_samples = int(delay_time_ms * sample_rate / 1000)
        self.delay_buffer: NDArray[np.float32] | None = None

    def process(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        ディレイを適用

        Args:
            audio: 音声データ

        Returns:
            NDArray[np.float32]: ディレイ適用後の音声
        """
        # ディレイサンプルが0の場合は元の音声とミックスのみ
        if self.delay_samples == 0:
            return audio * (1.0 + self.mix)
        
        # ディレイバッファの初期化
        if self.delay_buffer is None or len(self.delay_buffer) != self.delay_samples:
            self.delay_buffer = np.zeros(self.delay_samples).astype(np.float32)

        output = np.zeros_like(audio)

        for i in range(len(audio)):
            # ディレイバッファから読み込み
            delayed = self.delay_buffer[0]

            # 出力を計算
            output[i] = audio[i] + delayed * self.mix

            # ディレイバッファを更新
            self.delay_buffer = np.roll(self.delay_buffer, -1)
            self.delay_buffer[-1] = audio[i] + delayed * self.feedback

        return output

    def process_stereo(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        ステレオディレイを適用（ピンポン対応）

        Args:
            audio: モノラル音声データ

        Returns:
            NDArray[np.float32]: ステレオディレイ適用後の音声 (N, 2)
        """
        if not self.ping_pong:
            # 通常のステレオディレイ
            processed = self.process(audio)
            return np.column_stack([processed, processed])

        # ピンポンディレイ
        left_delay = self.delay_samples
        right_delay = int(self.delay_samples * 1.5)

        left = self._apply_delay(audio, left_delay, "left")
        right = self._apply_delay(audio, right_delay, "right")

        return np.column_stack([left, right])

    def _apply_delay(
        self, audio: NDArray[np.float32], delay_samples: int, channel: str
    ) -> NDArray[np.float32]:
        """チャンネル別ディレイ適用"""
        buffer = np.zeros(delay_samples)
        output = np.zeros_like(audio)

        feedback_scale = self.feedback * (0.7 if channel == "right" else 1.0)

        for i in range(len(audio)):
            delayed = buffer[0]
            output[i] = audio[i] + delayed * self.mix
            buffer = np.roll(buffer, -1)
            buffer[-1] = audio[i] + delayed * feedback_scale

        return output.astype(np.float32)

    @classmethod
    def from_tempo(
        cls, sample_rate: int, bpm: float, note_division: str, **kwargs
    ) -> "DelayProcessor":
        """
        テンポ同期ディレイを作成

        Args:
            sample_rate: サンプルレート
            bpm: テンポ（BPM）
            note_division: 音符分割 ('1/4', '1/8', '1/16', etc.)
            **kwargs: その他のパラメータ

        Returns:
            DelayProcessor: ディレイプロセッサー
        """
        divisions = {
            "1/4": 1.0,
            "1/8": 0.5,
            "1/16": 0.25,
            "1/8t": 0.333,  # Triplet
            "1/16t": 0.167,
        }

        if note_division not in divisions:
            raise ValueError(f"Unknown note division: {note_division}")

        # Calculate delay time in ms
        beat_time_ms = 60000 / bpm  # Time for one beat in ms
        delay_time_ms = beat_time_ms * divisions[note_division]

        return cls(sample_rate=sample_rate, delay_time_ms=delay_time_ms, **kwargs)


class ChorusProcessor:
    """コーラス処理クラス"""

    def __init__(
        self,
        sample_rate: int = 44100,
        rate_hz: float = 1.0,
        depth: float = 0.3,
        mix: float = 0.5,
        voices: int = 2,
    ):
        """
        Args:
            sample_rate: サンプルレート
            rate_hz: LFOレート（Hz）
            depth: 変調深度 (0.0-1.0)
            mix: ミックス量 (0.0-1.0)
            voices: ボイス数
        """
        self.sample_rate = sample_rate
        self.rate_hz = rate_hz
        self.depth = depth
        self.mix = mix
        self.voices = voices

        # LFO設定
        self.base_delay_ms = 20.0
        self.max_delay_ms = 40.0

    def process(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        コーラスを適用

        Args:
            audio: 音声データ

        Returns:
            NDArray[np.float32]: コーラス適用後の音声
        """
        output = audio.copy()

        for voice in range(self.voices):
            # 各ボイスで異なる位相のLFO
            phase_offset = 2 * np.pi * voice / self.voices

            # LFO生成
            t = np.arange(len(audio)) / self.sample_rate
            lfo = np.sin(2 * np.pi * self.rate_hz * t + phase_offset)

            # 可変ディレイ
            delay_ms = self.base_delay_ms + self.depth * self.max_delay_ms * lfo
            delay_samples = delay_ms * self.sample_rate / 1000

            # 線形補間による可変ディレイ
            delayed = self._variable_delay(audio, delay_samples)

            output += delayed * self.mix / self.voices

        return output

    def _variable_delay(
        self, audio: NDArray[np.float32], delay_samples: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """可変ディレイの適用"""
        output = np.zeros_like(audio)
        max_delay = int(np.max(delay_samples)) + 1

        # 循環バッファ
        buffer = np.zeros(max_delay)
        write_idx = 0

        for i in range(len(audio)):
            # 現在のディレイ量
            delay = delay_samples[i]

            # 読み込み位置（線形補間）
            read_idx = write_idx - delay
            if read_idx < 0:
                read_idx += max_delay

            # 線形補間
            idx_int = int(read_idx)
            frac = read_idx - idx_int

            sample1 = buffer[idx_int % max_delay]
            sample2 = buffer[(idx_int + 1) % max_delay]

            output[i] = sample1 * (1 - frac) + sample2 * frac

            # バッファに書き込み
            buffer[write_idx] = audio[i]
            write_idx = (write_idx + 1) % max_delay

        return output


class DeesserProcessor:
    """ディエッサー処理クラス"""

    def __init__(
        self,
        sample_rate: int = 44100,
        frequency: float = 6000.0,
        threshold_db: float = -30.0,
        ratio: float = 4.0,
    ):
        """
        Args:
            sample_rate: サンプルレート
            frequency: 検出周波数
            threshold_db: 閾値（dB）
            ratio: 圧縮比
        """
        self.sample_rate = sample_rate
        self.frequency = frequency
        self.threshold_db = threshold_db
        self.ratio = ratio

    def process(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        ディエッシングを適用

        Args:
            audio: 音声データ

        Returns:
            NDArray[np.float32]: ディエッシング後の音声
        """
        # 高周波成分を抽出
        sos = scipy.signal.butter(
            4, self.frequency, btype="highpass", fs=self.sample_rate, output="sos"
        )
        high_freq = scipy.signal.sosfilt(sos, audio)

        # エンベロープ検出
        envelope = np.abs(scipy.signal.hilbert(high_freq))

        # スムージング
        smooth_samples = int(0.01 * self.sample_rate)  # 10ms
        envelope = scipy.signal.filtfilt(np.ones(smooth_samples) / smooth_samples, 1, envelope)

        # ゲインリダクション計算
        threshold = 10 ** (self.threshold_db / 20)
        gain_reduction = np.ones_like(audio)

        mask = envelope > threshold
        if np.any(mask):
            over_threshold = envelope[mask] / threshold
            gain_reduction[mask] = 1 / (1 + (self.ratio - 1) * (over_threshold - 1))

        # 高周波成分のみに適用
        processed_high = high_freq * gain_reduction

        # 低周波成分と合成
        low_freq = audio - high_freq

        result = low_freq + processed_high
        return result.astype(np.float32)


class StereoProcessor:
    """ステレオ処理クラス"""

    def __init__(self, sample_rate: int = 44100):
        """
        Args:
            sample_rate: サンプルレート
        """
        self.sample_rate = sample_rate

    def widen(self, stereo: NDArray[np.float32], width: float = 1.5) -> NDArray[np.float32]:
        """
        ステレオ幅を拡張

        Args:
            stereo: ステレオ音声 (N, 2)
            width: 幅 (1.0=原音, >1.0=広げる, <1.0=狭める)

        Returns:
            NDArray[np.float32]: 処理後のステレオ音声
        """
        if stereo.ndim != 2 or stereo.shape[1] != 2:
            raise ValueError("Input must be stereo (N, 2)")

        # Mid/Side処理
        mid, side = self.to_mid_side(stereo)

        # Side成分を調整
        side = side * width

        # L/Rに戻す
        return self.from_mid_side(mid, side)

    def to_mid_side(
        self, stereo: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """L/R to Mid/Side変換"""
        mid = (stereo[:, 0] + stereo[:, 1]) / 2
        side = (stereo[:, 0] - stereo[:, 1]) / 2
        return mid, side

    def from_mid_side(
        self, mid: NDArray[np.float32], side: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Mid/Side to L/R変換"""
        left = mid + side
        right = mid - side
        return np.column_stack([left, right])

    def auto_pan(
        self, mono: NDArray[np.float32], rate_hz: float = 0.5, depth: float = 0.8
    ) -> NDArray[np.float32]:
        """
        オートパンを適用

        Args:
            mono: モノラル音声
            rate_hz: パンニングレート（Hz）
            depth: パンニング深度 (0.0-1.0)

        Returns:
            NDArray[np.float32]: ステレオ音声 (N, 2)
        """
        t = np.arange(len(mono)) / self.sample_rate

        # パンニングLFO（-1 to 1）
        pan = np.sin(2 * np.pi * rate_hz * t) * depth

        # パンニング係数
        left_gain = np.sqrt((1 - pan) / 2)
        right_gain = np.sqrt((1 + pan) / 2)

        # ステレオ化
        left = mono * left_gain
        right = mono * right_gain

        return np.column_stack([left, right])
