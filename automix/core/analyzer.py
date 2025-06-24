"""
音声解析モジュール
"""
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import librosa
import scipy.signal
from numpy.typing import NDArray


@dataclass
class PitchAnalysis:
    """ピッチ解析結果"""
    pitches: NDArray[np.float32]
    confidences: NDArray[np.float32]
    median_pitch: float
    pitch_range: Tuple[float, float]
    vibrato_rate: Optional[float] = None
    vibrato_extent: Optional[float] = None


@dataclass
class VolumeAnalysis:
    """音量解析結果"""
    rms: float
    peak: float
    lufs: float
    dynamic_range: float
    envelope: NDArray[np.float32]
    peak_locations: List[int]


@dataclass
class SpectralAnalysis:
    """スペクトル解析結果"""
    spectral_centroid: float
    spectral_rolloff: float
    spectral_bandwidth: float
    mfcc: NDArray[np.float32]
    spectral_contrast: NDArray[np.float32]
    zero_crossing_rate: float


@dataclass
class TempoAnalysis:
    """テンポ解析結果"""
    tempo: float
    beats: NDArray[np.float32]
    downbeats: Optional[NDArray[np.float32]] = None
    tempo_stability: Optional[float] = None


class AudioAnalyzer:
    """音声解析クラス"""
    
    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        window_size: int = 2048,
        n_mfcc: int = 13
    ):
        """
        Args:
            sample_rate: サンプルレート
            hop_length: ホップ長
            window_size: ウィンドウサイズ
            n_mfcc: MFCC係数の数
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.window_size = window_size
        self.n_mfcc = n_mfcc
    
    def analyze_pitch(
        self,
        audio: NDArray[np.float32],
        fmin: float = 80.0,
        fmax: float = 2000.0
    ) -> PitchAnalysis:
        """
        ピッチを解析する
        
        Args:
            audio: 音声データ
            fmin: 最小周波数
            fmax: 最大周波数
            
        Returns:
            PitchAnalysis: ピッチ解析結果
        """
        # ピッチ検出
        pitches, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=fmin,
            fmax=fmax,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # NaNを除去して統計を計算
        valid_pitches = pitches[~np.isnan(pitches)]
        
        if len(valid_pitches) > 0:
            median_pitch = float(np.median(valid_pitches))
            pitch_range = (float(np.min(valid_pitches)), float(np.max(valid_pitches)))
            
            # ビブラート検出（簡易版）
            if len(valid_pitches) > 10:
                pitch_diff = np.diff(valid_pitches)
                vibrato_rate = float(np.std(pitch_diff))
                vibrato_extent = float(np.max(np.abs(pitch_diff)))
            else:
                vibrato_rate = None
                vibrato_extent = None
        else:
            median_pitch = 0.0
            pitch_range = (0.0, 0.0)
            vibrato_rate = None
            vibrato_extent = None
        
        return PitchAnalysis(
            pitches=pitches,
            confidences=voiced_probs,
            median_pitch=median_pitch,
            pitch_range=pitch_range,
            vibrato_rate=vibrato_rate,
            vibrato_extent=vibrato_extent
        )
    
    def analyze_volume(self, audio: NDArray[np.float32]) -> VolumeAnalysis:
        """
        音量を解析する
        
        Args:
            audio: 音声データ
            
        Returns:
            VolumeAnalysis: 音量解析結果
        """
        # RMS（実効値）
        rms = float(np.sqrt(np.mean(audio ** 2)))
        
        # ピーク値
        peak = float(np.max(np.abs(audio)))
        
        # エンベロープ
        envelope = np.abs(scipy.signal.hilbert(audio))
        
        # ピーク位置検出
        peaks, _ = scipy.signal.find_peaks(
            envelope,
            height=peak * 0.7,
            distance=self.sample_rate // 10
        )
        
        # LUFS（簡易計算）
        # 実際のLUFS計算は複雑なので、ここでは簡易版
        lufs = float(20 * np.log10(rms + 1e-10) - 0.691)
        
        # ダイナミックレンジ
        percentile_95 = float(np.percentile(np.abs(audio), 95))
        percentile_5 = float(np.percentile(np.abs(audio), 5))
        dynamic_range = float(20 * np.log10((percentile_95 + 1e-10) / (percentile_5 + 1e-10)))
        
        return VolumeAnalysis(
            rms=rms,
            peak=peak,
            lufs=lufs,
            dynamic_range=dynamic_range,
            envelope=envelope,
            peak_locations=peaks.tolist()
        )
    
    def analyze_spectrum(self, audio: NDArray[np.float32]) -> SpectralAnalysis:
        """
        スペクトルを解析する
        
        Args:
            audio: 音声データ
            
        Returns:
            SpectralAnalysis: スペクトル解析結果
        """
        # スペクトラルセントロイド
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        spectral_centroid = float(np.mean(spectral_centroids))
        
        # スペクトラルロールオフ
        spectral_rolloffs = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        spectral_rolloff = float(np.mean(spectral_rolloffs))
        
        # スペクトラルバンドウィズ
        spectral_bandwidths = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        spectral_bandwidth = float(np.mean(spectral_bandwidths))
        
        # MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length
        )
        
        # スペクトラルコントラスト
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # ゼロ交差率
        zero_crossing_rates = librosa.feature.zero_crossing_rate(
            audio,
            hop_length=self.hop_length
        )
        zero_crossing_rate = float(np.mean(zero_crossing_rates))
        
        return SpectralAnalysis(
            spectral_centroid=spectral_centroid,
            spectral_rolloff=spectral_rolloff,
            spectral_bandwidth=spectral_bandwidth,
            mfcc=mfcc,
            spectral_contrast=spectral_contrast,
            zero_crossing_rate=zero_crossing_rate
        )
    
    def analyze_tempo(self, audio: NDArray[np.float32]) -> TempoAnalysis:
        """
        テンポを解析する
        
        Args:
            audio: 音声データ
            
        Returns:
            TempoAnalysis: テンポ解析結果
        """
        # テンポとビート検出
        tempo, beats = librosa.beat.beat_track(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # ビート時刻を秒単位に変換
        beat_times = librosa.frames_to_time(
            beats,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # テンポの安定性（ビート間隔の標準偏差）
        if len(beat_times) > 1:
            beat_intervals = np.diff(beat_times)
            tempo_stability = float(1.0 / (1.0 + np.std(beat_intervals)))
        else:
            tempo_stability = None
        
        return TempoAnalysis(
            tempo=float(tempo),
            beats=beat_times,
            downbeats=None,  # TODO: ダウンビート検出の実装
            tempo_stability=tempo_stability
        )
    
    def analyze_all(self, audio: NDArray[np.float32]) -> Dict[str, Any]:
        """
        全ての解析を実行する
        
        Args:
            audio: 音声データ
            
        Returns:
            Dict[str, Any]: 全ての解析結果
        """
        return {
            'pitch': self.analyze_pitch(audio),
            'volume': self.analyze_volume(audio),
            'spectrum': self.analyze_spectrum(audio),
            'tempo': self.analyze_tempo(audio)
        }
    
    def get_summary(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """
        解析結果のサマリーを取得
        
        Args:
            analysis: 解析結果
            
        Returns:
            Dict[str, float]: サマリー
        """
        return {
            'median_pitch': analysis['pitch'].median_pitch,
            'rms': analysis['volume'].rms,
            'peak': analysis['volume'].peak,
            'lufs': analysis['volume'].lufs,
            'spectral_centroid': analysis['spectrum'].spectral_centroid,
            'tempo': analysis['tempo'].tempo,
            'tempo_stability': analysis['tempo'].tempo_stability or 0.0
        }