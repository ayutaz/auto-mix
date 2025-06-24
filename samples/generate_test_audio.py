"""
テスト用の音声ファイルを生成するスクリプト
"""
import numpy as np
import soundfile as sf
from pathlib import Path


def generate_sine_wave(frequency: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """正弦波を生成"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    return 0.5 * np.sin(2 * np.pi * frequency * t)


def generate_vocal_simulation(duration: float = 10.0, sample_rate: int = 44100) -> np.ndarray:
    """ボーカルをシミュレートした音声を生成"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 基本周波数（歌声の基本ピッチ）
    fundamental = 440  # A4
    
    # 複数の倍音を含む
    vocal = (
        0.4 * np.sin(2 * np.pi * fundamental * t) +  # 基本音
        0.2 * np.sin(2 * np.pi * fundamental * 2 * t) +  # 2倍音
        0.1 * np.sin(2 * np.pi * fundamental * 3 * t) +  # 3倍音
        0.05 * np.sin(2 * np.pi * fundamental * 4 * t)   # 4倍音
    )
    
    # ビブラート効果を追加
    vibrato_freq = 5  # Hz
    vibrato_depth = 10  # Hz
    vibrato = vibrato_depth * np.sin(2 * np.pi * vibrato_freq * t)
    vocal_with_vibrato = 0.4 * np.sin(2 * np.pi * (fundamental + vibrato) * t)
    
    # エンベロープ（音量の時間変化）
    envelope = np.ones_like(t)
    # フレーズごとに音量を変える
    for i in range(0, int(duration), 2):
        start_idx = int(i * sample_rate)
        end_idx = int((i + 1.5) * sample_rate)
        if end_idx < len(envelope):
            # フェードイン
            fade_in_len = int(0.1 * sample_rate)
            envelope[start_idx:start_idx + fade_in_len] = np.linspace(0, 1, fade_in_len)
            # フェードアウト
            fade_out_len = int(0.2 * sample_rate)
            envelope[end_idx - fade_out_len:end_idx] = np.linspace(1, 0, fade_out_len)
            # 無音部分
            if end_idx < len(envelope):
                envelope[end_idx:min(end_idx + int(0.5 * sample_rate), len(envelope))] = 0
    
    return (vocal + vocal_with_vibrato * 0.5) * envelope


def generate_bgm_simulation(duration: float = 10.0, sample_rate: int = 44100) -> np.ndarray:
    """BGMをシミュレートした音声を生成"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # ベースライン（低音）
    bass_freq = 110  # A2
    bass = 0.3 * np.sin(2 * np.pi * bass_freq * t)
    
    # コード進行をシミュレート
    chord_progression = []
    chord_duration = 2.0  # 2秒ごとにコードチェンジ
    
    # Am - F - C - G のような進行
    chords = [
        [220, 261.63, 329.63],  # Am (A3, C4, E4)
        [174.61, 220, 261.63],  # F (F3, A3, C4)
        [261.63, 329.63, 392],  # C (C4, E4, G4)
        [196, 246.94, 293.66],  # G (G3, B3, D4)
    ]
    
    bgm = bass.copy()
    
    for i in range(int(duration / chord_duration)):
        chord_idx = i % len(chords)
        start_idx = int(i * chord_duration * sample_rate)
        end_idx = int((i + 1) * chord_duration * sample_rate)
        
        if end_idx > len(t):
            end_idx = len(t)
        
        t_segment = t[start_idx:end_idx]
        
        # コードの各音を追加
        for freq in chords[chord_idx]:
            bgm[start_idx:end_idx] += 0.15 * np.sin(2 * np.pi * freq * (t_segment - t[start_idx]))
    
    # ドラムパターンをシミュレート（クリック音）
    drum_interval = int(sample_rate * 0.5)  # 0.5秒ごと
    for i in range(0, len(bgm), drum_interval):
        if i + 100 < len(bgm):
            # キック
            bgm[i:i+100] += 0.2 * np.exp(-np.linspace(0, 50, 100)) * np.sin(2 * np.pi * 60 * np.linspace(0, 0.01, 100))
            # ハイハット
            if (i // drum_interval) % 2 == 1:
                bgm[i:i+50] += 0.05 * np.random.normal(0, 1, 50) * np.exp(-np.linspace(0, 100, 50))
    
    return bgm * 0.8  # 全体的に音量を調整


def main():
    """メイン処理"""
    # サンプルディレクトリを作成
    samples_dir = Path(__file__).parent
    samples_dir.mkdir(exist_ok=True)
    
    # サンプルレート
    sample_rate = 44100
    
    # 短いテスト用（5秒）
    print("Generating short test samples (5 seconds)...")
    
    # ボーカルシミュレーション
    vocal_short = generate_vocal_simulation(5.0, sample_rate)
    sf.write(samples_dir / "vocal_short.wav", vocal_short, sample_rate)
    
    # BGMシミュレーション
    bgm_short = generate_bgm_simulation(5.0, sample_rate)
    sf.write(samples_dir / "bgm_short.wav", bgm_short, sample_rate)
    
    # 長いテスト用（30秒）
    print("Generating long test samples (30 seconds)...")
    
    vocal_long = generate_vocal_simulation(30.0, sample_rate)
    sf.write(samples_dir / "vocal_long.wav", vocal_long, sample_rate)
    
    bgm_long = generate_bgm_simulation(30.0, sample_rate)
    sf.write(samples_dir / "bgm_long.wav", bgm_long, sample_rate)
    
    # シンプルなテスト音声（正弦波）
    print("Generating simple test tones...")
    
    # 440Hz (A4) - 1秒
    tone_a4 = generate_sine_wave(440, 1.0, sample_rate)
    sf.write(samples_dir / "tone_440hz.wav", tone_a4, sample_rate)
    
    # 1kHz - 1秒
    tone_1k = generate_sine_wave(1000, 1.0, sample_rate)
    sf.write(samples_dir / "tone_1khz.wav", tone_1k, sample_rate)
    
    # 無音ファイル
    silence = np.zeros(sample_rate * 2)  # 2秒
    sf.write(samples_dir / "silence.wav", silence, sample_rate)
    
    print("Sample files generated successfully!")
    print(f"Location: {samples_dir}")
    
    # 生成したファイルのリスト
    generated_files = [
        "vocal_short.wav",
        "bgm_short.wav",
        "vocal_long.wav",
        "bgm_long.wav",
        "tone_440hz.wav",
        "tone_1khz.wav",
        "silence.wav"
    ]
    
    print("\nGenerated files:")
    for file in generated_files:
        file_path = samples_dir / file
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            print(f"  - {file} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()