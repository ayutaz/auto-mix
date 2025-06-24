# 歌ってみた自動ミックス＆動画生成ツール 要求定義書（Python版）

## 1. 概要

### 1.1 プロジェクト名
Auto Mix - 歌ってみた自動ミックス＆動画生成ツール

### 1.2 目的
個人使用を前提とした、高品質な歌ってみた音声の自動ミックスと動画生成を行うPythonツールを開発する。処理速度と音質の両立を重視。

### 1.3 動作環境
- **対応OS**: Windows 10/11, macOS 11+, Linux (Ubuntu 20.04+)
- **Python**: 3.11以上 (uvで管理)
- **パッケージマネージャー**: uv
- **メモリ**: 8GB以上（16GB推奨）

## 2. コア機能

### 2.1 音声ミックス機能

#### 2.1.1 入力
- **ボーカルトラック**: WAV/MP3/M4A/FLAC対応
- **BGMトラック**: WAV/MP3/M4A/FLAC対応
- **リファレンス音源**（オプション）: 目標とするミックスバランスの参考

#### 2.1.2 音声解析・前処理
- **自動アライメント**: DTW（Dynamic Time Warping）によるボーカルとBGMの同期
- **ボーカル解析**:
  - ピッチ検出と補正（オプション）
  - 音量エンベロープ抽出
  - フォルマント解析
- **ノイズ除去**: スペクトラルサブトラクション

#### 2.1.3 ミックス処理
- **インテリジェント音量調整**:
  - LUFS（Loudness Units relative to Full Scale）基準
  - ダイナミックレンジ最適化
- **周波数帯域処理**:
  - マルチバンドコンプレッサー
  - ダイナミックEQ
  - ボーカルとBGMの周波数帯域分離
- **空間処理**:
  - ステレオイメージング
  - リバーブ（複数アルゴリズム選択可）
  - ディレイ
- **ボーカル特化処理**:
  - ディエッサー（歯擦音制御）
  - ブレスノイズ制御
  - オートチューン（オプション）

#### 2.1.4 マスタリング
- **最終段コンプレッション**: マルチバンドコンプレッサー
- **リミッター**: ピーク制御
- **ラウドネス正規化**: ストリーミングサービス基準（-14 LUFS）

### 2.2 動画生成機能

#### 2.2.1 ビジュアル要素
- **音声反応型ビジュアライザー**:
  - スペクトラムアナライザー
  - 波形表示
  - パーティクルエフェクト
- **背景オプション**:
  - グラデーション生成
  - 画像/動画インポート
  - プロシージャル生成パターン

#### 2.2.2 テキスト表示
- **メタデータ**:
  - 曲名、アーティスト名
  - 作詞作曲クレジット
- **歌詞表示**:
  - SRTファイル対応
  - カラオケ風ハイライト（オプション）
  - フェードイン/アウト効果

#### 2.2.3 出力
- **解像度**: 4K/1080p/720p
- **フレームレート**: 60/30fps
- **コーデック**: H.264/H.265

## 3. 技術仕様

### 3.1 依存関係管理（uv）

```toml
# pyproject.toml
[project]
name = "automix"
version = "0.1.0"
description = "Auto mixing and video generation for vocal covers"
requires-python = ">=3.11"

dependencies = [
    # 音声処理コア
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "librosa>=0.10.0",
    "soundfile>=0.12.0",
    "pydub>=0.25.0",
    
    # エフェクト処理
    "pedalboard>=0.8.0",
    "pyloudnorm>=0.1.0",
    
    # 動画生成
    "moviepy>=1.0.3",
    "opencv-python>=4.8.0",
    "pillow>=10.0.0",
    "matplotlib>=3.7.0",
    
    # ユーティリティ
    "click>=8.1.0",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
    "colorama>=0.4.6",  # Windows対応
]

[project.optional-dependencies]
gpu = [
    "cupy>=12.0.0",  # GPU acceleration
]

[project.scripts]
automix = "automix.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
]
```

### 3.2 クロスプラットフォーム対応

#### プラットフォーム別の考慮事項
```python
# core/platform.py
import platform
import sys
from pathlib import Path

class PlatformConfig:
    @staticmethod
    def get_cache_dir():
        system = platform.system()
        if system == "Windows":
            return Path.home() / "AppData" / "Local" / "AutoMix" / "cache"
        elif system == "Darwin":  # macOS
            return Path.home() / "Library" / "Caches" / "AutoMix"
        else:  # Linux
            return Path.home() / ".cache" / "automix"
    
    @staticmethod
    def get_ffmpeg_path():
        """プラットフォーム別のFFmpegパス取得"""
        # 自動ダウンロード or システムのFFmpegを使用
        pass
```

### 3.3 インストールと実行

```bash
# uvのインストール（まだの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# or
irm https://astral.sh/uv/install.ps1 | iex  # Windows PowerShell

# プロジェクトのセットアップ
git clone https://github.com/yourname/automix.git
cd automix

# 依存関係のインストールと仮想環境作成
uv sync

# 実行
uv run automix -v vocal.wav -b bgm.mp3 -o output.mp4

# または開発モード
uv pip install -e .
automix -v vocal.wav -b bgm.mp3 -o output.mp4
```

## 4. CLI インターフェース

### 4.1 基本コマンド
```bash
# シンプルモード（自動設定）
automix -v vocal.wav -b bgm.mp3 -o output.mp4

# 詳細設定モード
automix \
    -v vocal.wav \
    -b bgm.mp3 \
    -o output.mp4 \
    --reverb hall \
    --vocal-volume +2 \
    --bgm-volume -1 \
    --denoise strong \
    --video-template modern \
    --lyrics lyrics.srt

# プリセット使用
automix -v vocal.wav -b bgm.mp3 -o output.mp4 --preset pop

# 音声のみ出力
automix -v vocal.wav -b bgm.mp3 -o output.wav --audio-only
```

### 4.2 設定ファイル
```yaml
# config.yaml
audio:
  target_lufs: -14
  sample_rate: 48000
  bit_depth: 24
  reverb:
    type: "hall"
    mix: 0.15
    room_size: 0.8
  compressor:
    threshold: -20
    ratio: 4
    attack: 10
    release: 100
  
video:
  resolution: "1920x1080"
  fps: 30
  codec: "h264"
  bitrate: "8M"
  visualizer: "spectrum"
  background: "gradient"
  text_style:
    font_family: "system"  # クロスプラットフォーム対応
    size: 48
    color: "#FFFFFF"

platform:
  max_threads: -1  # auto-detect
  gpu_acceleration: auto
```

## 5. プロジェクト構成

```
auto_mix/
├── pyproject.toml          # uv設定ファイル
├── README.md              # 使用方法
├── .python-version        # Python バージョン指定
├── automix/
│   ├── __init__.py
│   ├── cli.py            # CLIエントリーポイント
│   ├── core/
│   │   ├── __init__.py
│   │   ├── audio_loader.py    # 音声ファイル読み込み
│   │   ├── analyzer.py        # 音声解析
│   │   ├── processor.py       # ミックス処理
│   │   ├── effects.py         # エフェクト適用
│   │   ├── mastering.py       # マスタリング
│   │   └── platform.py        # プラットフォーム別設定
│   ├── video/
│   │   ├── __init__.py
│   │   ├── visualizer.py      # ビジュアライザー
│   │   ├── text_overlay.py    # テキスト処理
│   │   └── encoder.py         # 動画エンコード
│   └── utils/
│       ├── __init__.py
│       ├── dsp.py            # DSPユーティリティ
│       ├── ffmpeg.py         # FFmpeg ラッパー
│       └── logger.py         # ロギング
├── config/
│   └── default.yaml      # デフォルト設定
├── presets/              # プリセット設定
│   ├── rock.yaml
│   ├── pop.yaml
│   └── ballad.yaml
└── tests/               # テストコード
    ├── __init__.py
    ├── test_audio.py
    └── test_video.py
```

## 6. 性能目標

- **処理速度**: 5分の楽曲を2分以内で処理完了
- **メモリ使用**: 最大4GB以内
- **CPU使用率**: マルチコア対応で効率的な並列処理
- **音質**: プロ仕様のミックスに近い品質

## 7. プラットフォーム別の注意点

### Windows
- FFmpegの自動ダウンロード対応
- パス区切り文字の自動変換
- コンソール出力のUnicode対応

### macOS
- Apple Silicon最適化
- Core Audioとの統合（オプション）

### Linux
- 各ディストリビューションのパッケージマネージャーとの連携
- JACK Audio対応（オプション）

## 8. 今後の拡張予定

### Phase 1（初期実装）
- 基本的な自動ミックス機能
- シンプルなビジュアライザー付き動画生成
- クロスプラットフォーム対応

### Phase 2（機能拡張）
- AIベースの音源分離（ボーカル抽出）
- 機械学習による最適パラメータ推定
- GUIインターフェース（cross-platform: Kivy/Dear PyGui）

### Phase 3（高度な機能）
- リアルタイムプレビュー
- VST3プラグイン対応
- DAW連携（ReaperスクリプトAPIなど）