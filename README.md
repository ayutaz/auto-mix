# AutoMix - 歌ってみた自動ミックス＆動画生成ツール

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![CI](https://github.com/ayutaz/auto-mix/actions/workflows/ci.yml/badge.svg)](https://github.com/ayutaz/auto-mix/actions/workflows/ci.yml)

AutoMixは、歌ってみた音声（ボーカル）とBGM（カラオケ音源）を自動的にミックスし、さらに動画も自動生成するPythonツールです。

## 特徴

- 🎤 **自動ミックス**: ボーカルとBGMの音量バランスを自動調整
- 🎵 **プロ品質のエフェクト**: リバーブ、コンプレッサー、EQなど
- 📊 **音声解析**: ピッチ、テンポ、音量などを自動解析
- 🎬 **動画自動生成**: 音声に反応するビジュアライザー付き動画
- 🎯 **マスタリング**: ストリーミングサービス向けLUFS正規化
- 💻 **クロスプラットフォーム**: Windows/macOS/Linux対応

## インストール

### 前提条件

- Python 3.11以上
- FFmpeg（動画生成に必要）
- uv（Pythonパッケージマネージャー）

### uvのインストール

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

### AutoMixのインストール

```bash
# リポジトリをクローン
git clone https://github.com/ayutaz/auto-mix.git
cd auto-mix

# 依存関係をインストール
uv sync

# 開発モードでインストール
uv pip install -e .
```

## 使い方

### 基本的な使用方法

```bash
# シンプルなミックス
automix -v vocal.wav -b bgm.mp3 -o output.mp4

# 音声のみ出力
automix -v vocal.wav -b bgm.mp3 -o output.wav --audio-only

# プリセット使用
automix -v vocal.wav -b bgm.mp3 -o output.mp4 --preset pop
```

### 詳細オプション

```bash
automix \
  -v vocal.wav \           # ボーカルファイル
  -b bgm.mp3 \             # BGMファイル
  -o output.mp4 \          # 出力ファイル
  --vocal-volume +3 \      # ボーカル音量調整 (dB)
  --bgm-volume -2 \        # BGM音量調整 (dB)
  --reverb hall \          # リバーブタイプ
  --denoise medium \       # ノイズ除去レベル
  --video-template modern  # 動画テンプレート
```

### 設定ファイル

`config.yaml`で詳細な設定が可能です：

```yaml
audio:
  target_lufs: -14        # ターゲットラウドネス
  sample_rate: 48000      # サンプルレート
  reverb:
    type: hall
    mix: 0.2
  compressor:
    threshold: -20
    ratio: 4

video:
  resolution: 1920x1080
  fps: 30
  visualizer: spectrum
  background: gradient
```

## 機能詳細

### 音声処理

- **自動音量調整**: LUFSベースの音量正規化
- **ノイズ除去**: スペクトラルサブトラクション
- **エフェクト**: リバーブ、ディレイ、コンプレッサー、EQ
- **マスタリング**: マルチバンドコンプレッション、リミッティング

### 動画生成

- **ビジュアライザー**: 波形、スペクトラム、パーティクル
- **テキストオーバーレイ**: 曲名、アーティスト名表示
- **歌詞表示**: SRTファイル対応（カラオケスタイル可能）
- **背景**: グラデーション、画像、動的パターン

### プリセット

- **Pop**: 明るく、ボーカルが前面に
- **Rock**: パワフルでバランスの取れたミックス
- **Ballad**: 温かみのある、ボーカル重視のミックス

## 開発

### テスト実行

```bash
# 全テスト実行
uv run pytest

# カバレッジ付き
uv run pytest --cov=automix

# 特定のテストのみ
uv run pytest tests/test_audio_loader.py
```

### コードフォーマット

```bash
# フォーマットチェック
uv run ruff format --check .

# 自動フォーマット
uv run ruff format .

# リントチェック
uv run ruff check .
```

### 型チェック

```bash
uv run mypy automix
```

## GUI インターフェース

GUIインターフェースを使用するには、tkinterが必要です。

### macOS (Homebrew Python)
```bash
brew install python-tk@3.11
```

### Ubuntu/Debian
```bash
sudo apt-get install python3-tk
```

### Windows
Windowsでは通常、tkinterはPythonに含まれています。

### GUI起動方法
```bash
python -m automix.gui
```

## トラブルシューティング

### FFmpegが見つからない

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows
choco install ffmpeg
```

### 音声ファイルが読み込めない

サポートされているフォーマット：
- WAV
- MP3
- M4A
- FLAC
- OGG

### メモリ不足エラー

大きなファイルを処理する場合は、設定ファイルで処理パラメータを調整してください。

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 貢献

プルリクエストを歓迎します！大きな変更を加える場合は、まずissueを開いて変更内容について議論してください。

## 作者

- ayutaz ([@ayutaz](https://github.com/ayutaz))

## 謝辞

このプロジェクトは以下のオープンソースライブラリを使用しています：

- [librosa](https://librosa.org/) - 音楽・音声解析
- [moviepy](https://zulko.github.io/moviepy/) - 動画編集
- [pedalboard](https://spotify.github.io/pedalboard/) - 音声エフェクト
- その他多数の素晴らしいライブラリ