"""
CLIインターフェース
"""

import sys
from pathlib import Path
from typing import Any

import click
import numpy as np
import soundfile as sf
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table

from . import __version__
from .core.analyzer import AudioAnalyzer
from .core.audio_loader import AudioFile, AudioLoader
from .core.effects import ReverbProcessor
from .core.mastering import MasteringProcessor
from .core.optimizer import MemoryOptimizedProcessor
from .core.processor import MixProcessor
from .plugins.base import NoiseGatePlugin, PitchShiftPlugin, PluginManager
from .plugins.custom_effects import (
    HarmonicExciterPlugin,
    StereoEnhancerPlugin,
    VintageWarmthPlugin,
    VocalEnhancerPlugin,
)
from .video.encoder import VideoEncoder, VideoSettings
from .video.visualizer import VisualizerComposite

console = Console()


@click.command()
@click.version_option(version=__version__, prog_name="automix")
@click.option("-v", "--vocal", required=True, type=click.Path(exists=True), help="Vocal audio file")
@click.option("-b", "--bgm", required=True, type=click.Path(exists=True), help="BGM audio file")
@click.option("-o", "--output", required=True, type=click.Path(), help="Output file path")
@click.option("--audio-only", is_flag=True, help="Output audio only (no video)")
@click.option("--preset", type=click.Choice(["pop", "rock", "ballad"]), help="Mixing preset")
@click.option("--vocal-volume", type=float, default=0.0, help="Vocal volume adjustment (dB)")
@click.option("--bgm-volume", type=float, default=0.0, help="BGM volume adjustment (dB)")
@click.option(
    "--reverb", type=click.Choice(["hall", "room", "plate", "spring"]), help="Reverb type"
)
@click.option("--denoise", type=click.Choice(["light", "medium", "strong"]), help="Denoise level")
@click.option(
    "--video-template",
    type=click.Choice(["modern", "classic", "minimal", "particle"]),
    default="modern",
    help="Video template",
)
@click.option("--lyrics", type=click.Path(exists=True), help="Lyrics file (SRT format)")
@click.option("--config", type=click.Path(exists=True), help="Configuration file")
@click.option("--verbose", is_flag=True, help="Verbose output")
@click.option("--version", is_flag=True, help="Show version")
@click.option("--chunk-processing", is_flag=True, help="Use chunk processing for large files")
@click.option("--streaming", is_flag=True, help="Use streaming processing (low memory usage)")
@click.option("--preview-mode", is_flag=True, help="Process only 30 seconds for preview")
@click.option(
    "--plugin-dir", type=click.Path(exists=True), help="Directory containing custom plugins"
)
@click.option("--list-plugins", is_flag=True, help="List available plugins and exit")
def main(
    vocal: str,
    bgm: str,
    output: str,
    audio_only: bool,
    preset: str | None,
    vocal_volume: float,
    bgm_volume: float,
    reverb: str | None,
    denoise: str | None,
    video_template: str,
    lyrics: str | None,
    config: str | None,
    verbose: bool,
    version: bool,
    chunk_processing: bool,
    streaming: bool,
    preview_mode: bool,
    plugin_dir: str | None,
    list_plugins: bool,
) -> None:
    """Auto mixing and video generation for vocal covers"""

    if version:
        from . import __version__

        console.print(f"automix version {__version__}")
        return

    # プラグインマネージャーを初期化
    plugin_manager = PluginManager()

    # 組み込みプラグインを登録
    plugin_manager.register_plugin(PitchShiftPlugin())
    plugin_manager.register_plugin(NoiseGatePlugin())
    plugin_manager.register_plugin(VintageWarmthPlugin())
    plugin_manager.register_plugin(VocalEnhancerPlugin())
    plugin_manager.register_plugin(StereoEnhancerPlugin())
    plugin_manager.register_plugin(HarmonicExciterPlugin())

    # カスタムプラグインディレクトリから読み込み
    if plugin_dir:
        plugin_manager.load_plugins_from_directory(Path(plugin_dir))

    # プラグインリストを表示
    if list_plugins:
        table = Table(title="Available Plugins")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Version", style="yellow")
        table.add_column("Description", style="white")

        for plugin_info in plugin_manager.list_plugins():
            plugin = plugin_manager.get_plugin(plugin_info["name"])
            if plugin:
                info = plugin.get_info()
                table.add_row(
                    info["name"],
                    info["type"],
                    info["version"],
                    info.get("description", "No description"),
                )

        console.print(table)
        return

    # 設定ファイルを読み込み
    settings = {}
    if config:
        settings = parse_config(Path(config))

    # プリセット適用
    if preset:
        settings.update(get_preset_settings(preset))

    # コマンドライン引数で上書き（デフォルト値以外の場合のみ）
    # Click doesn't provide a way to check if an option was explicitly provided,
    # so we check if the value is different from the default
    if vocal_volume != 0.0 or preset is None:
        settings["vocal_volume"] = vocal_volume
    if bgm_volume != 0.0 or preset is None:
        settings["bgm_volume"] = bgm_volume
    settings["audio_only"] = audio_only

    if reverb:
        settings["reverb"] = reverb
    if denoise:
        settings["denoise"] = denoise
    if lyrics:
        settings["lyrics"] = lyrics
    if video_template:
        settings["video_template"] = video_template

    # パフォーマンス最適化設定
    if chunk_processing:
        settings["chunk_processing"] = True
    if streaming:
        settings["streaming"] = True
    if preview_mode:
        settings["preview_mode"] = True

    # プラグインマネージャーを設定に追加
    settings["plugin_manager"] = plugin_manager

    # 処理実行
    try:
        with Progress(
            SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), console=console
        ) as progress:
            process_audio(Path(vocal), Path(bgm), Path(output), settings, progress, verbose)

        console.print(f"[green]✓[/green] Successfully created: {output}")

    except Exception as e:
        console.print(f"[red]✗[/red] Error: {str(e)}")
        if verbose:
            console.print_exception()
        sys.exit(1)


def process_audio(
    vocal_path: Path,
    bgm_path: Path,
    output_path: Path,
    settings: dict,
    progress: Progress,
    verbose: bool,
) -> None:
    """音声処理メイン"""

    # パフォーマンス最適化オプションの確認
    chunk_processing = settings.get("chunk_processing", False)
    streaming = settings.get("streaming", False)
    preview_mode = settings.get("preview_mode", False)

    # メモリ使用量の推定
    if verbose:
        vocal_info = sf.info(str(vocal_path))
        bgm_info = sf.info(str(bgm_path))
        total_duration = max(vocal_info.duration, bgm_info.duration)
        memory_estimate = MemoryOptimizedProcessor.estimate_memory_usage(total_duration)
        console.print(f"[yellow]Estimated memory usage: {memory_estimate:.1f} MB[/yellow]")

        if memory_estimate > 1000 and not (chunk_processing or streaming):
            console.print(
                "[yellow]Warning: Large file detected. Consider using --chunk-processing or --streaming[/yellow]"
            )

    # 1. 音声ファイル読み込み
    task = progress.add_task("[cyan]Loading audio files...", total=2)

    loader = AudioLoader(target_sample_rate=44100, normalize=True)

    # プレビューモードの場合は30秒のみ読み込む
    if preview_mode:
        vocal_audio = loader.load(vocal_path)
        bgm_audio = loader.load(bgm_path)

        # 30秒にトリミング
        vocal_audio.data = MemoryOptimizedProcessor.downsample_for_preview(
            vocal_audio.data, 30.0, vocal_audio.sample_rate
        )
        bgm_audio.data = MemoryOptimizedProcessor.downsample_for_preview(
            bgm_audio.data, 30.0, bgm_audio.sample_rate
        )

        if verbose:
            console.print("[yellow]Preview mode: Processing first 30 seconds only[/yellow]")
    else:
        vocal_audio = loader.load(vocal_path)
        bgm_audio = loader.load(bgm_path)

    progress.update(task, advance=2)

    if verbose:
        console.print(f"Vocal: {vocal_audio.duration:.1f}s, {vocal_audio.sample_rate}Hz")
        console.print(f"BGM: {bgm_audio.duration:.1f}s, {bgm_audio.sample_rate}Hz")

    # 2. 音声解析
    task = progress.add_task("[cyan]Analyzing audio...", total=2)

    analyzer = AudioAnalyzer(sample_rate=vocal_audio.sample_rate)
    vocal_analysis = analyzer.analyze_all(vocal_audio.data)
    progress.update(task, advance=1)

    bgm_analysis = analyzer.analyze_all(bgm_audio.data)
    progress.update(task, advance=1)

    if verbose:
        display_analysis_results(vocal_analysis, bgm_analysis)

    # 3. ミックス処理
    task = progress.add_task("[cyan]Mixing audio...", total=1)

    processor = MixProcessor(sample_rate=vocal_audio.sample_rate)
    mixed = processor.mix(
        vocal_audio.data,
        bgm_audio.data,
        vocal_gain_db=settings.get("vocal_volume", 0),
        bgm_gain_db=settings.get("bgm_volume", 0),
    )
    progress.update(task, advance=1)

    # 4. エフェクト適用
    if settings.get("reverb"):
        task = progress.add_task("[cyan]Applying effects...", total=1)

        reverb = ReverbProcessor.from_preset(
            settings["reverb"], sample_rate=vocal_audio.sample_rate
        )
        mixed = reverb.process(mixed)
        progress.update(task, advance=1)

    # 5. マスタリング
    task = progress.add_task("[cyan]Mastering...", total=1)

    mastering = MasteringProcessor(sample_rate=vocal_audio.sample_rate, target_lufs=-14)
    mastered = mastering.process(mixed)
    progress.update(task, advance=1)

    # 6. 出力
    if settings.get("audio_only"):
        # 音声のみ出力
        task = progress.add_task("[cyan]Saving audio...", total=1)
        loader.save(
            AudioFile(
                data=mastered,
                sample_rate=vocal_audio.sample_rate,
                duration=len(mastered) / vocal_audio.sample_rate,
                channels=1,
            ),
            output_path,
        )
        progress.update(task, advance=1)
    else:
        # 動画生成
        generate_video(mastered, vocal_audio.sample_rate, output_path, settings, progress, verbose)


def generate_video(
    audio: np.ndarray,
    sample_rate: int,
    output_path: Path,
    settings: dict,
    progress: Progress,
    verbose: bool,
) -> None:
    """動画生成"""

    # ビデオ設定
    video_settings = VideoSettings(width=1920, height=1080, fps=30)

    # ビジュアライザー初期化
    visualizer = VisualizerComposite(video_settings.width, video_settings.height, sample_rate)

    # フレーム生成
    total_frames = int(len(audio) / sample_rate * video_settings.fps)
    task = progress.add_task("[cyan]Generating frames...", total=total_frames)

    frames = []
    samples_per_frame = int(sample_rate / video_settings.fps)

    for i in range(0, len(audio), samples_per_frame):
        audio_chunk = audio[i : i + samples_per_frame]
        if len(audio_chunk) < samples_per_frame:
            break

        frame = visualizer.render_composite_frame(audio_chunk)
        frames.append(frame)
        progress.update(task, advance=1)

    # エンコード
    task = progress.add_task("[cyan]Encoding video...", total=1)

    encoder = VideoEncoder(video_settings)
    encoder.encode(frames, audio, output_path, sample_rate=sample_rate, show_progress=verbose)
    progress.update(task, advance=1)


def parse_config(config_path: Path) -> dict[Any, Any]:
    """設定ファイルを解析"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
        return config if config is not None else {}


def get_preset_settings(preset: str) -> dict[str, Any]:
    """プリセット設定を取得"""
    presets = {
        "pop": {"vocal_volume": 2.0, "bgm_volume": -1.0, "reverb": "room", "eq_preset": "bright"},
        "rock": {"vocal_volume": 0.0, "bgm_volume": 0.0, "reverb": "hall", "eq_preset": "neutral"},
        "ballad": {"vocal_volume": 3.0, "bgm_volume": -2.0, "reverb": "plate", "eq_preset": "warm"},
    }

    return presets.get(preset, {})


def display_analysis_results(vocal_analysis: dict[str, Any], bgm_analysis: dict[str, Any]) -> None:
    """解析結果を表示"""
    table = Table(title="Audio Analysis Results")
    table.add_column("Parameter", style="cyan")
    table.add_column("Vocal", style="green")
    table.add_column("BGM", style="yellow")

    # ピッチ
    table.add_row(
        "Median Pitch",
        f"{vocal_analysis['pitch'].median_pitch:.1f} Hz",
        f"{bgm_analysis['pitch'].median_pitch:.1f} Hz",
    )

    # 音量
    table.add_row(
        "RMS Level", f"{vocal_analysis['volume'].rms:.3f}", f"{bgm_analysis['volume'].rms:.3f}"
    )
    table.add_row(
        "LUFS", f"{vocal_analysis['volume'].lufs:.1f} dB", f"{bgm_analysis['volume'].lufs:.1f} dB"
    )

    # テンポ
    table.add_row(
        "Tempo",
        f"{vocal_analysis['tempo'].tempo:.1f} BPM",
        f"{bgm_analysis['tempo'].tempo:.1f} BPM",
    )

    console.print(table)


def validate_audio_file(file_path: Path) -> bool:
    """音声ファイルを検証"""
    if not file_path.exists():
        return False

    supported_formats = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    return file_path.suffix.lower() in supported_formats


if __name__ == "__main__":
    main()
