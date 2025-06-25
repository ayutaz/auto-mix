"""
CLIインターフェースのテスト
"""
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from automix.cli import (
    display_analysis_results,
    generate_video,
    get_preset_settings,
    main,
    parse_config,
    process_audio,
    validate_audio_file,
)


class TestCLI:
    """CLIのテスト"""

    @pytest.fixture
    def runner(self):
        """Click CLIテストランナー"""
        return CliRunner()

    @pytest.fixture
    def sample_audio_files(self):
        """テスト用音声ファイルを作成"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # ボーカルファイル
            vocal_path = Path(tmpdir) / "vocal.wav"
            vocal_data = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
            sf.write(vocal_path, vocal_data, 44100)

            # BGMファイル
            bgm_path = Path(tmpdir) / "bgm.wav"
            bgm_data = np.sin(2 * np.pi * 220 * np.linspace(0, 1, 44100))
            sf.write(bgm_path, bgm_data, 44100)

            yield vocal_path, bgm_path

    def test_basic_command(self, runner, sample_audio_files):
        """基本的なコマンド実行のテスト"""
        vocal_path, bgm_path = sample_audio_files

        with patch("automix.cli.process_audio") as mock_process:
            mock_process.return_value = None

            result = runner.invoke(
                main, ["-v", str(vocal_path), "-b", str(bgm_path), "-o", "output.mp4"]
            )

            assert result.exit_code == 0
            assert mock_process.called

    def test_help_message(self, runner):
        """ヘルプメッセージのテスト"""
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "Auto mixing and video generation" in result.output
        assert "--vocal" in result.output
        assert "--bgm" in result.output
        assert "--output" in result.output

    def test_version_info(self, runner):
        """バージョン情報表示のテスト"""
        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert "version" in result.output.lower()

    def test_missing_required_args(self, runner):
        """必須引数が不足している場合のテスト"""
        result = runner.invoke(main, ["-v", "vocal.wav"])

        assert result.exit_code != 0
        assert "Missing option" in result.output or "Error" in result.output

    def test_nonexistent_file(self, runner):
        """存在しないファイルのテスト"""
        result = runner.invoke(
            main, ["-v", "nonexistent.wav", "-b", "also_nonexistent.wav", "-o", "output.mp4"]
        )

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "does not exist" in result.output.lower()

    def test_audio_only_mode(self, runner, sample_audio_files):
        """音声のみ出力モードのテスト"""
        vocal_path, bgm_path = sample_audio_files

        with patch("automix.cli.process_audio") as mock_process:
            result = runner.invoke(
                main,
                ["-v", str(vocal_path), "-b", str(bgm_path), "-o", "output.wav", "--audio-only"],
            )

            assert result.exit_code == 0
            assert mock_process.called

            # audio_onlyフラグが渡されていることを確認
            call_args = mock_process.call_args
            # call_args[0]は位置引数、call_args[1]はキーワード引数
            # settings引数内にaudio_onlyが含まれる
            assert mock_process.called
            args, kwargs = call_args
            settings = args[3] if len(args) > 3 else kwargs.get("settings", {})
            assert settings.get("audio_only") is True

    def test_preset_selection(self, runner, sample_audio_files):
        """プリセット選択のテスト"""
        vocal_path, bgm_path = sample_audio_files

        presets = ["pop", "rock", "ballad"]

        for preset in presets:
            with patch("automix.cli.process_audio") as mock_process:
                result = runner.invoke(
                    main,
                    [
                        "-v",
                        str(vocal_path),
                        "-b",
                        str(bgm_path),
                        "-o",
                        f"output_{preset}.mp4",
                        "--preset",
                        preset,
                    ],
                )

                assert result.exit_code == 0
                assert mock_process.called

                # プリセットが渡されていることを確認
                call_args = mock_process.call_args
                args, kwargs = call_args
                settings = args[3] if len(args) > 3 else kwargs.get("settings", {})
                # プリセットによって適切な設定が適用されているか確認
                if preset == "pop":
                    assert settings.get("vocal_volume") == 2.0
                    assert settings.get("bgm_volume") == -1.0
                    assert settings.get("reverb") == "room"
                elif preset == "rock":
                    assert settings.get("vocal_volume") == 0.0
                    assert settings.get("bgm_volume") == 0.0
                    assert settings.get("reverb") == "hall"
                elif preset == "ballad":
                    assert settings.get("vocal_volume") == 3.0
                    assert settings.get("bgm_volume") == -2.0
                    assert settings.get("reverb") == "plate"

    def test_config_file(self, runner, sample_audio_files):
        """設定ファイル読み込みのテスト"""
        vocal_path, bgm_path = sample_audio_files

        # 設定ファイルを作成
        config_content = """
audio:
  target_lufs: -16
  reverb:
    type: hall
    mix: 0.2
video:
  resolution: 1920x1080
  fps: 30
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = Path(f.name)

        try:
            with patch("automix.cli.process_audio") as mock_process:
                result = runner.invoke(
                    main,
                    [
                        "-v",
                        str(vocal_path),
                        "-b",
                        str(bgm_path),
                        "-o",
                        "output.mp4",
                        "--config",
                        str(config_path),
                    ],
                )

                assert result.exit_code == 0
                assert mock_process.called
        finally:
            config_path.unlink()

    def test_verbose_mode(self, runner, sample_audio_files):
        """詳細出力モードのテスト"""
        vocal_path, bgm_path = sample_audio_files

        with patch("automix.cli.process_audio") as mock_process:
            result = runner.invoke(
                main, ["-v", str(vocal_path), "-b", str(bgm_path), "-o", "output.mp4", "--verbose"]
            )

            assert result.exit_code == 0
            # Verbose モードでも最低限のメッセージは出力される
            assert "Successfully created" in result.output

    def test_parameter_overrides(self, runner, sample_audio_files):
        """パラメータオーバーライドのテスト"""
        vocal_path, bgm_path = sample_audio_files

        with patch("automix.cli.process_audio") as mock_process:
            result = runner.invoke(
                main,
                [
                    "-v",
                    str(vocal_path),
                    "-b",
                    str(bgm_path),
                    "-o",
                    "output.mp4",
                    "--vocal-volume",
                    "+3",
                    "--bgm-volume",
                    "-2",
                    "--reverb",
                    "room",
                    "--denoise",
                    "strong",
                ],
            )

            assert result.exit_code == 0
            assert mock_process.called

            # パラメータが正しく渡されていることを確認
            call_args = mock_process.call_args
            args, kwargs = call_args
            settings = args[3] if len(args) > 3 else kwargs.get("settings", {})
            assert settings.get("vocal_volume") == 3
            assert settings.get("bgm_volume") == -2
            assert settings.get("reverb") == "room"
            assert settings.get("denoise") == "strong"

    def test_lyrics_input(self, runner, sample_audio_files):
        """歌詞ファイル入力のテスト"""
        vocal_path, bgm_path = sample_audio_files

        # SRTファイルを作成
        srt_content = """1
00:00:00,000 --> 00:00:02,000
Test lyrics line 1
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(srt_content)
            lyrics_path = Path(f.name)

        try:
            with patch("automix.cli.process_audio") as mock_process:
                result = runner.invoke(
                    main,
                    [
                        "-v",
                        str(vocal_path),
                        "-b",
                        str(bgm_path),
                        "-o",
                        "output.mp4",
                        "--lyrics",
                        str(lyrics_path),
                    ],
                )

                assert result.exit_code == 0
                assert mock_process.called

                # 歌詞ファイルが渡されていることを確認
                call_args = mock_process.call_args
                args, kwargs = call_args
                settings = args[3] if len(args) > 3 else kwargs.get("settings", {})
                assert settings.get("lyrics") == str(lyrics_path)
        finally:
            lyrics_path.unlink()

    def test_video_template(self, runner, sample_audio_files):
        """ビデオテンプレート選択のテスト"""
        vocal_path, bgm_path = sample_audio_files

        templates = ["modern", "classic", "minimal", "particle"]

        for template in templates:
            with patch("automix.cli.process_audio") as mock_process:
                result = runner.invoke(
                    main,
                    [
                        "-v",
                        str(vocal_path),
                        "-b",
                        str(bgm_path),
                        "-o",
                        f"output_{template}.mp4",
                        "--video-template",
                        template,
                    ],
                )

                assert result.exit_code == 0
                assert mock_process.called

                call_args = mock_process.call_args
                args, kwargs = call_args
                settings = args[3] if len(args) > 3 else kwargs.get("settings", {})
                assert settings.get("video_template") == template

    @pytest.mark.skip(reason="Batch processing not implemented")
    def test_batch_processing(self, runner):
        """バッチ処理のテスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 複数の音声ファイルペアを作成
            pairs = []
            for i in range(3):
                vocal = Path(tmpdir) / f"vocal_{i}.wav"
                bgm = Path(tmpdir) / f"bgm_{i}.wav"

                sf.write(vocal, np.zeros(44100), 44100)
                sf.write(bgm, np.zeros(44100), 44100)

                pairs.append((vocal, bgm))

            # バッチファイルを作成
            batch_content = "\n".join([f"{v},{b},output_{i}.mp4" for i, (v, b) in enumerate(pairs)])

            batch_file = Path(tmpdir) / "batch.txt"
            batch_file.write_text(batch_content)

            with patch("automix.cli.process_audio") as mock_process:
                result = runner.invoke(main, ["--batch", str(batch_file)])

                assert result.exit_code == 0
                # 3回処理が呼ばれる
                assert mock_process.call_count == 3


class TestValidation:
    """バリデーション関数のテスト"""

    def test_validate_audio_file(self):
        """音声ファイルバリデーションのテスト"""
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            # 有効なファイル
            assert validate_audio_file(Path(f.name)) is True

        # 無効なファイル
        assert validate_audio_file(Path("nonexistent.wav")) is False
        assert validate_audio_file(Path("test.txt")) is False

    def test_validate_audio_file_supported_formats(self):
        """サポートされる音声形式のテスト"""
        supported_formats = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]

        with tempfile.TemporaryDirectory() as tmpdir:
            for ext in supported_formats:
                file_path = Path(tmpdir) / f"test{ext}"
                file_path.touch()
                assert validate_audio_file(file_path) is True

            # サポートされない形式
            unsupported_file = Path(tmpdir) / "test.txt"
            unsupported_file.touch()
            assert validate_audio_file(unsupported_file) is False

    def test_parse_config(self):
        """設定ファイル解析のテスト"""
        config_yaml = """
audio:
  target_lufs: -14
  sample_rate: 48000
video:
  resolution: 1920x1080
  fps: 30
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            config_path = Path(f.name)

        try:
            config = parse_config(config_path)

            assert config["audio"]["target_lufs"] == -14
            assert config["audio"]["sample_rate"] == 48000
            assert config["video"]["resolution"] == "1920x1080"
            assert config["video"]["fps"] == 30
        finally:
            config_path.unlink()

    def test_parse_config_invalid_yaml(self):
        """無効なYAMLファイルのテスト"""
        invalid_yaml = """invalid: yaml: content: ["""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(invalid_yaml)
            config_path = Path(f.name)

        try:
            with pytest.raises(Exception):
                parse_config(config_path)
        finally:
            config_path.unlink()

    def test_parse_config_empty_file(self):
        """空の設定ファイルのテスト"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            config_path = Path(f.name)

        try:
            config = parse_config(config_path)
            assert config is None or config == {}
        finally:
            config_path.unlink()


class TestPresetSettings:
    """プリセット設定のテスト"""

    def test_get_preset_settings_valid(self):
        """有効なプリセットのテスト"""
        presets = ["pop", "rock", "ballad"]

        for preset in presets:
            settings = get_preset_settings(preset)
            assert isinstance(settings, dict)
            assert "vocal_volume" in settings
            assert "bgm_volume" in settings
            assert "reverb" in settings

    def test_get_preset_settings_invalid(self):
        """無効なプリセットのテスト"""
        settings = get_preset_settings("invalid_preset")
        assert settings == {}

    def test_preset_settings_values(self):
        """プリセット値の妥当性テスト"""
        # Pop preset
        pop = get_preset_settings("pop")
        assert pop["vocal_volume"] == 2.0
        assert pop["bgm_volume"] == -1.0
        assert pop["reverb"] == "room"

        # Rock preset
        rock = get_preset_settings("rock")
        assert rock["vocal_volume"] == 0.0
        assert rock["bgm_volume"] == 0.0
        assert rock["reverb"] == "hall"

        # Ballad preset
        ballad = get_preset_settings("ballad")
        assert ballad["vocal_volume"] == 3.0
        assert ballad["bgm_volume"] == -2.0
        assert ballad["reverb"] == "plate"


class TestProcessAudio:
    """音声処理関数のテスト"""

    @pytest.fixture
    def mock_dependencies(self):
        """依存関係のモック"""
        with patch("automix.cli.AudioLoader") as mock_loader, patch(
            "automix.cli.AudioAnalyzer"
        ) as mock_analyzer, patch("automix.cli.MixProcessor") as mock_processor, patch(
            "automix.cli.ReverbProcessor"
        ) as mock_reverb, patch("automix.cli.MasteringProcessor") as mock_mastering:
            # モックの戻り値を設定
            mock_audio = type(
                "MockAudio",
                (),
                {"data": np.zeros(44100), "sample_rate": 44100, "duration": 1.0, "channels": 1},
            )

            mock_loader.return_value.load.return_value = mock_audio
            mock_analyzer.return_value.analyze_all.return_value = {
                "pitch": type("", (), {"median_pitch": 440.0}),
                "volume": type("", (), {"rms": 0.5, "lufs": -14.0}),
                "tempo": type("", (), {"tempo": 120.0}),
            }
            mock_processor.return_value.mix.return_value = np.zeros(44100)
            mock_reverb.from_preset.return_value.process.return_value = np.zeros(44100)
            mock_mastering.return_value.process.return_value = np.zeros(44100)

            yield {
                "loader": mock_loader,
                "analyzer": mock_analyzer,
                "processor": mock_processor,
                "reverb": mock_reverb,
                "mastering": mock_mastering,
            }

    def test_process_audio_basic(self, mock_dependencies):
        """基本的な音声処理のテスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            vocal_path = Path(tmpdir) / "vocal.wav"
            bgm_path = Path(tmpdir) / "bgm.wav"
            output_path = Path(tmpdir) / "output.wav"

            # ダミーファイルを作成
            vocal_path.touch()
            bgm_path.touch()

            settings = {"audio_only": True}

            with patch("automix.cli.Progress") as mock_progress:
                process_audio(
                    vocal_path, bgm_path, output_path, settings, mock_progress.return_value, False
                )

            # 各コンポーネントが呼ばれたことを確認
            assert mock_dependencies["loader"].return_value.load.call_count == 2
            assert mock_dependencies["analyzer"].return_value.analyze_all.call_count == 2
            assert mock_dependencies["processor"].return_value.mix.called
            assert mock_dependencies["mastering"].return_value.process.called

    def test_process_audio_with_effects(self, mock_dependencies):
        """エフェクト適用のテスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            vocal_path = Path(tmpdir) / "vocal.wav"
            bgm_path = Path(tmpdir) / "bgm.wav"
            output_path = Path(tmpdir) / "output.wav"

            vocal_path.touch()
            bgm_path.touch()

            settings = {
                "audio_only": True,
                "reverb": "hall",
                "vocal_volume": 3.0,
                "bgm_volume": -2.0,
            }

            with patch("automix.cli.Progress") as mock_progress:
                process_audio(
                    vocal_path, bgm_path, output_path, settings, mock_progress.return_value, False
                )

            # リバーブが適用されたことを確認
            mock_dependencies["reverb"].from_preset.assert_called_with("hall", sample_rate=44100)
            mock_dependencies["reverb"].from_preset.return_value.process.assert_called()

            # ボリューム調整が適用されたことを確認
            mix_call_args = mock_dependencies["processor"].return_value.mix.call_args
            assert mix_call_args[1]["vocal_gain_db"] == 3.0
            assert mix_call_args[1]["bgm_gain_db"] == -2.0

    def test_process_audio_video_mode(self, mock_dependencies):
        """ビデオ生成モードのテスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            vocal_path = Path(tmpdir) / "vocal.wav"
            bgm_path = Path(tmpdir) / "bgm.wav"
            output_path = Path(tmpdir) / "output.mp4"

            vocal_path.touch()
            bgm_path.touch()

            settings = {"audio_only": False}

            with patch("automix.cli.Progress") as mock_progress, patch(
                "automix.cli.generate_video"
            ) as mock_generate:
                process_audio(
                    vocal_path, bgm_path, output_path, settings, mock_progress.return_value, False
                )

            # ビデオ生成が呼ばれたことを確認
            mock_generate.assert_called_once()

    def test_process_audio_verbose_mode(self, mock_dependencies):
        """詳細出力モードのテスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            vocal_path = Path(tmpdir) / "vocal.wav"
            bgm_path = Path(tmpdir) / "bgm.wav"
            output_path = Path(tmpdir) / "output.wav"

            vocal_path.touch()
            bgm_path.touch()

            settings = {"audio_only": True}

            with (
                patch("automix.cli.Progress") as mock_progress,
                patch("automix.cli.console") as mock_console,
                patch("automix.cli.display_analysis_results") as mock_display,
                patch("automix.cli.sf.info") as mock_sf_info,
            ):
                # Mock soundfile info to avoid file read error
                mock_sf_info.return_value = type("", (), {"duration": 1.0})

                process_audio(
                    vocal_path, bgm_path, output_path, settings, mock_progress.return_value, True
                )

                # 詳細情報が表示されたことを確認
                assert mock_console.print.called
                assert mock_display.called


class TestGenerateVideo:
    """ビデオ生成関数のテスト"""

    def test_generate_video_basic(self):
        """基本的なビデオ生成のテスト"""
        with patch("automix.cli.VideoSettings") as mock_settings, patch(
            "automix.cli.VisualizerComposite"
        ) as mock_visualizer, patch("automix.cli.VideoEncoder") as mock_encoder, patch(
            "automix.cli.Progress"
        ) as mock_progress:
            # モックの設定
            mock_settings.return_value.fps = 30
            mock_visualizer.return_value.render_composite_frame.return_value = np.zeros(
                (1080, 1920, 3), dtype=np.uint8
            )

            audio = np.zeros(44100)  # 1秒の音声
            sample_rate = 44100
            output_path = Path("output.mp4")
            settings = {}

            generate_video(
                audio, sample_rate, output_path, settings, mock_progress.return_value, False
            )

            # エンコーダーが呼ばれたことを確認
            mock_encoder.return_value.encode.assert_called_once()

    def test_generate_video_with_settings(self):
        """設定付きビデオ生成のテスト"""
        with patch("automix.cli.VideoSettings") as mock_settings, patch(
            "automix.cli.VisualizerComposite"
        ) as mock_visualizer, patch("automix.cli.VideoEncoder") as mock_encoder, patch(
            "automix.cli.Progress"
        ) as mock_progress:
            mock_settings.return_value.fps = 60
            mock_settings.return_value.width = 3840
            mock_settings.return_value.height = 2160

            audio = np.zeros(44100 * 2)  # 2秒の音声
            sample_rate = 44100
            output_path = Path("output.mp4")
            settings = {"video_template": "particle"}

            generate_video(
                audio, sample_rate, output_path, settings, mock_progress.return_value, True
            )

            # 設定が適用されたことを確認
            mock_visualizer.assert_called_with(3840, 2160, 44100)


class TestDisplayAnalysisResults:
    """解析結果表示関数のテスト"""

    def test_display_analysis_results(self):
        """解析結果表示のテスト"""
        vocal_analysis = {
            "pitch": type("", (), {"median_pitch": 440.0}),
            "volume": type("", (), {"rms": 0.5, "lufs": -14.0}),
            "tempo": type("", (), {"tempo": 120.0}),
        }

        bgm_analysis = {
            "pitch": type("", (), {"median_pitch": 220.0}),
            "volume": type("", (), {"rms": 0.3, "lufs": -16.0}),
            "tempo": type("", (), {"tempo": 120.0}),
        }

        with patch("automix.cli.console") as mock_console, patch("automix.cli.Table") as mock_table:
            display_analysis_results(vocal_analysis, bgm_analysis)

            # テーブルが作成されたことを確認
            mock_table.assert_called_once()

            # 各行が追加されたことを確認
            table_instance = mock_table.return_value
            assert table_instance.add_row.call_count >= 3  # 最低3行（ピッチ、音量、テンポ）

            # コンソールに出力されたことを確認
            mock_console.print.assert_called_with(table_instance)


class TestErrorHandling:
    """エラーハンドリングのテスト"""

    @pytest.fixture
    def runner(self):
        """ "クリック CLIテストランナー"""
        return CliRunner()

    @pytest.fixture
    def sample_audio_files(self):
        """テスト用音声ファイルを作成"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # ボーカルファイル
            vocal_path = Path(tmpdir) / "vocal.wav"
            vocal_data = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
            sf.write(vocal_path, vocal_data, 44100)

            # BGMファイル
            bgm_path = Path(tmpdir) / "bgm.wav"
            bgm_data = np.sin(2 * np.pi * 220 * np.linspace(0, 1, 44100))
            sf.write(bgm_path, bgm_data, 44100)

            yield vocal_path, bgm_path

    def test_main_exception_handling(self, runner, sample_audio_files):
        """メイン関数の例外処理テスト"""
        vocal_path, bgm_path = sample_audio_files

        with patch("automix.cli.process_audio") as mock_process:
            mock_process.side_effect = Exception("Test error")

            result = runner.invoke(
                main, ["-v", str(vocal_path), "-b", str(bgm_path), "-o", "output.mp4"]
            )

            assert result.exit_code == 1
            assert "Error: Test error" in result.output

    def test_main_exception_verbose(self, runner, sample_audio_files):
        """詳細モードでの例外処理テスト"""
        vocal_path, bgm_path = sample_audio_files

        with patch("automix.cli.process_audio") as mock_process:
            mock_process.side_effect = ValueError("Detailed error")

            result = runner.invoke(
                main, ["-v", str(vocal_path), "-b", str(bgm_path), "-o", "output.mp4", "--verbose"]
            )

            assert result.exit_code == 1
            # 詳細モードではスタックトレースが表示される
            assert "Detailed error" in result.output


class TestIntegration:
    """統合テスト"""

    @pytest.fixture
    def runner(self):
        """ "クリック CLIテストランナー"""
        return CliRunner()

    def test_full_pipeline_audio_only(self, runner):
        """音声のみの完全なパイプラインテスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # テスト用音声ファイルを作成
            vocal_path = Path(tmpdir) / "vocal.wav"
            bgm_path = Path(tmpdir) / "bgm.wav"
            output_path = Path(tmpdir) / "output.wav"

            # 短い音声データを作成
            duration = 0.1  # 0.1秒
            sample_rate = 44100
            samples = int(duration * sample_rate)

            vocal_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
            bgm_data = np.sin(2 * np.pi * 220 * np.linspace(0, duration, samples))

            sf.write(vocal_path, vocal_data, sample_rate)
            sf.write(bgm_path, bgm_data, sample_rate)

            # 実際にコマンドを実行
            result = runner.invoke(
                main,
                [
                    "-v",
                    str(vocal_path),
                    "-b",
                    str(bgm_path),
                    "-o",
                    str(output_path),
                    "--audio-only",
                    "--preset",
                    "pop",
                ],
            )

            # エラーがあれば出力
            if result.exit_code != 0:
                print(result.output)

            # 成功を確認
            assert result.exit_code == 0
            assert output_path.exists()
