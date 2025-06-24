"""
CLIインターフェースのテスト
"""
import pytest
from click.testing import CliRunner
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf
from unittest.mock import patch, MagicMock

from automix.cli import main, validate_audio_file, parse_config


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
        
        with patch('automix.cli.process_audio') as mock_process:
            mock_process.return_value = None
            
            result = runner.invoke(main, [
                '-v', str(vocal_path),
                '-b', str(bgm_path),
                '-o', 'output.mp4'
            ])
            
            assert result.exit_code == 0
            assert mock_process.called
    
    def test_help_message(self, runner):
        """ヘルプメッセージのテスト"""
        result = runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert 'Auto mixing and video generation' in result.output
        assert '--vocal' in result.output
        assert '--bgm' in result.output
        assert '--output' in result.output
    
    def test_version_info(self, runner):
        """バージョン情報表示のテスト"""
        result = runner.invoke(main, ['--version'])
        
        assert result.exit_code == 0
        assert 'automix version' in result.output.lower()
    
    def test_missing_required_args(self, runner):
        """必須引数が不足している場合のテスト"""
        result = runner.invoke(main, ['-v', 'vocal.wav'])
        
        assert result.exit_code != 0
        assert 'Missing option' in result.output or 'Error' in result.output
    
    def test_nonexistent_file(self, runner):
        """存在しないファイルのテスト"""
        result = runner.invoke(main, [
            '-v', 'nonexistent.wav',
            '-b', 'also_nonexistent.wav',
            '-o', 'output.mp4'
        ])
        
        assert result.exit_code != 0
        assert 'not found' in result.output.lower() or 'does not exist' in result.output.lower()
    
    def test_audio_only_mode(self, runner, sample_audio_files):
        """音声のみ出力モードのテスト"""
        vocal_path, bgm_path = sample_audio_files
        
        with patch('automix.cli.process_audio') as mock_process:
            result = runner.invoke(main, [
                '-v', str(vocal_path),
                '-b', str(bgm_path),
                '-o', 'output.wav',
                '--audio-only'
            ])
            
            assert result.exit_code == 0
            assert mock_process.called
            
            # audio_onlyフラグが渡されていることを確認
            call_args = mock_process.call_args
            assert call_args[1].get('audio_only') is True
    
    def test_preset_selection(self, runner, sample_audio_files):
        """プリセット選択のテスト"""
        vocal_path, bgm_path = sample_audio_files
        
        presets = ['pop', 'rock', 'ballad']
        
        for preset in presets:
            with patch('automix.cli.process_audio') as mock_process:
                result = runner.invoke(main, [
                    '-v', str(vocal_path),
                    '-b', str(bgm_path),
                    '-o', f'output_{preset}.mp4',
                    '--preset', preset
                ])
                
                assert result.exit_code == 0
                assert mock_process.called
                
                # プリセットが渡されていることを確認
                call_args = mock_process.call_args
                assert call_args[1].get('preset') == preset
    
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
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = Path(f.name)
        
        try:
            with patch('automix.cli.process_audio') as mock_process:
                result = runner.invoke(main, [
                    '-v', str(vocal_path),
                    '-b', str(bgm_path),
                    '-o', 'output.mp4',
                    '--config', str(config_path)
                ])
                
                assert result.exit_code == 0
                assert mock_process.called
        finally:
            config_path.unlink()
    
    def test_verbose_mode(self, runner, sample_audio_files):
        """詳細出力モードのテスト"""
        vocal_path, bgm_path = sample_audio_files
        
        with patch('automix.cli.process_audio') as mock_process:
            result = runner.invoke(main, [
                '-v', str(vocal_path),
                '-b', str(bgm_path),
                '-o', 'output.mp4',
                '--verbose'
            ])
            
            assert result.exit_code == 0
            # 詳細なログが出力される
            assert len(result.output) > 100
    
    def test_parameter_overrides(self, runner, sample_audio_files):
        """パラメータオーバーライドのテスト"""
        vocal_path, bgm_path = sample_audio_files
        
        with patch('automix.cli.process_audio') as mock_process:
            result = runner.invoke(main, [
                '-v', str(vocal_path),
                '-b', str(bgm_path),
                '-o', 'output.mp4',
                '--vocal-volume', '+3',
                '--bgm-volume', '-2',
                '--reverb', 'room',
                '--denoise', 'strong'
            ])
            
            assert result.exit_code == 0
            assert mock_process.called
            
            # パラメータが正しく渡されていることを確認
            call_args = mock_process.call_args[1]
            assert call_args.get('vocal_volume') == 3
            assert call_args.get('bgm_volume') == -2
            assert call_args.get('reverb') == 'room'
            assert call_args.get('denoise') == 'strong'
    
    def test_lyrics_input(self, runner, sample_audio_files):
        """歌詞ファイル入力のテスト"""
        vocal_path, bgm_path = sample_audio_files
        
        # SRTファイルを作成
        srt_content = """1
00:00:00,000 --> 00:00:02,000
Test lyrics line 1
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            f.write(srt_content)
            lyrics_path = Path(f.name)
        
        try:
            with patch('automix.cli.process_audio') as mock_process:
                result = runner.invoke(main, [
                    '-v', str(vocal_path),
                    '-b', str(bgm_path),
                    '-o', 'output.mp4',
                    '--lyrics', str(lyrics_path)
                ])
                
                assert result.exit_code == 0
                assert mock_process.called
                
                # 歌詞ファイルが渡されていることを確認
                call_args = mock_process.call_args[1]
                assert call_args.get('lyrics') == str(lyrics_path)
        finally:
            lyrics_path.unlink()
    
    def test_video_template(self, runner, sample_audio_files):
        """ビデオテンプレート選択のテスト"""
        vocal_path, bgm_path = sample_audio_files
        
        templates = ['modern', 'classic', 'minimal', 'particle']
        
        for template in templates:
            with patch('automix.cli.process_audio') as mock_process:
                result = runner.invoke(main, [
                    '-v', str(vocal_path),
                    '-b', str(bgm_path),
                    '-o', f'output_{template}.mp4',
                    '--video-template', template
                ])
                
                assert result.exit_code == 0
                assert mock_process.called
                
                call_args = mock_process.call_args[1]
                assert call_args.get('video_template') == template
    
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
            batch_content = "\n".join([
                f"{v},{b},output_{i}.mp4"
                for i, (v, b) in enumerate(pairs)
            ])
            
            batch_file = Path(tmpdir) / "batch.txt"
            batch_file.write_text(batch_content)
            
            with patch('automix.cli.process_audio') as mock_process:
                result = runner.invoke(main, [
                    '--batch', str(batch_file)
                ])
                
                assert result.exit_code == 0
                # 3回処理が呼ばれる
                assert mock_process.call_count == 3


class TestValidation:
    """バリデーション関数のテスト"""
    
    def test_validate_audio_file(self):
        """音声ファイルバリデーションのテスト"""
        with tempfile.NamedTemporaryFile(suffix='.wav') as f:
            # 有効なファイル
            assert validate_audio_file(Path(f.name)) is True
        
        # 無効なファイル
        assert validate_audio_file(Path("nonexistent.wav")) is False
        assert validate_audio_file(Path("test.txt")) is False
    
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
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_yaml)
            config_path = Path(f.name)
        
        try:
            config = parse_config(config_path)
            
            assert config['audio']['target_lufs'] == -14
            assert config['audio']['sample_rate'] == 48000
            assert config['video']['resolution'] == '1920x1080'
            assert config['video']['fps'] == 30
        finally:
            config_path.unlink()