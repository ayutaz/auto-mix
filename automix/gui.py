"""
AutoMix GUI インターフェース
"""
import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from .cli import process_audio
from .core.optimizer import MemoryOptimizedProcessor


class AutoMixGUI:
    """AutoMixのGUIアプリケーション"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AutoMix - Auto Mixing Tool")
        self.root.geometry("800x600")
        
        # ファイルパス変数
        self.vocal_path = tk.StringVar()
        self.bgm_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.lyrics_path = tk.StringVar()
        
        # 設定変数
        self.preset = tk.StringVar(value="pop")
        self.vocal_volume = tk.DoubleVar(value=0.0)
        self.bgm_volume = tk.DoubleVar(value=0.0)
        self.reverb_type = tk.StringVar(value="")
        self.video_template = tk.StringVar(value="modern")
        self.audio_only = tk.BooleanVar(value=False)
        
        # パフォーマンス設定
        self.chunk_processing = tk.BooleanVar(value=False)
        self.streaming = tk.BooleanVar(value=False)
        self.preview_mode = tk.BooleanVar(value=False)
        
        # プログレスバー
        self.progress = tk.DoubleVar(value=0)
        self.progress_text = tk.StringVar(value="Ready")
        
        self._create_widgets()
        
    def _create_widgets(self):
        """ウィジェットを作成"""
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # ファイル選択セクション
        self._create_file_section(main_frame)
        
        # 設定セクション
        self._create_settings_section(main_frame)
        
        # パフォーマンス設定セクション
        self._create_performance_section(main_frame)
        
        # プログレスセクション
        self._create_progress_section(main_frame)
        
        # ボタンセクション
        self._create_button_section(main_frame)
        
        # グリッドの重み設定
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
    def _create_file_section(self, parent):
        """ファイル選択セクション"""
        file_frame = ttk.LabelFrame(parent, text="Files", padding="10")
        file_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Vocal file
        ttk.Label(file_frame, text="Vocal:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.vocal_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self._browse_vocal).grid(row=0, column=2)
        
        # BGM file
        ttk.Label(file_frame, text="BGM:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.bgm_path, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self._browse_bgm).grid(row=1, column=2)
        
        # Output file
        ttk.Label(file_frame, text="Output:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.output_path, width=50).grid(row=2, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self._browse_output).grid(row=2, column=2)
        
        # Lyrics file (optional)
        ttk.Label(file_frame, text="Lyrics (optional):").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.lyrics_path, width=50).grid(row=3, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self._browse_lyrics).grid(row=3, column=2)
        
    def _create_settings_section(self, parent):
        """設定セクション"""
        settings_frame = ttk.LabelFrame(parent, text="Settings", padding="10")
        settings_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Preset
        ttk.Label(settings_frame, text="Preset:").grid(row=0, column=0, sticky=tk.W, pady=2)
        preset_combo = ttk.Combobox(settings_frame, textvariable=self.preset, 
                                    values=["pop", "rock", "ballad"], state="readonly", width=20)
        preset_combo.grid(row=0, column=1, padx=5, sticky=tk.W)
        
        # Volume controls
        ttk.Label(settings_frame, text="Vocal Volume (dB):").grid(row=1, column=0, sticky=tk.W, pady=2)
        vocal_scale = ttk.Scale(settings_frame, from_=-20, to=20, variable=self.vocal_volume, 
                               orient=tk.HORIZONTAL, length=200)
        vocal_scale.grid(row=1, column=1, padx=5)
        ttk.Label(settings_frame, textvariable=self.vocal_volume, width=6).grid(row=1, column=2)
        
        ttk.Label(settings_frame, text="BGM Volume (dB):").grid(row=2, column=0, sticky=tk.W, pady=2)
        bgm_scale = ttk.Scale(settings_frame, from_=-20, to=20, variable=self.bgm_volume,
                             orient=tk.HORIZONTAL, length=200)
        bgm_scale.grid(row=2, column=1, padx=5)
        ttk.Label(settings_frame, textvariable=self.bgm_volume, width=6).grid(row=2, column=2)
        
        # Reverb
        ttk.Label(settings_frame, text="Reverb:").grid(row=3, column=0, sticky=tk.W, pady=2)
        reverb_combo = ttk.Combobox(settings_frame, textvariable=self.reverb_type,
                                   values=["", "hall", "room", "plate", "spring"], 
                                   state="readonly", width=20)
        reverb_combo.grid(row=3, column=1, padx=5, sticky=tk.W)
        
        # Video template
        ttk.Label(settings_frame, text="Video Template:").grid(row=4, column=0, sticky=tk.W, pady=2)
        template_combo = ttk.Combobox(settings_frame, textvariable=self.video_template,
                                     values=["modern", "classic", "minimal", "particle"],
                                     state="readonly", width=20)
        template_combo.grid(row=4, column=1, padx=5, sticky=tk.W)
        
        # Audio only checkbox
        ttk.Checkbutton(settings_frame, text="Audio Only (No Video)", 
                       variable=self.audio_only).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=5)
        
    def _create_performance_section(self, parent):
        """パフォーマンス設定セクション"""
        perf_frame = ttk.LabelFrame(parent, text="Performance Options", padding="10")
        perf_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Checkbutton(perf_frame, text="Chunk Processing (for large files)",
                       variable=self.chunk_processing).grid(row=0, column=0, sticky=tk.W, pady=2)
        
        ttk.Checkbutton(perf_frame, text="Streaming Mode (low memory usage)",
                       variable=self.streaming).grid(row=1, column=0, sticky=tk.W, pady=2)
        
        ttk.Checkbutton(perf_frame, text="Preview Mode (30 seconds only)",
                       variable=self.preview_mode).grid(row=2, column=0, sticky=tk.W, pady=2)
        
    def _create_progress_section(self, parent):
        """プログレスセクション"""
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding="10")
        progress_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # プログレスバー
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress,
                                           maximum=100, length=500)
        self.progress_bar.grid(row=0, column=0, columnspan=2, pady=5)
        
        # ステータステキスト
        ttk.Label(progress_frame, textvariable=self.progress_text).grid(row=1, column=0, columnspan=2)
        
    def _create_button_section(self, parent):
        """ボタンセクション"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=4, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Process", command=self._process,
                  style="Accent.TButton").grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Reset", command=self._reset).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).grid(row=0, column=2, padx=5)
        
    def _browse_vocal(self):
        """ボーカルファイルを選択"""
        filename = filedialog.askopenfilename(
            title="Select Vocal File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.flac"), ("All Files", "*.*")]
        )
        if filename:
            self.vocal_path.set(filename)
            
    def _browse_bgm(self):
        """BGMファイルを選択"""
        filename = filedialog.askopenfilename(
            title="Select BGM File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.flac"), ("All Files", "*.*")]
        )
        if filename:
            self.bgm_path.set(filename)
            
    def _browse_output(self):
        """出力ファイルを選択"""
        filename = filedialog.asksaveasfilename(
            title="Save Output As",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("WAV Audio", "*.wav"), ("All Files", "*.*")]
        )
        if filename:
            self.output_path.set(filename)
            
    def _browse_lyrics(self):
        """歌詞ファイルを選択"""
        filename = filedialog.askopenfilename(
            title="Select Lyrics File",
            filetypes=[("SRT Files", "*.srt"), ("All Files", "*.*")]
        )
        if filename:
            self.lyrics_path.set(filename)
            
    def _reset(self):
        """設定をリセット"""
        self.vocal_path.set("")
        self.bgm_path.set("")
        self.output_path.set("")
        self.lyrics_path.set("")
        self.preset.set("pop")
        self.vocal_volume.set(0.0)
        self.bgm_volume.set(0.0)
        self.reverb_type.set("")
        self.video_template.set("modern")
        self.audio_only.set(False)
        self.chunk_processing.set(False)
        self.streaming.set(False)
        self.preview_mode.set(False)
        self.progress.set(0)
        self.progress_text.set("Ready")
        
    def _validate_inputs(self) -> bool:
        """入力を検証"""
        if not self.vocal_path.get():
            messagebox.showerror("Error", "Please select a vocal file")
            return False
            
        if not self.bgm_path.get():
            messagebox.showerror("Error", "Please select a BGM file")
            return False
            
        if not self.output_path.get():
            messagebox.showerror("Error", "Please specify an output file")
            return False
            
        return True
        
    def _process(self):
        """処理を実行"""
        if not self._validate_inputs():
            return
            
        # 設定を準備
        settings = {
            "preset": self.preset.get(),
            "vocal_volume": self.vocal_volume.get(),
            "bgm_volume": self.bgm_volume.get(),
            "video_template": self.video_template.get(),
            "audio_only": self.audio_only.get(),
            "chunk_processing": self.chunk_processing.get(),
            "streaming": self.streaming.get(),
            "preview_mode": self.preview_mode.get(),
        }
        
        if self.reverb_type.get():
            settings["reverb"] = self.reverb_type.get()
            
        if self.lyrics_path.get():
            settings["lyrics"] = self.lyrics_path.get()
            
        # 別スレッドで処理を実行
        thread = threading.Thread(target=self._process_thread, args=(settings,))
        thread.daemon = True
        thread.start()
        
    def _process_thread(self, settings):
        """処理スレッド"""
        try:
            # GUIプログレスの更新
            class GUIProgress:
                def __init__(self, gui):
                    self.gui = gui
                    self.tasks = []
                    
                def add_task(self, description, total):
                    task_id = len(self.tasks)
                    self.tasks.append({"description": description, "total": total, "completed": 0})
                    self.gui.progress_text.set(description)
                    return task_id
                    
                def update(self, task_id, advance):
                    if 0 <= task_id < len(self.tasks):
                        self.tasks[task_id]["completed"] += advance
                        # 全体の進捗を計算
                        total_progress = sum(t["completed"] / t["total"] for t in self.tasks) / len(self.tasks) * 100
                        self.gui.progress.set(total_progress)
                        
            progress = GUIProgress(self)
            
            # 処理を実行
            process_audio(
                Path(self.vocal_path.get()),
                Path(self.bgm_path.get()),
                Path(self.output_path.get()),
                settings,
                progress,
                verbose=False
            )
            
            self.progress.set(100)
            self.progress_text.set("Complete!")
            messagebox.showinfo("Success", f"Processing complete!\nOutput saved to: {self.output_path.get()}")
            
        except Exception as e:
            self.progress_text.set("Error")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            
    def run(self):
        """アプリケーションを実行"""
        self.root.mainloop()


def main():
    """GUIアプリケーションのエントリーポイント"""
    app = AutoMixGUI()
    app.run()


if __name__ == "__main__":
    main()