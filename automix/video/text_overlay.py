"""
テキストオーバーレイモジュール
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class TextStyle:
    """テキストスタイル設定"""

    font_family: str = "system"
    font_size: int = 48
    font_color: tuple[int, int, int] = (255, 255, 255)
    font_thickness: int = 2
    background_color: tuple[int, int, int, int] | None = None
    outline_color: tuple[int, int, int] | None = None
    outline_thickness: int = 2
    shadow_offset: tuple[int, int] = (2, 2)
    shadow_color: tuple[int, int, int] = (0, 0, 0)


class TextOverlay:
    """テキストオーバーレイクラス"""

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        font_size: int = 48,
        font_color: tuple[int, int, int] = (255, 255, 255),
        background_color: tuple[int, int, int, int] | None = None,
        line_spacing: float = 1.2,
    ):
        """
        Args:
            width: 画像幅
            height: 画像高さ
            font_size: フォントサイズ
            font_color: フォント色 (B, G, R)
            background_color: 背景色 (B, G, R, A)
            line_spacing: 行間隔
        """
        self.width = width
        self.height = height
        self.font_size = font_size
        self.font_color = font_color
        self.background_color = background_color
        self.line_spacing = line_spacing

        # OpenCVのフォント
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_size / 30.0
        self.font_thickness = max(1, int(font_size / 20))

    def add_text(
        self,
        frame: NDArray[np.uint8],
        text: str,
        position: tuple[int, int],
        alpha: float = 1.0,
        style: TextStyle | None = None,
    ) -> NDArray[np.uint8]:
        """
        テキストを追加

        Args:
            frame: 背景画像
            text: テキスト
            position: 位置 (x, y)
            alpha: 透明度 (0.0-1.0)
            style: テキストスタイル

        Returns:
            NDArray[np.uint8]: テキスト追加後の画像
        """
        if cv2 is None:
            return frame

        if style is None:
            style = TextStyle(font_size=self.font_size, font_color=self.font_color)

        result = frame.copy()

        # テキストサイズを取得
        font_scale = style.font_size / 30.0
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, font_scale, style.font_thickness
        )

        x, y = position

        # 背景を描画
        if style.background_color and len(style.background_color) == 4:
            bg_alpha = style.background_color[3] / 255.0 * alpha
            overlay = frame.copy()

            padding = 10
            cv2.rectangle(
                overlay,
                (x - padding, y - text_height - padding),
                (x + text_width + padding, y + baseline + padding),
                style.background_color[:3],
                -1,
            )

            result = cv2.addWeighted(result, 1 - bg_alpha, overlay, bg_alpha, 0)

        # 影を描画
        if style.shadow_offset != (0, 0):
            shadow_x = x + style.shadow_offset[0]
            shadow_y = y + style.shadow_offset[1]
            shadow_color = tuple(int(c * alpha * 0.5) for c in style.shadow_color)

            cv2.putText(
                result,
                text,
                (shadow_x, shadow_y),
                self.font,
                font_scale,
                shadow_color,
                style.font_thickness,
            )

        # アウトラインを描画
        if style.outline_color:
            cv2.putText(
                result,
                text,
                (x, y),
                self.font,
                font_scale,
                style.outline_color,
                style.font_thickness + style.outline_thickness * 2,
            )

        # メインテキストを描画
        text_color = tuple(int(c * alpha) for c in style.font_color)
        cv2.putText(result, text, (x, y), self.font, font_scale, text_color, style.font_thickness)

        return result

    def add_multiline_text(
        self,
        frame: NDArray[np.uint8],
        text: str,
        position: tuple[int, int],
        max_width: int | None = None,
        align: str = "left",
        style: TextStyle | None = None,
    ) -> NDArray[np.uint8]:
        """
        複数行テキストを追加

        Args:
            frame: 背景画像
            text: テキスト（改行含む）
            position: 開始位置 (x, y)
            max_width: 最大幅（自動折り返し）
            align: テキスト配置 ('left', 'center', 'right')
            style: テキストスタイル

        Returns:
            NDArray[np.uint8]: テキスト追加後の画像
        """
        if style is None:
            style = TextStyle(font_size=self.font_size, font_color=self.font_color)

        result = frame.copy()
        lines = text.split("\n")

        # 必要に応じて自動折り返し
        if max_width:
            wrapped_lines = []
            for line in lines:
                wrapped_lines.extend(self._wrap_text(line, max_width, style))
            lines = wrapped_lines

        # 各行を描画
        x, y = position
        line_height = int(style.font_size * self.line_spacing)

        for i, line in enumerate(lines):
            line_y = y + i * line_height

            # 配置を調整
            if align in ["center", "right"]:
                font_scale = style.font_size / 30.0
                (text_width, _), _ = cv2.getTextSize(
                    line, self.font, font_scale, style.font_thickness
                )

                if align == "center":
                    line_x = x - text_width // 2
                else:  # right
                    line_x = x - text_width
            else:
                line_x = x

            result = self.add_text(result, line, (line_x, line_y), style=style)

        return result

    def _wrap_text(self, text: str, max_width: int, style: TextStyle) -> list[str]:
        """テキストを折り返し"""
        words = text.split(" ")
        lines = []
        current_line: list[str] = []

        font_scale = style.font_size / 30.0

        for word in words:
            test_line = " ".join(current_line + [word])
            (text_width, _), _ = cv2.getTextSize(
                test_line, self.font, font_scale, style.font_thickness
            )

            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)

        if current_line:
            lines.append(" ".join(current_line))

        return lines


class LyricsRenderer:
    """歌詞レンダラークラス"""

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        srt_file: Path | None = None,
        font_size: int = 60,
        font_color: tuple[int, int, int] = (255, 255, 255),
        highlight_color: tuple[int, int, int] = (255, 255, 0),
        karaoke_mode: bool = False,
    ):
        """
        Args:
            width: 画像幅
            height: 画像高さ
            srt_file: SRTファイルパス
            font_size: フォントサイズ
            font_color: フォント色
            highlight_color: ハイライト色
            karaoke_mode: カラオケモード
        """
        self.width = width
        self.height = height
        self.font_size = font_size
        self.font_color = font_color
        self.highlight_color = highlight_color
        self.karaoke_mode = karaoke_mode

        self.text_overlay = TextOverlay(width, height, font_size, font_color)

        # SRTファイルを解析
        self.subtitles: list[dict[str, Any]] = []
        if srt_file:
            self.load_srt(srt_file)

    def load_srt(self, srt_file: Path) -> None:
        """SRTファイルを読み込み"""
        self.subtitles = []

        with open(srt_file, encoding="utf-8") as f:
            content = f.read()

        # SRTエントリーを解析
        entries = content.strip().split("\n\n")

        for entry in entries:
            lines = entry.strip().split("\n")
            if len(lines) >= 3:
                # タイムコードを解析
                timecode = lines[1]
                start_time, end_time = self._parse_timecode(timecode)

                # テキストを結合
                text = "\n".join(lines[2:])

                self.subtitles.append({"start": start_time, "end": end_time, "text": text})

    def _parse_timecode(self, timecode: str) -> tuple[float, float]:
        """タイムコードを解析"""
        start_str, end_str = timecode.split(" --> ")

        def parse_time(time_str: str) -> float:
            parts = time_str.replace(",", ".").split(":")
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds

        return parse_time(start_str), parse_time(end_str)

    def get_lyrics_at_time(self, time_seconds: float) -> str | None:
        """指定時刻の歌詞を取得"""
        for subtitle in self.subtitles:
            if subtitle["start"] <= time_seconds <= subtitle["end"]:
                text = subtitle["text"]
                return str(text) if text is not None else None
        return None

    def render_karaoke(
        self, frame: NDArray[np.uint8], lyrics: str, progress: float, position: tuple[int, int]
    ) -> NDArray[np.uint8]:
        """
        カラオケスタイルで歌詞を描画

        Args:
            frame: 背景画像
            lyrics: 歌詞テキスト
            progress: 進行度 (0.0-1.0)
            position: 表示位置

        Returns:
            NDArray[np.uint8]: 描画後の画像
        """
        result = frame.copy()

        if not lyrics:
            return result

        # 文字数に基づいてハイライト位置を計算
        highlight_chars = int(len(lyrics) * progress)

        # ハイライト部分と通常部分を分ける
        highlighted = lyrics[:highlight_chars]
        normal = lyrics[highlight_chars:]

        x, y = position

        # ハイライト部分を描画
        if highlighted:
            style = TextStyle(
                font_size=self.font_size,
                font_color=self.highlight_color,
                outline_color=(0, 0, 0),
                outline_thickness=3,
            )
            result = self.text_overlay.add_text(result, highlighted, (x, y), style=style)

            # 幅を計算
            font_scale = self.font_size / 30.0
            (text_width, _), _ = cv2.getTextSize(
                highlighted, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
            )
            x += text_width

        # 通常部分を描画
        if normal:
            style = TextStyle(
                font_size=self.font_size,
                font_color=self.font_color,
                outline_color=(0, 0, 0),
                outline_thickness=3,
            )
            result = self.text_overlay.add_text(result, normal, (x, y), style=style)

        return result

    def render_lyrics_frame(
        self, frame: NDArray[np.uint8], time_seconds: float
    ) -> NDArray[np.uint8]:
        """
        指定時刻の歌詞を描画

        Args:
            frame: 背景画像
            time_seconds: 現在時刻（秒）

        Returns:
            NDArray[np.uint8]: 歌詞描画後の画像
        """
        lyrics = self.get_lyrics_at_time(time_seconds)

        if not lyrics:
            return frame

        # 画面下部中央に表示
        position = (self.width // 2, self.height - 100)

        if self.karaoke_mode:
            # カラオケモード：現在の字幕の進行度を計算
            for subtitle in self.subtitles:
                if subtitle["start"] <= time_seconds <= subtitle["end"]:
                    duration = subtitle["end"] - subtitle["start"]
                    elapsed = time_seconds - subtitle["start"]
                    progress = elapsed / duration if duration > 0 else 0

                    return self.render_karaoke(frame, subtitle["text"], progress, position)
        else:
            # 通常モード
            style = TextStyle(
                font_size=self.font_size,
                font_color=self.font_color,
                background_color=(0, 0, 0, 180),
                outline_color=(0, 0, 0),
                outline_thickness=2,
            )

            return self.text_overlay.add_multiline_text(
                frame,
                lyrics,
                position,
                max_width=int(self.width * 0.8),
                align="center",
                style=style,
            )

        return frame


class MetadataDisplay:
    """メタデータ表示クラス"""

    def __init__(self, width: int = 1920, height: int = 1080):
        """
        Args:
            width: 画像幅
            height: 画像高さ
        """
        self.width = width
        self.height = height
        self.text_overlay = TextOverlay(width, height)

    def render_metadata(
        self,
        frame: NDArray[np.uint8],
        title: str,
        artist: str,
        additional_info: dict | None = None,
        position: str = "top-left",
        fade_alpha: float = 1.0,
    ) -> NDArray[np.uint8]:
        """
        メタデータを描画

        Args:
            frame: 背景画像
            title: 曲名
            artist: アーティスト名
            additional_info: 追加情報
            position: 表示位置
            fade_alpha: フェード透明度

        Returns:
            NDArray[np.uint8]: 描画後の画像
        """
        result = frame.copy()

        # 位置を決定
        padding = 50
        positions = {
            "top-left": (padding, padding),
            "top-right": (self.width - padding, padding),
            "bottom-left": (padding, self.height - padding * 3),
            "bottom-right": (self.width - padding, self.height - padding * 3),
        }

        x, y = positions.get(position, positions["top-left"])

        # 右寄せの場合は調整が必要
        align = "right" if "right" in position else "left"

        # タイトルを描画
        title_style = TextStyle(
            font_size=72,
            font_color=(255, 255, 255),
            background_color=(0, 0, 0, 200),
            shadow_offset=(3, 3),
        )

        result = self.text_overlay.add_text(
            result, title, (x, y), alpha=fade_alpha, style=title_style
        )

        # アーティスト名を描画
        artist_style = TextStyle(
            font_size=48, font_color=(200, 200, 200), background_color=(0, 0, 0, 200)
        )

        result = self.text_overlay.add_text(
            result, artist, (x, y + 80), alpha=fade_alpha, style=artist_style
        )

        # 追加情報を描画
        if additional_info:
            info_y = y + 140
            info_style = TextStyle(font_size=36, font_color=(150, 150, 150))

            for key, value in additional_info.items():
                info_text = f"{key}: {value}"
                result = self.text_overlay.add_text(
                    result, info_text, (x, info_y), alpha=fade_alpha * 0.8, style=info_style
                )
                info_y += 40

        return result
