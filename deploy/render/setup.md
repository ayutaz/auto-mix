# Render.com Setup Guide for AutoMix

## 前提条件
- GitHubアカウント
- AutoMixのコードがGitHubにプッシュされている

## セットアップ手順

### 1. Renderアカウント作成
1. [Render.com](https://render.com) にアクセス
2. GitHubアカウントでサインアップ

### 2. 新しいWebサービスを作成
1. ダッシュボードで「New +」→「Web Service」をクリック
2. GitHubリポジトリを接続
3. `auto-mix`リポジトリを選択

### 3. サービス設定
- **Name**: automix（または好きな名前）
- **Region**: Oregon（US West）推奨
- **Branch**: main
- **Root Directory**: （空白のまま）
- **Runtime**: Docker
- **Instance Type**: Free

### 4. 環境変数設定
以下の環境変数を追加：
```
FLASK_ENV=production
```

### 5. デプロイ
「Create Web Service」をクリックしてデプロイ開始

## 制限事項（無料プラン）

### ⚠️ 重要な制限
1. **自動スリープ**: 15分間アクセスがないとサービスが停止
2. **再起動時間**: 停止後の最初のアクセスに約30-60秒かかる
3. **月間制限**: 750時間（継続稼働で約31日分）
4. **ストレージ**: 永続ストレージなし（アップロードファイルは再起動で消える）

### 対策
1. **ファイルストレージ**
   - Cloudflare R2（無料枠あり）を別途利用
   - または処理後すぐダウンロードを促す

2. **スリープ対策**
   - UptimeRobotなどで定期的にpingを送る（非推奨）
   - またはユーザーに遅延を説明

## 最適化のヒント

### 1. 軽量版Dockerfileを使用
```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
# ... 最小限の依存関係のみ
```

### 2. メモリ使用量を削減
- チャンク処理モードを有効化
- プレビューモード（30秒）をデフォルトに

### 3. タイムアウト対策
- 処理を5分以内に収める
- 長い処理は分割して実行

## トラブルシューティング

### ビルドが失敗する場合
1. Dockerfileのパスを確認
2. `.dockerignore`の設定を確認
3. ビルドログでエラーを確認

### メモリ不足エラー
- 無料プランは512MBメモリ制限
- 大きなファイルの処理は避ける
- ストリーミングモードを使用

### FFmpegが動作しない
- Dockerfileでffmpegのインストールを確認
- aptパッケージが正しくインストールされているか確認