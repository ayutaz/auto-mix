# 無料枠での運用最適化ガイド

## 推奨構成（完全無料）

### メインアプリケーション: Render.com
- **理由**: デプロイが簡単、GitHubとの連携が良い
- **制限**: 15分でスリープ、永続ストレージなし

### ファイルストレージ: Cloudflare R2
- **無料枠**: 10GB/月、100万リクエスト/月
- **帯域幅**: 無料（エグレス料金なし）

### 実装方法

## 1. Cloudflare R2の設定

### R2バケット作成
1. [Cloudflare Dashboard](https://dash.cloudflare.com)にログイン
2. R2 → Create Bucketで新規バケット作成
3. API トークンを生成

### 環境変数設定（Render.com）
```
AWS_ACCESS_KEY_ID=your_r2_access_key
AWS_SECRET_ACCESS_KEY=your_r2_secret_key
S3_BUCKET_NAME=your_bucket_name
S3_ENDPOINT_URL=https://YOUR_ACCOUNT_ID.r2.cloudflarestorage.com
```

## 2. アプリケーションの最適化

### メモリ効率化
```python
# automix/web/app.pyに追加
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MBに制限

# 処理時の設定
settings = {
    "chunk_processing": True,  # 必須
    "streaming": True,        # 必須
    "preview_mode": True,     # 30秒制限
}
```

### 自動削除の実装
```python
import threading
import time
from datetime import datetime, timedelta

def cleanup_old_files():
    """1時間以上古いファイルを削除"""
    while True:
        time.sleep(3600)  # 1時間ごと
        # S3から古いファイルを削除
        storage = get_storage_backend()
        # 実装...

# アプリ起動時に開始
cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()
```

## 3. ユーザー体験の改善

### 起動遅延の説明
```javascript
// static/app.js に追加
function showWakeupMessage() {
    if (isFirstAccess) {
        alert('サービスが休止状態から復帰しています。\n初回アクセスは30-60秒かかる場合があります。');
    }
}
```

### プログレス表示の改善
```javascript
// リアルタイムプログレス表示
function updateProgress() {
    fetch('/api/status')
        .then(res => res.json())
        .then(data => {
            progressBar.style.width = data.progress + '%';
            progressText.textContent = data.message;
        });
}
setInterval(updateProgress, 1000);
```

## 4. 制限事項の回避策

### ファイルサイズ制限
- 動画生成をオプション化（音声のみモードを推奨）
- 低解像度オプションを追加（720p、480p）
- 処理時間を表示して期待値を調整

### 処理時間制限（5分）
- 長い曲は分割処理
- バックグラウンドジョブの代わりにクライアント側でポーリング

### コスト削減のコツ
1. **プレビューモードをデフォルトに**
   - フル処理は明示的に選択
   
2. **キャッシュの活用**
   - 同じ設定の処理結果をR2に保存
   - ハッシュで重複チェック

3. **使用量モニタリング**
   ```python
   # 使用量を記録
   def track_usage():
       # CloudflareのAPIで使用量を取得
       # 制限に近づいたら警告
   ```

## 5. 代替構成

### A. Google Cloud Run + Firebase Storage
- **メリット**: Googleの統合環境
- **デメリット**: CPU制限が厳しい

### B. GitHub Actions + GitHub Pages
- **メリット**: 完全無料、CI/CD統合
- **デメリット**: 
  - 月2000分の制限
  - リアルタイム処理不可
  - バッチ処理のみ

実装例:
```yaml
# .github/workflows/process-audio.yml
on:
  workflow_dispatch:
    inputs:
      vocal_url:
        required: true
      bgm_url:
        required: true

jobs:
  process:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Process audio
        run: |
          # AutoMixで処理
          # 結果をGitHub Releasesにアップロード
```

## まとめ

### 推奨構成（シンプル＆無料）
1. **Render.com** - メインアプリ
2. **Cloudflare R2** - ファイルストレージ  
3. **制限を前提とした設計** - プレビューモード、サイズ制限

### ユーザーへの説明
- 無料版の制限を明記
- 処理時間の目安を表示
- 有料版へのアップグレードパスを用意

これらの最適化により、完全無料でAutoMixを運用できます。