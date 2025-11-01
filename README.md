# Fitbit睡眠最適化ML予測サービス

Google Colabで開発したML予測ロジックをRender.comで24時間稼働させるWebサービスです。

## 機能

- Google SheetsのML予測キューを自動監視
- ランダムフォレスト回帰による睡眠の質予測
- HRV・RHR分析による回復スコア算出
- 睡眠傾向分析とプランBシミュレーション

## エンドポイント

- `GET /` - ヘルスチェック
- `GET /health` - サービスステータス
- `POST /predict` - ML予測実行

## 環境変数

- `GOOGLE_CREDENTIALS_JSON` - Googleサービスアカウントの認証情報（JSON形式）
- `PORT` - サービスのポート番号（Render.comが自動設定）

## デプロイ

Render.comで以下の設定でデプロイ：

- Environment: Python 3
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn main:app`
