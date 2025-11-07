"""
Fitbit睡眠最適化ML予測サービス (v2 - 安定版)
Render.comで24時間稼働するFlaskアプリケーション

変更点:
- メモリ不足によるクラッシュを防ぐため、/predict が呼び出された際に
  キュー(ML予測キュー)から1件だけ(pendingの最初の1件)を処理するように変更。
- Cloud Scheduler (GCP) などで定期的に /predict を呼び出すことで、
  キューに溜まったジョブを1件ずつ着実に処理する設計。
"""

from flask import Flask, request, jsonify
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from datetime import datetime, timedelta
import os
import json
import time
import traceback # エラーログ出力用

app = Flask(__name__)

# =================================================================
# 設定
# =================================================================

# スプレッドシートIDを環境変数から取得（推奨）
SPREADSHEET_ID = os.environ.get('SPREADSHEET_ID', '1ZGgw8i9ecNb__f8M8PLJY33NV76dzL5dFhg-e6rOQdc')
PREDICTION_SHEET_NAME = '睡眠最適化予測'
QUEUE_SHEET_NAME = 'ML予測キュー'

# =================================================================
# Google Sheets認証
# =================================================================

def get_gspread_client():
    """Google Sheets APIクライアントを取得"""
    try:
        creds_json = os.environ.get('GOOGLE_CREDENTIALS_JSON')
        if not creds_json:
            print("❌ FATAL: GOOGLE_CREDENTIALS_JSON環境変数が設定されていません")
            raise Exception('GOOGLE_CREDENTIALS_JSON環境変数が設定されていません')
        
        creds_dict = json.loads(creds_json)
        creds = Credentials.from_service_account_info(
            creds_dict,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        return gspread.authorize(creds)
    except Exception as e:
        print(f"❌ 認証エラー: {e}")
        raise

# =================================================================
# ヘルパー関数（変更なし）
# =================================================================

def get_sheet_data_as_df(ss, sheet_name):
    """シート名からデータをPandas DataFrameとして取得"""
    try:
        ws = ss.worksheet(sheet_name)
        data = ws.get_all_values()
        if len(data) > 0:
            header = data[0]
            df = pd.DataFrame(data[1:], columns=header)
            return df
        else:
            return pd.DataFrame()
    except gspread.exceptions.WorksheetNotFound:
        print(f"  ⚠️ シートが見つかりません: {sheet_name}")
        return None
    except Exception as e:
        print(f"  ⚠️ データ取得エラー: {sheet_name}, {e}")
        return None

def define_sleep_quality(df):
    """睡眠の質を計算する (0-100)"""
    sleep_hours = df['minutesAsleep'] / 60
    time_score = -16 * (sleep_hours - 7.5)**2 + 100
    time_score = np.clip(time_score, 0, 100)
    
    efficiency_score = (df['efficiency'] / 85) * 100
    efficiency_score = np.clip(efficiency_score, 0, 100)
    
    deep_percent = (df['deep.minutes'] / df['minutesAsleep']).replace([np.inf, -np.inf], 0).fillna(0) * 100
    deep_score = (deep_percent / 15) * 100
    deep_score = np.clip(deep_score, 0, 100)
    
    quality = (efficiency_score * 0.5) + (time_score * 0.3) + (deep_score * 0.2)
    return quality.fillna(0)

def preprocess_data(ss, user_sheet_name):
    """ユーザーデータを読み込み、前処理と結合を行う"""
    sleep_df = get_sheet_data_as_df(ss, f"sleep_{user_sheet_name}")
    if sleep_df is None or sleep_df.empty:
        print(f"  [{user_sheet_name}] 睡眠データがありません。")
        return None
    
    sleep_df['dateOfSleep'] = pd.to_datetime(sleep_df['dateOfSleep'])
    
    num_cols = ['minutesAsleep', 'efficiency', 'deep.minutes',
                'rem.minutes', 'light.minutes', 'minutesToFallAsleep', 'timeInBed']
    for col in num_cols:
        sleep_df[col] = pd.to_numeric(sleep_df[col], errors='coerce').fillna(0)
    
    sleep_df['startTime'] = pd.to_datetime(sleep_df['startTime'])
    sleep_df['endTime'] = pd.to_datetime(sleep_df['endTime'])
    base_hour = 4
    bedtime_minutes = sleep_df['startTime'].dt.hour * 60 + sleep_df['startTime'].dt.minute
    bedtime_minutes = bedtime_minutes.apply(lambda x: x - 1440 if x > base_hour * 60 else x)
    sleep_df['bedtime_minutes'] = bedtime_minutes
    
    hrv_df = get_sheet_data_as_df(ss, f"hrv_{user_sheet_name}")
    rhr_df = get_sheet_data_as_df(ss, f"rhr_{user_sheet_name}")
    
    if hrv_df is not None and not hrv_df.empty:
        hrv_df['date'] = pd.to_datetime(hrv_df['date'])
        sleep_df = pd.merge(sleep_df, hrv_df[['date', 'dailyRmssd']], 
                           left_on='dateOfSleep', right_on='date', how='left')
        sleep_df['dailyRmssd'] = pd.to_numeric(sleep_df['dailyRmssd'], errors='coerce')
    
    if rhr_df is not None and not rhr_df.empty:
        rhr_df['date'] = pd.to_datetime(rhr_df['date'])
        sleep_df = pd.merge(sleep_df, rhr_df[['date', 'restingHeartRate']], 
                           left_on='dateOfSleep', right_on='date', how='left')
        sleep_df['restingHeartRate'] = pd.to_numeric(sleep_df['restingHeartRate'], errors='coerce')
    
    sleep_df['sleep_quality'] = define_sleep_quality(sleep_df)
    sleep_df = sleep_df.fillna(0)
    
    return sleep_df

def format_minutes_to_time(minutes):
    """分をHH:MM形式に変換"""
    if np.isnan(minutes): 
        return "N/A"
    minutes = float(minutes)
    if minutes < 0:
        minutes += 1440
    hour = int(minutes // 60)
    minute = int(minutes % 60)
    return f"{hour:02d}:{minute:02d}"

def calculate_recovery_score(df, today_hrv, today_rhr):
    """HRVとRHRから回復スコア（文字列）を算出"""
    if 'dailyRmssd' not in df.columns or 'restingHeartRate' not in df.columns:
        return "安定"
    
    hrv_baseline = df['dailyRmssd'].tail(30).median()
    rhr_baseline = df['restingHeartRate'].tail(30).median()
    
    if pd.isna(hrv_baseline) or pd.isna(rhr_baseline) or hrv_baseline == 0 or rhr_baseline == 0:
        return "安定"
    
    hrv_score = (today_hrv / hrv_baseline) * 50 if hrv_baseline > 0 else 50
    hrv_score = np.clip(hrv_score, 0, 100)
    
    rhr_score = (rhr_baseline / today_rhr) * 50 if today_rhr > 0 else 50
    rhr_score = np.clip(rhr_score, 0, 100)
    
    recovery_score = (hrv_score * 0.6) + (rhr_score * 0.4)
    
    score_val = int(recovery_score)
    if score_val > 65:
        return "良好"
    elif score_val < 35:
        return "注意"
    else:
        return "安定"

def analyze_trends(df):
    """過去7日間の傾向を分析"""
    if len(df) < 7:
        return "安定", "安定"
    
    hrv_7day_avg = df['dailyRmssd'].rolling(window=7).mean()
    deep_7day_avg = df['deep.minutes'].rolling(window=7).mean()
    
    valid_hrv_avg = hrv_7day_avg.dropna()
    valid_deep_avg = deep_7day_avg.dropna()
    
    if len(valid_hrv_avg) < 2:
        hrv_trend_val = 0
    else:
        hrv_trend_val = valid_hrv_avg.iloc[-1] - valid_hrv_avg.iloc[-2]
    
    if len(valid_deep_avg) < 2:
        deep_trend_val = 0
    else:
        deep_trend_val = valid_deep_avg.iloc[-1] - valid_deep_avg.iloc[-2]
    
    if hrv_trend_val > 2: 
        trend_hrv = "上昇傾向 (良い兆候)"
    elif hrv_trend_val < -2: 
        trend_hrv = "下降傾向 (要注意)"
    else: 
        trend_hrv = "安定"
    
    if deep_trend_val > 5: 
        trend_deep = "上昇傾向"
    elif deep_trend_val < -5: 
        trend_deep = "減少傾向"
    else: 
        trend_deep = "安定"
    
    return trend_hrv, trend_deep

def get_key_factor(model, features):
    """モデルから最も重要な要因を取得"""
    importances = model.feature_importances_
    feature_names = {
        'bedtime_minutes': '就寝時刻のズレ',
        'timeInBed': 'ベッドにいた時間',
        'dailyRmssd': '心拍変動(HRV)',
        'restingHeartRate': '安静時心拍数(RHR)'
    }
    
    key_index = np.argmax(importances)
    key_name = features[key_index]
    return feature_names.get(key_name, key_name)

def simulate_plan_b(model, features, avg_features_for_pred, best_bedtime):
    """プランB（推奨より1時間遅く寝た場合）をシミュレート"""
    plan_b_bedtime = best_bedtime + 60
    
    times_in_bed = np.arange(360, 540 + 15, 15)
    
    grid = []
    for tib in times_in_bed:
        grid.append({'bedtime_minutes': plan_b_bedtime, 'timeInBed': tib})
    
    search_df = pd.DataFrame(grid)
    
    for feature in avg_features_for_pred:
        if feature not in search_df.columns:
            search_df[feature] = avg_features_for_pred[feature]
    
    search_df = search_df[features]
    
    predictions_b = model.predict(search_df)
    best_index_b = predictions_b.argmax()
    best_time_in_bed_b = search_df.iloc[best_index_b]['timeInBed']
    
    plan_b_waketime = plan_b_bedtime + best_time_in_bed_b
    
    return format_minutes_to_time(plan_b_bedtime), format_minutes_to_time(plan_b_waketime)

# =================================================================
# ML予測処理（変更なし）
# =================================================================

def predict_for_single_user(ss, user_sheet_name, target_date_str):
    """
    単一ユーザーの単一日付に対してML予測を実行する
    """
    try:
        print(f"\n{'─'*70}")
        print(f"🤖 予測処理: {user_sheet_name} - {target_date_str}")
        print(f"{'─'*70}")
        
        df = preprocess_data(ss, user_sheet_name)
        if df is None or df.empty:
            print(f"  ⚠️ データ不足: {user_sheet_name}")
            return False
        
        today_data = df[df['dateOfSleep'] == pd.to_datetime(target_date_str)]
        today_hrv = 0
        today_rhr = 0
        
        if not today_data.empty:
            if 'dailyRmssd' in today_data.columns and len(today_data['dailyRmssd'].values) > 0:
                today_hrv = today_data['dailyRmssd'].values[0] if not pd.isna(today_data['dailyRmssd'].values[0]) else 0
            if 'restingHeartRate' in today_data.columns and len(today_data['restingHeartRate'].values) > 0:
                today_rhr = today_data['restingHeartRate'].values[0] if not pd.isna(today_data['restingHeartRate'].values[0]) else 0
        
        MIN_DATA_DAYS = 30
        
        if len(df) < MIN_DATA_DAYS:
            print(f"  ⚠️ データ不足 ({len(df)}日分) - 過去最高実績を推奨")
            
            best_day = df.loc[df['sleep_quality'].idxmax()]
            best_bedtime = best_day['bedtime_minutes']
            best_time_in_bed = best_day['timeInBed']
            best_quality = best_day['sleep_quality']
            best_waketime = best_bedtime + best_time_in_bed
            confidence = 'low'
            
            recovery_score = "安定"
            trend_hrv, trend_deep = "データ収集中", "データ収集中"
            key_factor = "データ収集中"
            plan_b_bedtime, plan_b_waketime = "N/A", "N/A"
        
        else:
            print(f"  🤖 機械学習モデルで予測 ({len(df)}日分のデータ)")
            
            features = ['bedtime_minutes', 'timeInBed']
            avg_features_for_pred = {}
            
            if 'dailyRmssd' in df.columns and df['dailyRmssd'].sum() > 0:
                features.append('dailyRmssd')
                avg_features_for_pred['dailyRmssd'] = df['dailyRmssd'].tail(7).mean()
            if 'restingHeartRate' in df.columns and df['restingHeartRate'].sum() > 0:
                features.append('restingHeartRate')
                avg_features_for_pred['restingHeartRate'] = df['restingHeartRate'].tail(7).mean()
            
            X = df[features].fillna(0)
            y = df['sleep_quality']
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            bedtimes = np.arange(-180, 120 + 15, 15)
            times_in_bed = np.arange(360, 540 + 15, 15)
            
            grid = []
            for bt in bedtimes:
                for tib in times_in_bed:
                    grid.append({'bedtime_minutes': bt, 'timeInBed': tib})
            
            search_df = pd.DataFrame(grid)
            
            for feature, value in avg_features_for_pred.items():
                search_df[feature] = value
            
            search_df = search_df[features]
            predictions = model.predict(search_df)
            
            best_index = predictions.argmax()
            best_params = search_df.iloc[best_index]
            
            best_bedtime = best_params['bedtime_minutes']
            best_time_in_bed = best_params['timeInBed']
            best_quality = predictions[best_index]
            best_waketime = best_bedtime + best_time_in_bed
            confidence = 'high' if len(df) > 90 else 'medium'
            
            recovery_score = calculate_recovery_score(df, today_hrv, today_rhr)
            trend_hrv, trend_deep = analyze_trends(df)
            key_factor = get_key_factor(model, features)
            plan_b_bedtime, plan_b_waketime = simulate_plan_b(model, features, avg_features_for_pred, best_bedtime)
        
        pred_bedtime_str = format_minutes_to_time(best_bedtime)
        pred_waketime_str = format_minutes_to_time(best_waketime)
        
        print(f"  ✅ 推奨就寝: {pred_bedtime_str} | 推奨起床: {pred_waketime_str}")
        print(f"  ✅ 予測品質: {best_quality:.1f} | 信頼度: {confidence}")
        
        ws = ss.worksheet(PREDICTION_SHEET_NAME)
        
        existing_data = ws.get_all_records()
        existing_row_index = None
        
        for idx, row in enumerate(existing_data, start=2):
            if (row.get('user_sheet_name') == user_sheet_name and 
                row.get('date') == target_date_str):
                existing_row_index = idx
                break
        
        result_row = [
            target_date_str,
            user_sheet_name,
            pred_bedtime_str,
            pred_waketime_str,
            f"{best_quality:.1f}",
            confidence,
            recovery_score,
            trend_hrv,
            trend_deep,
            key_factor,
            plan_b_bedtime,
            plan_b_waketime
        ]
        
        if existing_row_index:
            ws.update(f'A{existing_row_index}:L{existing_row_index}', [result_row])
            print(f"  📝 既存の予測を更新 (行: {existing_row_index})")
        else:
            ws.append_row(result_row, value_input_option='USER_ENTERED')
            print(f"  📝 新規予測を追加")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 予測処理でエラー: {e}")
        traceback.print_exc()
        return False

# =================================================================
# ★★★ 修正版：ML予測キュー処理 ★★★
# =================================================================
def process_prediction_queue(ss):
    """
    「ML予測キュー」シートを監視し、pendingステータスの「最初の1件」だけを処理する
    （Render.comのメモリ不足クラッシュ対策）
    """
    print("\n" + "="*70)
    print("🔄 ML予測キュー処理を開始します (v2: 1件のみ処理)")
    print("="*70)
    
    try:
        queue_sheet = ss.worksheet(QUEUE_SHEET_NAME)
    except gspread.exceptions.WorksheetNotFound:
        print(f"⚠️ 「{QUEUE_SHEET_NAME}」シートが見つかりません。")
        return {
            'success': False,
            'message': 'キューシートが見つかりません'
        }
    
    # --- 1. ヘッダーを読み込み、列のインデックスを動的に見つける ---
    #    (列の順番が変わっても動くようにするため)
    try:
        headers = queue_sheet.row_values(1)
        # gspreadは1-indexed
        status_col = headers.index('status') + 1
        sheet_name_col = headers.index('userSheetName') + 1
        target_date_col = headers.index('targetDate') + 1
        processed_at_col = headers.index('processedAt') + 1
        error_col = headers.index('errorMessage') + 1
    except ValueError as e:
        error_msg = f"FATAL: '{QUEUE_SHEET_NAME}'シートのヘッダーが不正です。'{e.args[0]}'列が見つかりません。"
        print(f"❌ {error_msg}")
        return {'success': False, 'message': error_msg}
    
    # --- 2. 'pending' の「最初の1件」を探す ---
    all_values = queue_sheet.get_all_values()[1:] # ヘッダー行(1行目)を除く
    
    target_row_index = -1 # シート上の実際の行番号 (2行目から)
    request = None

    for i, row in enumerate(all_values, start=2): # 2行目からスキャン
        # status列 (0-indexed) が 'pending' かどうか
        if row[status_col - 1] == 'pending':
            target_row_index = i
            request = {
                'userSheetName': row[sheet_name_col - 1],
                'targetDate': row[target_date_col - 1]
            }
            break # ★重要★ 最初の1件を見つけたらループを抜ける

    # --- 3. 処理対象がなければ正常終了 ---
    if not request:
        print("✅ pendingステータスのリクエストはありません。")
        return {
            'success': True,
            'message': 'pendingリクエストなし',
            'processed': 0
        }
    
    user_sheet_name = request['userSheetName']
    target_date_str = request['targetDate']
    
    print(f"📋 処理対象 (1件のみ): {user_sheet_name} @ {target_date_str} (シート行: {target_row_index})")
    
    # --- 4. 1件のジョブを実行 ---
    try:
        # 4a. ジョブを「processing」にロックし、他のワーカーが重複処理しないようにする
        queue_sheet.update_cell(target_row_index, status_col, 'processing')
        
        # 4b. 重いML処理を実行
        success = predict_for_single_user(ss, user_sheet_name, target_date_str)
        
        # 4c. 結果に応じてステータスを更新
        if success:
            queue_sheet.update_cell(target_row_index, status_col, 'completed')
            queue_sheet.update_cell(target_row_index, processed_at_col, 
                                  datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print(f"  ✅ 成功: 1件")
            return {
                'success': True,
                'processed': 1,
                'failed': 0,
                'total': 1
            }
        else:
            queue_sheet.update_cell(target_row_index, status_col, 'failed')
            queue_sheet.update_cell(target_row_index, error_col, 'データ不足または予測エラー')
            print(f"  ❌ 失敗: 1件 (MLエラー)")
            return {
                'success': True, # スクリプト自体は成功
                'processed': 0,
                'failed': 1,
                'total': 1
            }

    except Exception as e:
        # 4d. スクリプト自体がクラッシュした場合のフェイルセーフ
        error_msg = f"処理中に致命的なエラー (行: {target_row_index}): {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        try:
            # クラッシュした場合も、キューに「failed」と記録する
            queue_sheet.update_cell(target_row_index, status_col, 'failed')
            queue_sheet.update_cell(target_row_index, error_col, error_msg)
        except Exception as e_inner:
            print(f"  ❌ キューへのエラー書き込みにも失敗: {e_inner}")
        
        return {
            'success': False, # スクリプト自体が失敗
            'message': error_msg
        }

# =================================================================
# Flaskエンドポイント（変更なし）
# =================================================================

@app.route('/')
def home():
    """ヘルスチェック用エンドポイント"""
    return jsonify({
        'status': 'running',
        'service': 'Fitbit ML Prediction API',
        'version': '2.0 (Stable)', # バージョンを更新
        'endpoints': {
            '/': 'ヘルスチェック',
            '/predict': 'ML予測キューを1件処理（POST）',
            '/health': 'サービスステータス'
        }
    })

@app.route('/health')
def health():
    """サービスステータス確認"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    ML予測を実行するメインエンドポイント
    （Cloud Schedulerから呼び出される）
    """
    try:
        print("\n" + "="*70)
        print("📬 予測リクエストを受信しました (v2)")
        print("="*70)
        
        # Google Sheetsクライアントを取得
        gc = get_gspread_client()
        ss = gc.open_by_key(SPREADSHEET_ID)
        
        print(f"✅ スプレッドシート「{ss.title}」を開きました")
        
        # ★ 修正 ★ 1件のみ処理するキュー関数を呼び出す
        result = process_prediction_queue(ss)
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        error_msg = str(e)
        print(f"❌ エラーが発生しました: {error_msg}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }), 500

# =================================================================
# アプリケーション起動
# =================================================================

if __name__ == '__main__':
    # Render.comはPORT環境変数を自動で設定する
    port = int(os.environ.get('PORT', 10000))
    print(f"\n🚀 Fitbit ML予測サービス v2 を起動します (ポート: {port})")
    # debug=False は本番環境で重要
    app.run(host='0.0.0.0', port=port, debug=False)
