"""
Fitbit睡眠最適化ML予測サービス (v3.0 - 夜間データ派生特徴量追加版)
Render.comで24時間稼働するFlaskアプリケーション

変更点:
- v2: メモリ不足対策として、キュー処理を1件ずつに変更。
- v2.1: SpO2, 呼吸数(BR), 皮膚温(Temp)のデータを特徴量としてモデルに追加。
- v3.0: 夜間データのみから16種類の派生特徴量を生成（曜日、季節性、移動平均、比率など）
"""

from flask import Flask, request, jsonify
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from datetime import datetime, timedelta
import os
import json
import time
import traceback

app = Flask(__name__)

# =================================================================
# 設定
# =================================================================

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
# ヘルパー関数
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
    """
    ★v3.0: 夜間データのみから派生特徴量を生成
    ユーザーデータを読み込み、前処理と結合を行う
    """
    sleep_df = get_sheet_data_as_df(ss, f"sleep_{user_sheet_name}")
    if sleep_df is None or sleep_df.empty:
        print(f"  [{user_sheet_name}] 睡眠データがありません。")
        return None
    
    sleep_df = sleep_df.drop_duplicates(subset=['dateOfSleep'], keep='last')
    sleep_df['dateOfSleep'] = pd.to_datetime(sleep_df['dateOfSleep'])
    
    # 睡眠データの数値化
    num_cols = ['minutesAsleep', 'efficiency', 'deep.minutes',
                'rem.minutes', 'light.minutes', 'minutesToFallAsleep', 
                'timeInBed', 'wake.minutes', 'wake.count']
    for col in num_cols:
        if col in sleep_df.columns:
            sleep_df[col] = pd.to_numeric(sleep_df[col], errors='coerce').fillna(0)
    
    # 就寝時刻を分に変換
    sleep_df['startTime'] = pd.to_datetime(sleep_df['startTime'])
    sleep_df['endTime'] = pd.to_datetime(sleep_df['endTime'])
    base_hour = 4
    bedtime_minutes = sleep_df['startTime'].dt.hour * 60 + sleep_df['startTime'].dt.minute
    bedtime_minutes = bedtime_minutes.apply(lambda x: x - 1440 if x > base_hour * 60 else x)
    sleep_df['bedtime_minutes'] = bedtime_minutes
    
    # ==========================================
    # ★★★ 新規特徴量（夜間データから派生）★★★
    # ==========================================
    
    # 1. 曜日（0=月曜, 6=日曜）
    sleep_df['day_of_week'] = sleep_df['dateOfSleep'].dt.dayofweek
    
    # 2. 週末フラグ（金曜・土曜の夜）
    sleep_df['is_weekend'] = sleep_df['day_of_week'].apply(lambda x: 1 if x >= 4 else 0)
    
    # 3. 月（季節性）
    sleep_df['month'] = sleep_df['dateOfSleep'].dt.month
    
    # 4. 深睡眠の比率（%）
    sleep_df['deep_ratio'] = (sleep_df['deep.minutes'] / sleep_df['minutesAsleep'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    
    # 5. レム睡眠の比率（%）
    sleep_df['rem_ratio'] = (sleep_df['rem.minutes'] / sleep_df['minutesAsleep'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    
    # 6. 中途覚醒の深刻度（回数 × 時間）
    if 'wake.count' in sleep_df.columns and 'wake.minutes' in sleep_df.columns:
        sleep_df['wake_severity'] = sleep_df['wake.count'] * sleep_df['wake.minutes']
    else:
        sleep_df['wake_severity'] = 0
    
    # 7. 入眠速度
    sleep_df['sleep_onset_speed'] = sleep_df['minutesToFallAsleep']
    
    # 8. 過去3日間の平均睡眠時間（移動平均）
    sleep_df['sleep_3day_avg'] = sleep_df['minutesAsleep'].rolling(window=3, min_periods=1).mean()
    
    # 9. 過去7日間の睡眠時間の標準偏差（安定性指標）
    sleep_df['sleep_7day_std'] = sleep_df['minutesAsleep'].rolling(window=7, min_periods=3).std().fillna(0)
    
    # 10. 前日との睡眠時間の差分
    sleep_df['sleep_diff_prev_day'] = sleep_df['minutesAsleep'].diff().fillna(0)
    
    # 11. 就寝時刻の標準偏差（過去7日間）
    sleep_df['bedtime_7day_std'] = sleep_df['bedtime_minutes'].rolling(window=7, min_periods=3).std().fillna(0)
    
    # 12. 連続睡眠不足日数（7時間未満）
    sleep_df['sleep_deficit_streak'] = 0
    deficit_count = 0
    for idx in sleep_df.index:
        if sleep_df.loc[idx, 'minutesAsleep'] < 420:  # 7時間 = 420分
            deficit_count += 1
        else:
            deficit_count = 0
        sleep_df.loc[idx, 'sleep_deficit_streak'] = deficit_count
    
    # ==========================================
    # 既存の外部データマージ処理（HRV, RHR, SpO2等）
    # ==========================================
    
    # HRV
    hrv_df = get_sheet_data_as_df(ss, f"hrv_{user_sheet_name}")
    if hrv_df is not None and not hrv_df.empty:
        hrv_df = hrv_df.drop_duplicates(subset=['date'], keep='last')
        hrv_df['date'] = pd.to_datetime(hrv_df['date'])
        sleep_df = pd.merge(sleep_df, hrv_df[['date', 'dailyRmssd']], 
                           left_on='dateOfSleep', right_on='date', how='left')
        sleep_df = sleep_df.drop(columns=['date'])
        sleep_df['dailyRmssd'] = pd.to_numeric(sleep_df['dailyRmssd'], errors='coerce')
        
        # ★13. HRVの7日間移動平均
        sleep_df['hrv_7day_avg'] = sleep_df['dailyRmssd'].rolling(window=7, min_periods=3).mean().fillna(0)
        
        # ★14. HRVの前日比
        sleep_df['hrv_diff_prev_day'] = sleep_df['dailyRmssd'].diff().fillna(0)
    
    # RHR
    rhr_df = get_sheet_data_as_df(ss, f"rhr_{user_sheet_name}")
    if rhr_df is not None and not rhr_df.empty:
        rhr_df = rhr_df.drop_duplicates(subset=['date'], keep='last')
        rhr_df['date'] = pd.to_datetime(rhr_df['date'])
        sleep_df = pd.merge(sleep_df, rhr_df[['date', 'restingHeartRate']], 
                           left_on='dateOfSleep', right_on='date', how='left')
        sleep_df = sleep_df.drop(columns=['date'])
        sleep_df['restingHeartRate'] = pd.to_numeric(sleep_df['restingHeartRate'], errors='coerce')
        
        # ★15. RHRの7日間移動平均
        sleep_df['rhr_7day_avg'] = sleep_df['restingHeartRate'].rolling(window=7, min_periods=3).mean().fillna(0)

    # SpO2
    spo2_df = get_sheet_data_as_df(ss, f"spo2_{user_sheet_name}")
    if spo2_df is not None and not spo2_df.empty:
        spo2_df = spo2_df.drop_duplicates(subset=['date'], keep='last')
        spo2_df['date'] = pd.to_datetime(spo2_df['date'])
        sleep_df = pd.merge(sleep_df, spo2_df[['date', 'spo2_avg']], 
                           left_on='dateOfSleep', right_on='date', how='left')
        sleep_df = sleep_df.drop(columns=['date'])
        sleep_df['spo2_avg'] = pd.to_numeric(sleep_df['spo2_avg'], errors='coerce')

    # 呼吸数
    br_df = get_sheet_data_as_df(ss, f"br_{user_sheet_name}")
    if br_df is not None and not br_df.empty:
        br_df = br_df.drop_duplicates(subset=['date'], keep='last')
        br_df['date'] = pd.to_datetime(br_df['date'])
        sleep_df = pd.merge(sleep_df, br_df[['date', 'breathingRate']], 
                           left_on='dateOfSleep', right_on='date', how='left')
        sleep_df = sleep_df.drop(columns=['date'])
        sleep_df['breathingRate'] = pd.to_numeric(sleep_df['breathingRate'], errors='coerce')

    # 皮膚温
    temp_df = get_sheet_data_as_df(ss, f"temp_{user_sheet_name}")
    if temp_df is not None and not temp_df.empty:
        temp_df = temp_df.drop_duplicates(subset=['date'], keep='last')
        temp_df['date'] = pd.to_datetime(temp_df['date'])
        sleep_df = pd.merge(sleep_df, temp_df[['date', 'tempVariation']], 
                           left_on='dateOfSleep', right_on='date', how='left')
        sleep_df = sleep_df.drop(columns=['date'])
        sleep_df['tempVariation'] = pd.to_numeric(sleep_df['tempVariation'], errors='coerce')
    
    # 睡眠の質を計算
    sleep_df['sleep_quality'] = define_sleep_quality(sleep_df)
    sleep_df = sleep_df.fillna(0)
    
    return sleep_df

def get_user_preferences(ss, user_sheet_name):
    """ユーザー管理シートからユーザーの設定（希望就寝時刻）を取得"""
    try:
        ws = ss.worksheet('ユーザー管理')
        data = ws.get_all_values()
        if len(data) <= 1:
            return None
            
        df = pd.DataFrame(data[1:], columns=data[0])
        user_row = df[df['sheetName'] == user_sheet_name]
        
        if user_row.empty:
            print(f"  ⚠️ [get_user_preferences] ユーザー管理シートに {user_sheet_name} が見つかりません。")
            return None

        if 'targetBedtime' not in df.columns:
             print(f"  ⚠️ [get_user_preferences] ユーザー管理シートに 'targetBedtime' 列がありません。")
             return None

        target_bedtime_str = user_row.iloc[0].get('targetBedtime')
        
        if not target_bedtime_str or target_bedtime_str == "":
            print(f"  ℹ️ [get_user_preferences] {user_sheet_name} の希望就寝時刻が未設定です。")
            return None
            
        print(f"  ℹ️ [get_user_preferences] {user_sheet_name} の希望就寝時刻「{target_bedtime_str}」を取得しました。")
        return target_bedtime_str

    except Exception as e:
        print(f"  ⚠️ [get_user_preferences] ユーザー設定の読み込みエラー: {e}")
        return None

def convert_time_str_to_minutes(time_str):
    """HH:MM形式の文字列を、基準時間(4時)からの相対分に変換"""
    try:
        hours, minutes = map(int, time_str.split(':'))
        total_minutes = hours * 60 + minutes
        
        base_hour = 4
        if total_minutes > base_hour * 60:
            total_minutes -= 1440
            
        return total_minutes
    except Exception as e:
        print(f"  ⚠️ [convert_time_str_to_minutes] 時刻変換エラー: {time_str}, {e}")
        return None
        
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
        'day_of_week': '曜日',
        'is_weekend': '週末',
        'month': '季節',
        'deep_ratio': '深睡眠の比率',
        'rem_ratio': 'レム睡眠の比率',
        'wake_severity': '中途覚醒の深刻度',
        'sleep_onset_speed': '入眠速度',
        'sleep_3day_avg': '3日間平均睡眠時間',
        'sleep_7day_std': '睡眠時間の安定性',
        'sleep_diff_prev_day': '前日との睡眠時間差',
        'bedtime_7day_std': '就寝時刻の安定性',
        'sleep_deficit_streak': '連続睡眠不足日数',
        'dailyRmssd': '心拍変動(HRV)',
        'hrv_7day_avg': 'HRVの7日間平均',
        'hrv_diff_prev_day': 'HRVの前日比',
        'restingHeartRate': '安静時心拍数(RHR)',
        'rhr_7day_avg': 'RHRの7日間平均',
        'spo2_avg': '睡眠中の平均SpO2',
        'breathingRate': '睡眠中の呼吸数',
        'tempVariation': '皮膚温の変化'
    }
    
    key_index = np.argmax(importances)
    key_name = features[key_index]
    return feature_names.get(key_name, key_name)

# 既存の simulate_plan_b をこれに丸ごと置き換えてください
def simulate_plan_b(model, features, avg_features_for_pred, best_bedtime, target_date_str):
    """
    プランB（推奨より1時間遅く寝た場合）をシミュレート
    ★修正: target_date_strを受け取り、日付関連の特徴量を追加
    """
    plan_b_bedtime = best_bedtime + 60
    times_in_bed = np.arange(360, 540 + 15, 15)
    
    grid = []
    for tib in times_in_bed:
        grid.append({'bedtime_minutes': plan_b_bedtime, 'timeInBed': tib})
    
    search_df = pd.DataFrame(grid)
    
    # 数値系の平均値を適用
    for feature, value in avg_features_for_pred.items():
        if feature not in search_df.columns:
            search_df[feature] = value

    # ★★★ 修正追加: 日付情報の計算と適用 ★★★
    # 明日の日付（起床日）を基準にする
    tomorrow = pd.to_datetime(target_date_str) + pd.Timedelta(days=1)
    tomorrow_dow = tomorrow.dayofweek
    tomorrow_is_weekend = 1 if tomorrow_dow >= 4 else 0
    tomorrow_month = tomorrow.month

    if 'day_of_week' in features:
        search_df['day_of_week'] = tomorrow_dow
    if 'is_weekend' in features:
        search_df['is_weekend'] = tomorrow_is_weekend
    if 'month' in features:
        search_df['month'] = tomorrow_month
    # ★★★★★★★★★★★★★★★★★★★★★★★

    search_df = search_df[features]
    predictions_b = model.predict(search_df)
    best_index_b = predictions_b.argmax()
    best_time_in_bed_b = search_df.iloc[best_index_b]['timeInBed']
    
    plan_b_waketime = plan_b_bedtime + best_time_in_bed_b
    
    return format_minutes_to_time(plan_b_bedtime), format_minutes_to_time(plan_b_waketime)
# =================================================================
# ML予測処理
# =================================================================

def predict_for_single_user(ss, user_sheet_name, target_date_str):
    """
    ★v3.0: 新規特徴量を使用した予測
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
            
            # ★★★ 新規特徴量を含む候補リスト ★★★
            feature_candidates = [
                'bedtime_minutes', 'timeInBed',
                'day_of_week', 'is_weekend', 'month',
                'deep_ratio', 'rem_ratio', 'wake_severity',
                'sleep_onset_speed', 'sleep_3day_avg', 'sleep_7day_std',
                'sleep_diff_prev_day', 'bedtime_7day_std', 'sleep_deficit_streak',
                'dailyRmssd', 'hrv_7day_avg', 'hrv_diff_prev_day',
                'restingHeartRate', 'rhr_7day_avg',
                'spo2_avg', 'breathingRate', 'tempVariation'
            ]
            
            # 実際に存在し、データがある特徴量のみ使用
            features = []
            avg_features_for_pred = {}
            
            for feature in feature_candidates:
                if feature in df.columns:
                    # bedtime_minutes と timeInBed は予測時に変動させる
                    if feature in ['bedtime_minutes', 'timeInBed']:
                        features.append(feature)
                    # カテゴリカル変数（曜日・月）
                    elif feature in ['day_of_week', 'is_weekend', 'month']:
                        features.append(feature)
                        # 明日の値を後で設定（avg不要）
                    # その他の数値特徴量
                    elif df[feature].sum() != 0:
                        features.append(feature)
                        avg_features_for_pred[feature] = df[feature].tail(7).mean()
            
            print(f"  ℹ️  使用する特徴量数: {len(features)}")
            print(f"  ℹ️  特徴量: {features}")
            
            X = df[features].fillna(0)
            y = df['sleep_quality']
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # ユーザー希望時刻の取得
            target_bedtime_str = get_user_preferences(ss, user_sheet_name)
            target_bedtime_minutes = None
            if target_bedtime_str:
                target_bedtime_minutes = convert_time_str_to_minutes(target_bedtime_str)

            if target_bedtime_minutes is not None:
                print(f"  ℹ️ ユーザー希望時刻 ({target_bedtime_str}) に基づいてグリッドを生成します。")
                search_center = target_bedtime_minutes
                search_radius = 60
                bedtimes = np.arange(search_center - search_radius, search_center + search_radius + 5, 5)
            else:
                print(f"  ℹ️ ユーザー希望時刻が未設定のため、標準範囲でグリッドを生成します。")
                bedtimes = np.arange(-180, 120 + 5, 5)

            times_in_bed = np.arange(360, 540 + 5, 5)
            print(f"  ℹ️  5分間隔で計算中 (計算パターン: {len(bedtimes) * len(times_in_bed)}件)")

            # ★★★ 明日の日付情報を計算 ★★★
            tomorrow = pd.to_datetime(target_date_str) + pd.Timedelta(days=1)
            tomorrow_dow = tomorrow.dayofweek
            tomorrow_is_weekend = 1 if tomorrow_dow >= 4 else 0
            tomorrow_month = tomorrow.month

            # グリッド作成
            grid = []
            for bt in bedtimes:
                for tib in times_in_bed:
                    grid.append({'bedtime_minutes': bt, 'timeInBed': tib})
            
            search_df = pd.DataFrame(grid)
            
            # 平均値を適用
            for feature, value in avg_features_for_pred.items():
                search_df[feature] = value
            
            # ★★★ 明日の日付情報を設定 ★★★
            if 'day_of_week' in features:
                search_df['day_of_week'] = tomorrow_dow
            if 'is_weekend' in features:
                search_df['is_weekend'] = tomorrow_is_weekend
            if 'month' in features:
                search_df['month'] = tomorrow_month
            
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
            plan_b_bedtime, plan_b_waketime = simulate_plan_b(model, features, avg_features_for_pred, best_bedtime, target_date_str)
        
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

def process_prediction_queue(ss):
    """ML予測キューを監視し、pendingステータスの最初の1件だけを処理する"""
    print("\n" + "="*70)
    print("🔄 ML予測キュー処理を開始します (v3.0: 新規特徴量対応)")
    print("="*70)
    
    try:
        queue_sheet = ss.worksheet(QUEUE_SHEET_NAME)
    except gspread.exceptions.WorksheetNotFound:
        print(f"⚠️ 「{QUEUE_SHEET_NAME}」シートが見つかりません。")
        return {
            'success': False,
            'message': 'キューシートが見つかりません'
        }
    
    try:
        headers = queue_sheet.row_values(1)
        status_col = headers.index('status') + 1
        sheet_name_col = headers.index('userSheetName') + 1
        target_date_col = headers.index('targetDate') + 1
        processed_at_col = headers.index('processedAt') + 1
        error_col = headers.index('errorMessage') + 1
    except ValueError as e:
        error_msg = f"FATAL: '{QUEUE_SHEET_NAME}'シートのヘッダーが不正です。'{e.args[0]}'列が見つかりません。"
        print(f"❌ {error_msg}")
        return {'success': False, 'message': error_msg}
    
    all_values = queue_sheet.get_all_values()[1:]
    target_row_index = -1
    request = None

    for i, row in enumerate(all_values, start=2):
        if row[status_col - 1] == 'pending':
            target_row_index = i
            request = {
                'userSheetName': row[sheet_name_col - 1],
                'targetDate': row[target_date_col - 1]
            }
            break

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
    
    try:
        queue_sheet.update_cell(target_row_index, status_col, 'processing')
        success = predict_for_single_user(ss, user_sheet_name, target_date_str)
        
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
                'success': True,
                'processed': 0,
                'failed': 1,
                'total': 1
            }

    except Exception as e:
        error_msg = f"処理中に致命的なエラー (行: {target_row_index}): {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        try:
            queue_sheet.update_cell(target_row_index, status_col, 'failed')
            queue_sheet.update_cell(target_row_index, error_col, error_msg)
        except Exception as e_inner:
            print(f"  ❌ キューへのエラー書き込みにも失敗: {e_inner}")
        
        return {
            'success': False,
            'message': error_msg
        }

# =================================================================
# Flaskエンドポイント
# =================================================================

@app.route('/')
def home():
    """ヘルスチェック用エンドポイント"""
    return jsonify({
        'status': 'running',
        'service': 'Fitbit ML Prediction API',
        'version': '3.0 (夜間データ派生特徴量追加版)',
        'endpoints': {
            '/': 'ヘルスチェック',
            '/predict': 'ML予測キューを1件処理（POST）',
            '/evaluate': 'モデル精度評価（POST）',
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
    """ML予測を実行するメインエンドポイント"""
    try:
        print("\n" + "="*70)
        print("📬 予測リクエストを受信しました (v3.0)")
        print("="*70)
        
        gc = get_gspread_client()
        ss = gc.open_by_key(SPREADSHEET_ID)
        
        print(f"✅ スプレッドシート「{ss.title}」を開きました")
        
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
# 【新規追加】モデル評価機能
# =================================================================

def evaluate_model_performance(ss, user_sheet_name):
    """モデルの予測精度を評価する（クロスバリデーション）"""
    print(f"\n{'='*70}")
    print(f"📊 モデル評価: {user_sheet_name}")
    print(f"{'='*70}")
    
    df = preprocess_data(ss, user_sheet_name)
    if df is None or len(df) < 50:
        print(f"  ⚠️ データ不足（{len(df) if df is not None else 0}日分）")
        return None
    
    # ★★★ v3.0: 新規特徴量を含む評価 ★★★
    feature_candidates = [
        'bedtime_minutes', 'timeInBed',
        'day_of_week', 'is_weekend', 'month',
        'deep_ratio', 'rem_ratio', 'wake_severity',
        'sleep_onset_speed', 'sleep_3day_avg', 'sleep_7day_std',
        'sleep_diff_prev_day', 'bedtime_7day_std', 'sleep_deficit_streak',
        'dailyRmssd', 'hrv_7day_avg', 'hrv_diff_prev_day',
        'restingHeartRate', 'rhr_7day_avg',
        'spo2_avg', 'breathingRate', 'tempVariation'
    ]
    
    features = []
    for feature in feature_candidates:
        if feature in df.columns and df[feature].sum() != 0:
            features.append(feature)
    
    if len(features) < 2:
        print(f"  ⚠️ 有効な特徴量が不足")
        return None
    
    X = df[features].fillna(0)
    y = df['sleep_quality']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n  📈 評価結果:")
    print(f"     RMSE: {rmse:.2f}")
    print(f"     MAE:  {mae:.2f}")
    print(f"     R²:   {r2:.3f}")
    print(f"     特徴量数: {len(features)}")
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'features': features,
        'n_samples': len(df)
    }

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """モデル評価用エンドポイント"""
    try:
        print("\n" + "="*70)
        print("📊 モデル評価リクエストを受信 (v3.0)")
        print("="*70)
        
        gc = get_gspread_client()
        ss = gc.open_by_key(SPREADSHEET_ID)
        
        user_sheet = ss.worksheet('ユーザー管理')
        user_data = user_sheet.get_all_values()
        
        results = {}
        
        for i in range(1, len(user_data)):
            sheet_name = user_data[i][2]
            is_active = user_data[i][3]
            
            if str(is_active).upper() == 'TRUE' and sheet_name:
                print(f"\n評価中: {sheet_name}")
                result = evaluate_model_performance(ss, sheet_name)
                if result:
                    results[sheet_name] = result
        
        if not results:
            return jsonify({
                'success': False,
                'message': '評価可能なユーザーがいません',
                'timestamp': datetime.now().isoformat()
            })
        
        return jsonify({
            'success': True,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ エラー: {error_msg}")
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
    port = int(os.environ.get('PORT', 10000))
    print(f"\n🚀 Fitbit ML予測サービス v3.0 を起動します (ポート: {port})")
    app.run(host='0.0.0.0', port=port, debug=False)
