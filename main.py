"""
Fitbit睡眠最適化ML予測サービス (v2.1 - 追加特徴量 対応版)
Render.comで24時間稼働するFlaskアプリケーション

変更点:
- v2: メモリ不足対策として、キュー処理を1件ずつに変更。
- v2.1: SpO2, 呼吸数(BR), 皮膚温(Temp)のデータを特徴量としてモデルに追加。
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

# ★★★ 修正箇所 (v2.3 - MergeError & 重複データ 修正版) ★★★
def preprocess_data(ss, user_sheet_name):
    """ユーザーデータを読み込み、前処理と結合を行う"""
    sleep_df = get_sheet_data_as_df(ss, f"sleep_{user_sheet_name}")
    if sleep_df is None or sleep_df.empty:
        print(f"  [{user_sheet_name}] 睡眠データがありません。")
        return None
    
    # ★ 睡眠データも重複を削除 (念のため)
    sleep_df = sleep_df.drop_duplicates(subset=['dateOfSleep'], keep='last')
    sleep_df['dateOfSleep'] = pd.to_datetime(sleep_df['dateOfSleep'])
    
    # 睡眠データの数値化
    num_cols = ['minutesAsleep', 'efficiency', 'deep.minutes',
                'rem.minutes', 'light.minutes', 'minutesToFallAsleep', 'timeInBed']
    for col in num_cols:
        sleep_df[col] = pd.to_numeric(sleep_df[col], errors='coerce').fillna(0)
    
    # 就寝時刻を分に変換
    sleep_df['startTime'] = pd.to_datetime(sleep_df['startTime'])
    sleep_df['endTime'] = pd.to_datetime(sleep_df['endTime'])
    base_hour = 4
    bedtime_minutes = sleep_df['startTime'].dt.hour * 60 + sleep_df['startTime'].dt.minute
    bedtime_minutes = bedtime_minutes.apply(lambda x: x - 1440 if x > base_hour * 60 else x)
    sleep_df['bedtime_minutes'] = bedtime_minutes
    
    # --- 他の指標データを読み込んでマージ ---
    
    # HRV (hrv_user_XX)
    hrv_df = get_sheet_data_as_df(ss, f"hrv_{user_sheet_name}")
    if hrv_df is not None and not hrv_df.empty:
        hrv_df = hrv_df.drop_duplicates(subset=['date'], keep='last') # ★ 修正: マージ前に重複削除
        hrv_df['date'] = pd.to_datetime(hrv_df['date'])
        sleep_df = pd.merge(sleep_df, hrv_df[['date', 'dailyRmssd']], 
                           left_on='dateOfSleep', right_on='date', how='left')
        sleep_df = sleep_df.drop(columns=['date'])
        sleep_df['dailyRmssd'] = pd.to_numeric(sleep_df['dailyRmssd'], errors='coerce')
    
    # RHR (rhr_user_XX)
    rhr_df = get_sheet_data_as_df(ss, f"rhr_{user_sheet_name}")
    if rhr_df is not None and not rhr_df.empty:
        rhr_df = rhr_df.drop_duplicates(subset=['date'], keep='last') # ★ 修正: マージ前に重複削除
        rhr_df['date'] = pd.to_datetime(rhr_df['date'])
        sleep_df = pd.merge(sleep_df, rhr_df[['date', 'restingHeartRate']], 
                           left_on='dateOfSleep', right_on='date', how='left')
        sleep_df = sleep_df.drop(columns=['date'])
        sleep_df['restingHeartRate'] = pd.to_numeric(sleep_df['restingHeartRate'], errors='coerce')

    # SpO2 (spo2_user_XX)
    spo2_df = get_sheet_data_as_df(ss, f"spo2_{user_sheet_name}")
    if spo2_df is not None and not spo2_df.empty:
        spo2_df = spo2_df.drop_duplicates(subset=['date'], keep='last') # ★ 修正: マージ前に重複削除
        spo2_df['date'] = pd.to_datetime(spo2_df['date'])
        sleep_df = pd.merge(sleep_df, spo2_df[['date', 'spo2_avg']], 
                           left_on='dateOfSleep', right_on='date', how='left')
        sleep_df = sleep_df.drop(columns=['date'])
        sleep_df['spo2_avg'] = pd.to_numeric(sleep_df['spo2_avg'], errors='coerce')

    # 呼吸数 (br_user_XX)
    br_df = get_sheet_data_as_df(ss, f"br_{user_sheet_name}")
    if br_df is not None and not br_df.empty:
        br_df = br_df.drop_duplicates(subset=['date'], keep='last') # ★ 修正: マージ前に重複削除
        br_df['date'] = pd.to_datetime(br_df['date'])
        sleep_df = pd.merge(sleep_df, br_df[['date', 'breathingRate']], 
                           left_on='dateOfSleep', right_on='date', how='left')
        sleep_df = sleep_df.drop(columns=['date'])
        sleep_df['breathingRate'] = pd.to_numeric(sleep_df['breathingRate'], errors='coerce')

    # 皮膚温 (temp_user_XX)
    temp_df = get_sheet_data_as_df(ss, f"temp_{user_sheet_name}")
    if temp_df is not None and not temp_df.empty:
        temp_df = temp_df.drop_duplicates(subset=['date'], keep='last') # ★ 修正: マージ前に重複削除
        temp_df['date'] = pd.to_datetime(temp_df['date'])
        sleep_df = pd.merge(sleep_df, temp_df[['date', 'tempVariation']], 
                           left_on='dateOfSleep', right_on='date', how='left')
        sleep_df = sleep_df.drop(columns=['date'])
        sleep_df['tempVariation'] = pd.to_numeric(sleep_df['tempVariation'], errors='coerce')
    
    # 睡眠の質を計算し、全データをfillna(0)
    sleep_df['sleep_quality'] = define_sleep_quality(sleep_df)
    sleep_df = sleep_df.fillna(0)
    
    return sleep_df
def get_user_preferences(ss, user_sheet_name):
    """ユーザー管理シートからユーザーの設定（希望就寝時刻）を取得"""
    try:
        ws = ss.worksheet('ユーザー管理') # 'ユーザー管理'シートを開く
        data = ws.get_all_values()
        if len(data) <= 1:
            return None
            
        df = pd.DataFrame(data[1:], columns=data[0])
        
        user_row = df[df['sheetName'] == user_sheet_name]
        
        if user_row.empty:
            print(f"  ⚠️ [get_user_preferences] ユーザー管理シートに {user_sheet_name} が見つかりません。")
            return None

        # 'targetBedtime' 列（K列を想定）が存在するか確認
        if 'targetBedtime' not in df.columns:
             print(f"  ⚠️ [get_user_preferences] ユーザー管理シートに 'targetBedtime' 列がありません。（ステップ1を確認してください）")
             return None

        target_bedtime_str = user_row.iloc[0].get('targetBedtime')
        
        if not target_bedtime_str or target_bedtime_str == "":
            print(f"  ℹ️ [get_user_preferences] {user_sheet_name} の希望就寝時刻が未設定です。")
            return None
            
        print(f"  ℹ️ [get_user_preferences] {user_sheet_name} の希望就寝時刻「{target_bedtime_str}」を取得しました。")
        return target_bedtime_str # "23:00" などの文字列

    except Exception as e:
        print(f"  ⚠️ [get_user_preferences] ユーザー設定の読み込みエラー: {e}")
        return None

def convert_time_str_to_minutes(time_str):
    """ "HH:MM" 形式の文字列を、基準時間(4時)からの相対分に変換 """
    try:
        hours, minutes = map(int, time_str.split(':'))
        total_minutes = hours * 60 + minutes
        
        # 基準時間（AM 4:00 = 240分）より大きい時刻は、前日の時刻として扱う (マイナス値にする)
        base_hour = 4
        if total_minutes > base_hour * 60:
            total_minutes -= 1440 # 24 * 60
            
        return total_minutes # 例: "23:00" -> -60, "01:00" -> 60
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

# ★★★ 修正箇所 (v2.1) ★★★
def get_key_factor(model, features):
    """モデルから最も重要な要因を取得"""
    importances = model.feature_importances_
    # 新しい特徴量の日本語名を追加
    feature_names = {
        'bedtime_minutes': '就寝時刻のズレ',
        'timeInBed': 'ベッドにいた時間',
        'dailyRmssd': '心拍変動(HRV)',
        'restingHeartRate': '安静時心P数(RHR)',
        'spo2_avg': '睡眠中の平均SpO2',
        'breathingRate': '睡眠中の呼吸数',
        'tempVariation': '皮膚温の変化'
    }
    
    key_index = np.argmax(importances)
    key_name = features[key_index]
    return feature_names.get(key_name, key_name)

def simulate_plan_b(model, features, avg_features_for_pred, best_bedtime):
    """プランB（推奨より1時間遅く寝た場合）をシミュレート"""
    plan_b_bedtime = best_bedtime + 60
    
    # Plan Bの計算は粒度を荒く(15分)して計算負荷を下げる
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
# ML予測処理
# =================================================================

# ★★★ 修正箇所 (v2.1) ★★★
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
        
        # 予測対象日のデータを取得（分析用）
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
            
            # --- 特徴量(features)と予測用平均値(avg_features)を動的に構築 ---
            features = ['bedtime_minutes', 'timeInBed']
            avg_features_for_pred = {}
            
            # HRV
            if 'dailyRmssd' in df.columns and df['dailyRmssd'].sum() > 0:
                features.append('dailyRmssd')
                avg_features_for_pred['dailyRmssd'] = df['dailyRmssd'].tail(7).mean()
            # RHR
            if 'restingHeartRate' in df.columns and df['restingHeartRate'].sum() > 0:
                features.append('restingHeartRate')
                avg_features_for_pred['restingHeartRate'] = df['restingHeartRate'].tail(7).mean()
            # SpO2
            if 'spo2_avg' in df.columns and df['spo2_avg'].sum() > 0:
                features.append('spo2_avg')
                avg_features_for_pred['spo2_avg'] = df['spo2_avg'].tail(7).mean()
            # 呼吸数
            if 'breathingRate' in df.columns and df['breathingRate'].sum() > 0:
                features.append('breathingRate')
                avg_features_for_pred['breathingRate'] = df['breathingRate'].tail(7).mean()
            # 皮膚温 (0でないこともチェック)
            if 'tempVariation' in df.columns and df['tempVariation'].sum() != 0:
                features.append('tempVariation')
                avg_features_for_pred['tempVariation'] = df['tempVariation'].tail(7).mean()
            
            print(f"  ℹ️  使用する特徴量: {features}")
            
            X = df[features].fillna(0)
            y = df['sleep_quality']
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
# --- 予測グリッドの作成 (v3.0 - ユーザー希望時刻ベース) ---
            target_bedtime_str = get_user_preferences(ss, user_sheet_name)
            target_bedtime_minutes = None
            if target_bedtime_str:
                target_bedtime_minutes = convert_time_str_to_minutes(target_bedtime_str)

            # 希望時刻が設定されている場合、その周辺(±60分)を探す
            if target_bedtime_minutes is not None:
                print(f"  ℹ️ ユーザー希望時刻 ({target_bedtime_str}) に基づいてグリッドを生成します。")
                search_center = target_bedtime_minutes
                search_radius = 60 # 希望時刻の前後60分
                bedtimes = np.arange(search_center - search_radius, search_center + search_radius + 5, 5)
            else:
                # 未設定の場合、従来通りの広い範囲を探す
                print(f"  ℹ️ ユーザー希望時刻が未設定のため、標準範囲でグリッドを生成します。")
                bedtimes = np.arange(-180, 120 + 5, 5) # 21:00から02:00まで「5分」間隔

            times_in_bed = np.arange(360, 540 + 5, 5) # 6時間から9時間まで「5分」間隔
            print(f"  ℹ️  5分間隔で計算中 (計算パターン: {len(bedtimes) * len(times_in_bed)}件)")

            grid = []

            for bt in bedtimes:
                for tib in times_in_bed:
                    grid.append({'bedtime_minutes': bt, 'timeInBed': tib})
            
            search_df = pd.DataFrame(grid)
            
            # 平均値をグリッドに適用
            for feature, value in avg_features_for_pred.items():
                search_df[feature] = value
            
            search_df = search_df[features] # モデルが学習した特徴量と順番を合わせる
            predictions = model.predict(search_df)
            
            best_index = predictions.argmax()
            best_params = search_df.iloc[best_index]
            
            best_bedtime = best_params['bedtime_minutes']
            best_time_in_bed = best_params['timeInBed']
            best_quality = predictions[best_index]
            best_waketime = best_bedtime + best_time_in_bed
            confidence = 'high' if len(df) > 90 else 'medium'
            
            # --- 分析パート ---
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
# ★★★ v2 (1件処理) キュー処理 ★★★
# =================================================================
def process_prediction_queue(ss):
    """
    「ML予測キュー」シートを監視し、pendingステータスの「最初の1件」だけを処理する
    （Render.comのメモリ不足クラッシュ対策）
    """
    print("\n" + "="*70)
    print("🔄 ML予測キュー処理を開始します (v2.1: 1件のみ処理)")
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
    
    # --- 2. 'pending' の「最初の1件」を探す ---
    all_values = queue_sheet.get_all_values()[1:] # ヘッダー行(1行目)を除く
    
    target_row_index = -1 # シート上の実際の行番号 (2行目から)
    request = None

    for i, row in enumerate(all_values, start=2): # 2行目からスキャン
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
        # 4a. ジョブを「processing」にロック
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
            queue_sheet.update_cell(target_row_index, status_col, 'failed')
            queue_sheet.update_cell(target_row_index, error_col, error_msg)
        except Exception as e_inner:
            print(f"  ❌ キューへのエラー書き込みにも失敗: {e_inner}")
        
        return {
            'success': False, # スクリプト自体が失敗
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
        'version': '2.1 (Features: SpO2, BR, Temp)', # バージョンを更新
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
        print("📬 予測リクエストを受信しました (v2.1)")
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
# アプリケーション起動
# =================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"\n🚀 Fitbit ML予測サービス v2.1 を起動します (ポート: {port})")
    app.run(host='0.0.0.0', port=port, debug=False)

# ==========================================
# 【新規追加】モデル評価関数
# ==========================================
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def evaluate_model_performance(ss, user_sheet_name):
    """
    モデルの予測精度を評価する（クロスバリデーション）
    """
    print(f"\n{'='*70}")
    print(f"📊 モデル評価: {user_sheet_name}")
    print(f"{'='*70}")
    
    df = preprocess_data(ss, user_sheet_name)
    if df is None or len(df) < 50:
        print(f"  ⚠️ データ不足（{len(df) if df is not None else 0}日分）")
        return None
    
    # 特徴量とターゲット
    features = ['bedtime_minutes', 'timeInBed']
    
    # HRV
    if 'dailyRmssd' in df.columns and df['dailyRmssd'].sum() > 0:
        features.append('dailyRmssd')
    # RHR
    if 'restingHeartRate' in df.columns and df['restingHeartRate'].sum() > 0:
        features.append('restingHeartRate')
    # SpO2
    if 'spo2_avg' in df.columns and df['spo2_avg'].sum() > 0:
        features.append('spo2_avg')
    # 呼吸数
    if 'breathingRate' in df.columns and df['breathingRate'].sum() > 0:
        features.append('breathingRate')
    # 皮膚温
    if 'tempVariation' in df.columns and df['tempVariation'].sum() != 0:
        features.append('tempVariation')
    
    X = df[features].fillna(0)
    y = df['sleep_quality']
    
    # 訓練データとテストデータに分割（8:2）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 現在のRandomForestモデルで学習
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 予測
    y_pred = model.predict(X_test)
    
    # 評価指標
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n  📈 評価結果:")
    print(f"     RMSE (二乗平均平方根誤差): {rmse:.2f}")
    print(f"     MAE  (平均絶対誤差):        {mae:.2f}")
    print(f"     R²   (決定係数):            {r2:.3f}")
    print(f"\n  💡 解釈:")
    print(f"     - 予測誤差は平均 ±{mae:.1f}点")
    print(f"     - モデルは睡眠質の変動の {r2*100:.1f}% を説明")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'model': model,
        'features': features
    }


# GASから呼び出すためのエンドポイント
@app.route('/evaluate', methods=['POST'])
def evaluate():
    """モデル評価用エンドポイント"""
    try:
        gc = get_gspread_client()
        ss = gc.open_by_key(SPREADSHEET_ID)
        
        # 全ユーザーを評価
        user_sheets = ['user_01', 'user_02', 'user_03']  # ←実際のシート名に変更
        results = {}
        
        for sheet_name in user_sheets:
            result = evaluate_model_performance(ss, sheet_name)
            if result:
                results[sheet_name] = {
                    'rmse': result['rmse'],
                    'mae': result['mae'],
                    'r2': result['r2']
                }
        
        return jsonify({
            'success': True,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

