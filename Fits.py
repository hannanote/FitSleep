import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

# 프로젝트 내 파일 위치를 기준으로 경로 설정
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATA_DIR = os.path.join(BASE_DIR, 'data')

MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.keras')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.joblib')
DATA_PATH = os.path.join(DATA_DIR, 'fitbit_merged_processed.pkl')

# 1. 모델 및 도구 로드
@st.cache_resource
def load_model_and_tools():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, scaler, label_encoder

# 2. 데이터 로드
@st.cache_data
def load_user_data():
    df = pd.read_pickle(DATA_PATH)
    return df

# 3. 예측 및 요약 함수
def predict_and_summarize(user_df, model, scaler, le):
    features = ['TotalSteps', 'VeryActiveMinutes', 'FairlyActiveMinutes',
                'LightlyActiveMinutes', 'SedentaryMinutes', 'VeryActiveDistance',
                'ModeratelyActiveDistance', 'LightActiveDistance', 'Calories',
                'TotalMinutesAsleep', 'TotalTimeInBed']

    user_recent = user_df[-14:].copy()

    # DayType encoded (LabelEncoder로 transform)
    user_recent['DayType_encoded'] = le.transform(user_recent['DayType'])

    # 수치형 features 스케일링
    X_num = scaler.transform(user_recent[features].values)

    # 범주형 DayType 인코딩 추가
    X_combined = np.concatenate([X_num, user_recent['DayType_encoded'].values.reshape(-1, 1)], axis=1)

    # 입력 형태 변환
    X_input = np.expand_dims(X_combined, axis=0)

    # 예측 (shape: (1,3,1)으로 나올 것이라 가정)
    predictions = model.predict(X_input)
    predicted_labels = (predictions > 0.5).astype(int).flatten()

    # 최근 14일 요약 통계
    user_14days = user_df.tail(14)
    summary_stats = {
        '평균 수면시간 (시간)': round(user_14days['TotalMinutesAsleep'].mean() / 60, 2),
        '평균 수면 효율 (%)': round((user_14days['TotalMinutesAsleep'].sum() / user_14days['TotalTimeInBed'].sum()) * 100, 1),
        '평균 걸음 수 (보)': int(user_14days['TotalSteps'].mean()),
        '아주 활동적인 시간 (분)': round(user_14days['VeryActiveMinutes'].mean(), 1),
        '앉아있는 시간 (분)': round(user_14days['SedentaryMinutes'].mean(), 1)
    }

    return predicted_labels, summary_stats

# 4. Streamlit 앱
def main():
    st.title("💤 FitSleep AI 수면 예측 및 건강 요약")
    st.markdown("최근 14일 데이터로 향후 3일 수면 상태 예측 및 건강 지표 요약을 제공합니다.")

    df = load_user_data()
    model, scaler, le = load_model_and_tools()

    user_ids = df['Id'].unique()
    selected_id = st.selectbox("사용자 ID 선택", user_ids)

    if st.button("예측 실행"):
        user_df = df[df['Id'] == selected_id].copy()

        if len(user_df) < 14:
            st.warning("해당 사용자의 데이터가 14일보다 적어 예측할 수 없습니다.")
            return

        predicted_labels, summary_stats = predict_and_summarize(user_df, model, scaler, le)

        st.success("✅ 수면 예측 완료!")
        st.subheader("📈 향후 3일 수면 상태 예측 (0=불량, 1=양질)")
        for i, label in enumerate(predicted_labels, 1):
            st.write(f"Day {i}: **{label}**")

        st.subheader("📝 최근 14일 건강 데이터 요약")
        for k, v in summary_stats.items():
            st.write(f"- {k}: {v}")

if __name__ == "__main__":
    main()
