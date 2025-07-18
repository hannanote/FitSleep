import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- 모델 및 도구 로드 ---
@st.cache_resource(show_spinner=False)
def load_model_and_tools():
    model = load_model('C:/Users/choin/Downloads/FitBit/best_model.keras')
    scaler = joblib.load('C:/Users/choin/Downloads/FitBit/scaler.joblib')
    label_encoder = joblib.load('C:/Users/choin/Downloads/FitBit/label_encoder.joblib')
    return model, scaler, label_encoder

# --- 사용자 데이터 로드 ---
@st.cache_data(show_spinner=False)
def load_user_data():
    df = pd.read_pickle('C:/Users/choin/Downloads/FitBit/fitbit_merged_processed.pkl')
    return df

# --- 예측 및 요약 계산 함수 ---
def predict_and_summarize(user_df, model, scaler, le):
    features = ['TotalSteps', 'VeryActiveMinutes', 'FairlyActiveMinutes',
                'LightlyActiveMinutes', 'SedentaryMinutes', 'VeryActiveDistance',
                'ModeratelyActiveDistance', 'LightActiveDistance', 'Calories',
                'TotalMinutesAsleep', 'TotalTimeInBed']
    
    user_recent = user_df[-14:].copy()
    user_recent['DayType_encoded'] = le.transform(user_recent['DayType'])
    
    X_num = scaler.transform(user_recent[features].values)
    X_combined = np.concatenate([X_num, user_recent['DayType_encoded'].values.reshape(-1,1)], axis=1)
    X_input = np.expand_dims(X_combined, axis=0)
    
    predictions = model.predict(X_input)
    predicted_labels = (predictions > 0.5).astype(int).flatten()
    
    user_14days = user_df.tail(14)
    summary_stats = {
        '평균 수면시간 (시간)': round(user_14days['TotalMinutesAsleep'].mean() / 60, 2),
        '평균 수면 효율 (%)': round((user_14days['TotalMinutesAsleep'].sum() / user_14days['TotalTimeInBed'].sum()) * 100, 1),
        '평균 걸음 수 (보)': int(user_14days['TotalSteps'].mean()),
        '아주 활동적인 시간 (분)': round(user_14days['VeryActiveMinutes'].mean(), 1),
        '앉아있는 시간 (분)': round(user_14days['SedentaryMinutes'].mean(), 1)
    }
    
    return predicted_labels, summary_stats

# --- OpenAI 피드백 생성 함수 ---
def generate_feedback(summary_stats, predicted_labels):
    summary_text = (
        f"- 평균 수면시간: {summary_stats['평균 수면시간 (시간)']}시간\n"
        f"- 평균 수면 효율: {summary_stats['평균 수면 효율 (%)']}%\n"
        f"- 평균 걸음 수: {summary_stats['평균 걸음 수 (보)']}보\n"
        f"- 아주 활동적인 시간: {summary_stats['아주 활동적인 시간 (분)']}분\n"
        f"- 앉아있는 시간: 하루 평균 {summary_stats['앉아있는 시간 (분)']}분"
    )
    
    prompt = f"""
당신은 수면 건강 코치입니다. 아래의 사용자 데이터를 바탕으로 개인 맞춤형 건강 피드백을 작성해주세요.

### 사용자 건강 데이터 요약
{summary_text}

### 수면 예측 결과
- 향후 3일 간 예측된 수면 상태 (0=불량, 1=양질): {list(predicted_labels)}

### 작성 지침:
1. 위 데이터에 기반하여 사용자에게 따뜻하고 실질적인 피드백을 주세요.
2. 수면 시간이 부족하거나 활동량이 과소/과다할 경우 원인을 짐작하고 조언을 제시해주세요.
3. 예측된 수면 상태가 3일 모두 양질이면 축하 메시지를 포함해주세요. 불량일 경우 행동 개선안을 포함해주세요.
4. 다음 항목으로 구성된 피드백을 작성해주세요:

- 🌙 ** 예측된 3일 수면 패턴**
- ✅ **행동 피드백**
- 📊 **분석 근거**
- 💡 **동기부여 메시지**
- ❓ **자기 점검 질문** (2개)

말투는 전문가이면서도 친절하게 해주세요. 한글로 작성해주세요.
"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "당신은 수면 건강 전문 코치입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8
    )
    return response.choices[0].message.content

# --- Streamlit UI 메인 ---
def main():
    st.title("💤 FitSleep AI 수면 예측 및 맞춤 피드백")
    st.markdown("최근 14일 데이터로 향후 3일 수면 상태 예측과 맞춤형 건강 피드백을 제공합니다.")
    
    df = load_user_data()
    model, scaler, le = load_model_and_tools()
    
    user_ids = df['Id'].unique()
    selected_id = st.selectbox("사용자 ID 선택", user_ids)
    
    if st.button("예측 및 피드백 생성"):
        user_df = df[df['Id'] == selected_id].copy()
        
        if len(user_df) < 14:
            st.warning("해당 사용자의 데이터가 14일보다 적어 예측할 수 없습니다.")
            return
        
        predicted_labels, summary_stats = predict_and_summarize(user_df, model, scaler, le)
        
        st.subheader("📈 향후 3일 수면 상태 예측 (0=불량, 1=양질)")
        for i, label in enumerate(predicted_labels, 1):
            st.write(f"Day {i}: **{label}**")
        
        st.subheader("📝 최근 14일 건강 데이터 요약")
        for k, v in summary_stats.items():
            st.write(f"- {k}: {v}")
        
        with st.spinner("💬 AI 맞춤형 피드백 생성 중... 잠시만 기다려주세요."):
            feedback = generate_feedback(summary_stats, predicted_labels)
            st.subheader("🤖 맞춤형 건강 피드백")
            st.markdown(feedback)

if __name__ == "__main__":
    main()
