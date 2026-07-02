import streamlit as st
import joblib
import pandas as pd


# 1. 載入模型
@st.cache_resource
def load_model():
    return joblib.load('rf_model.pkl')


model = load_model()

st.set_page_config(page_title="學生風險早期預警系統", layout="wide")
st.title("🎓 學生學習風險早期預警系統")

# 2. 建立雙軌分頁
tab1, tab2 = st.tabs(["👤 單一學生風險雷達", "📁 全班批次風險預測"])

# --- 分頁 1：單一預測 ---
with tab1:
    st.subheader("手動輸入單一學生行為特徵")
    with st.form("single_pred"):
        c1, c2 = st.columns(2)
        with c1:
            sum_click = st.slider("開學總點擊次數", 0, 3000, 500)
            score = st.slider("平時測驗平均分數", 0, 100, 60)
            activity_type = st.slider("互動多樣性", 0, 10, 3)
        with c2:
            studied_credits = st.selectbox("修習學分", [30, 60, 90, 120])
            late_count = st.slider("作業遲交總次數", 0, 10, 0)
            trend = st.selectbox("學習動力趨勢", ["成長", "平穩", "微幅衰退", "嚴重衰退"])

        submitted = st.form_submit_button("預測風險")

        if submitted:
            slope_map = {"成長": 10.0, "平穩": 0.0, "微幅衰退": -5.0, "嚴重衰退": -20.0}
            data = pd.DataFrame([[sum_click, score, activity_type, studied_credits, late_count, slope_map[trend]]],
                                columns=['sum_click', 'score', 'activity_type', 'studied_credits', 'late_count',
                                         'learning_trend_slope'])
            prob = model.predict_proba(data)[0][1] * 100

            # 顯示結果
            col_res, col_gauge = st.columns(2)
            col_res.metric("退選風險機率", f"{prob:.1f}%")
            if prob > 60:
                col_gauge.error("🔴 高風險")
            elif prob > 35:
                col_gauge.warning("🟡 中風險")
            else:
                col_gauge.success("🟢 低風險")

# --- 分頁 2：批次預測 ---
with tab2:
    st.subheader("全班數據批次匯入")
    uploaded_file = st.file_uploader("上傳全班 CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if st.button("執行全班預測"):
            probs = model.predict_proba(
                df[['sum_click', 'score', 'activity_type', 'studied_credits', 'late_count', 'learning_trend_slope']])[:,
                    1]
            df['風險機率'] = (probs * 100).round(1)
            st.dataframe(df.sort_values('風險機率', ascending=False), use_container_width=True)
