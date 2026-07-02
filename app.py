import streamlit as st
import joblib
import pandas as pd


# ==========================================
# 1. 載入模型與環境設定
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load('rf_model.pkl')


model = load_model()

st.set_page_config(page_title="學生風險早期預警系統", layout="wide")
st.title("🎓 學生學習風險早期預警系統")

# ==========================================
# 🌟 新增：六大特徵說明折疊面板 (UI/UX 大加分)
# ==========================================
with st.expander("📊 點此查看「六大行為特徵」定義與說明"):
    st.markdown("""
    | 特徵名稱 | 變數名稱 | 定義與實務意義 |
    | :--- | :--- | :--- |
    | **開學總點擊次數** | `sum_click` | 前 28 天內於平台的總點擊量，反映初期基礎**系統參與度**。 |
    | **平時測驗平均分數** | `score` | 初期測驗或作業平均得分，反映最直接的**學業掌握度**。 |
    | **互動多樣性** | `activity_type` | 觸及的相異系統組件數量，校正盲目點擊，評估學習**廣度與品質**。 |
    | **本學期修習學分** | `studied_credits` | 當學期註冊總學分數，用以評估**整體課業負擔**是否過重。 |
    | **作業遲交總次數** | `late_count` | 晚於系統截止日期的總次數，精準捕捉學生的**拖延與逃避行為**。 |
    | **學習動力趨勢斜率** | `learning_trend_slope` | 前四週點擊量之線性迴歸斜率，敏銳偵測**學習熱度是否迅速衰退**。 |
    """)

# ==========================================
# 2. 建立雙軌分頁
# ==========================================
tab1, tab2 = st.tabs(["👤 單一學生風險雷達", "📁 全班批次風險預測"])

# ==========================================
# --- 分頁 1：單一學生預測 ---
# ==========================================
with tab1:
    st.subheader("手動輸入單一學生行為特徵")
    with st.form("single_pred"):
        c1, c2 = st.columns(2)

        with c1:
            sum_click = st.slider("開學總點擊次數", 0, 3000, 500)
            score = st.slider("平時測驗平均分數", 0, 100, 60)
            activity_type = st.slider("互動多樣性", 0, 10, 3)

        with c2:
            # 🌟 完美兼顧實務：公開定義對應關係字典
            credit_mapping = {
                30: "30 (約台灣 5~6 學分，極度輕量)",
                60: "60 (約台灣 10~12 學分，低負擔)",
                90: "90 (約台灣 15~18 學分，正常標準)",
                120: "120 (約台灣 20~25 學分，滿載/超修)"
            }

            # 使用 format_func 進行顯示轉換
            studied_credits = st.selectbox(
                "本學期修習學分 (OULAD 資料集標準)",
                options=list(credit_mapping.keys()),
                format_func=lambda x: credit_mapping[x]
            )
            # 顯示背景實際數值，展現系統透明度
            st.caption(f"💡 背景模型實際接收並運算之學分數值：**{studied_credits}**")

            late_count = st.slider("作業遲交總次數", 0, 10, 0)
            trend = st.selectbox("學習動力趨勢", ["成長", "平穩", "微幅衰退", "嚴重衰退"])

        submitted = st.form_submit_button("預測風險")

        if submitted:
            # 轉換動力趨勢為實際斜率數值
            slope_map = {"成長": 10.0, "平穩": 0.0, "微幅衰退": -5.0, "嚴重衰退": -20.0}

            # 組合特徵投入模型
            data = pd.DataFrame([[sum_click, score, activity_type, studied_credits, late_count, slope_map[trend]]],
                                columns=['sum_click', 'score', 'activity_type', 'studied_credits', 'late_count',
                                         'learning_trend_slope'])
            prob = model.predict_proba(data)[0][1] * 100

            # 顯示結果儀表板
            col_res, col_gauge = st.columns(2)
            col_res.metric("退選風險機率", f"{prob:.1f}%")
            if prob > 60:
                col_gauge.error("🔴 高風險")
            elif prob > 35:
                col_gauge.warning("🟡 中風險")
            else:
                col_gauge.success("🟢 低風險")

# ==========================================
# --- 分頁 2：全班數據批次匯入 ---
# ==========================================
with tab2:
    st.subheader("全班數據批次匯入")
    uploaded_file = st.file_uploader("上傳全班 CSV (請包含對應之特徵欄位)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if st.button("執行全班預測"):
            # 1. 呼叫模型計算機率
            probs = model.predict_proba(
                df[['sum_click', 'score', 'activity_type', 'studied_credits', 'late_count', 'learning_trend_slope']])[:,
                    1]
            df['風險機率 (%)'] = (probs * 100).round(1)


            # 2. 新增紅綠燈標籤判斷函數
            def get_risk_label(prob):
                if prob > 60:
                    return "🔴 高風險"
                elif prob > 35:
                    return "🟡 中風險"
                else:
                    return "🟢 低風險"


            df['風險標籤'] = df['風險機率 (%)'].apply(get_risk_label)

            # 3. 排序與重新排列欄位 (高風險置頂，重要欄位往前挪)
            df = df.sort_values('風險機率 (%)', ascending=False)
            cols = ['學號', '姓名', '風險標籤', '風險機率 (%)'] + [c for c in df.columns if
                                                                   c not in ['學號', '姓名', '風險標籤',
                                                                             '風險機率 (%)']]
            df = df[cols]


            # 4. 🌟 設定紅綠燈醒目底色
            def highlight_risk(val):
                if val == "🔴 高風險":
                    return 'background-color: #ffe6e6; color: #cc0000; font-weight: bold'
                elif val == "🟡 中風險":
                    return 'background-color: #fff2e6; color: #cc6600; font-weight: bold'
                elif val == "🟢 低風險":
                    return 'background-color: #e6ffe6; color: #006600'
                return ''


            # 5. 輸出帶有顏色的精美表格
            # 處理不同 Pandas 版本的相容性問題
            if hasattr(df.style, 'map'):
                styled_df = df.style.map(highlight_risk, subset=['風險標籤'])
            else:
                styled_df = df.style.applymap(highlight_risk, subset=['風險標籤'])

            st.dataframe(styled_df, use_container_width=True)
