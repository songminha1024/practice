
import streamlit as st
import pandas as pd
import plotly.express as px
import json

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==================================================
# 1. 기본 설정
# ==================================================

st.set_page_config(
    page_title="당뇨병 예측 실습실",
    page_icon="🩺",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

DATA_FILE = BASE_DIR / "diabetes_prediction_dataset.csv"

SAVE_DIR = BASE_DIR / "내_기록"
SAVE_FILE = SAVE_DIR / "기록.json"

SAVE_DIR.mkdir(exist_ok=True)


# ==================================================
# 2. 디자인 설정
# ==================================================

st.markdown("""
<style>
.stApp {
    background-color: #F6F9FE;
}

.main-title {
    color: #174A7E;
    font-size: 36px;
    font-weight: 800;
}

section[data-testid="stSidebar"] {
    background-color: #EAF2FC;
}

div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #DCE8F5;
    padding: 18px;
    border-radius: 12px;
}

div[data-testid="stMetricValue"] {
    color: #174A7E;
}

h2, h3 {
    color: #245B91;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. 데이터 불러오기
# ==================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE, encoding="utf-8")


if not DATA_FILE.exists():
    st.error("CSV 파일을 찾을 수 없습니다.")
    st.info(
        "diabetes_prediction_dataset.csv 파일을 "
        "main.py와 같은 폴더에 넣어 주세요."
    )
    st.stop()

try:
    df = load_data()

except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()


required_columns = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "diabetes"
]

if not all(col in df.columns for col in required_columns):
    st.error("CSV 파일의 열 이름을 확인해 주세요.")
    st.stop()

if df["diabetes"].isna().any():
    st.error("당뇨병 여부가 비어 있는 데이터가 있습니다.")
    st.stop()

if not set(df["diabetes"].dropna().unique()).issubset({0, 1}):
    st.error("diabetes 열은 0과 1로 구성되어야 합니다.")
    st.stop()


# ==================================================
# 4. 기록 불러오기
# ==================================================

def load_saved_data():

    if SAVE_FILE.exists():

        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as file:
                return json.load(file)

        except (OSError, json.JSONDecodeError):
            return {}

    return {}


saved_data = load_saved_data()

saved_meanings = saved_data.get("column_meanings", {})

saved_source = saved_data.get(
    "source",
    "Kaggle: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset"
)


# ==================================================
# 5. 데이터 기본 정보
# ==================================================

total_people = len(df)
total_columns = len(df.columns)

diabetes_count = int((df["diabetes"] == 1).sum())

diabetes_rate = (
    diabetes_count / total_people * 100
    if total_people > 0 else 0
)

clean_df = df.drop_duplicates().copy()

duplicate_count = total_people - len(clean_df)


# ==================================================
# 6. 머신러닝 모델 학습 함수
# ==================================================

@st.cache_data
def train_models(data):

    # 중복 데이터 제거
    data = data.drop_duplicates().copy()

    # 입력 데이터와 정답 데이터 분리
    X = data.drop(columns=["diabetes"])
    y = data["diabetes"].astype(int)

    # 숫자형과 문자형 열 구분
    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # 학습용 80%, 테스트용 20% 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 전처리 파이프라인 생성
    def make_preprocessor(scale_numeric):

        numeric_steps = [
            ("imputer", SimpleImputer(strategy="median"))
        ]

        if scale_numeric:
            numeric_steps.append(
                ("scaler", StandardScaler())
            )

        numeric_pipeline = Pipeline(numeric_steps)

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers = []

        if numeric_features:
            transformers.append(
                ("numeric", numeric_pipeline, numeric_features)
            )

        if categorical_features:
            transformers.append(
                ("categorical", categorical_pipeline, categorical_features)
            )

        return ColumnTransformer(transformers)

    # 로지스틱 회귀
    logistic_model = Pipeline([
        ("preprocessor", make_preprocessor(True)),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # 의사결정나무
    tree_model = Pipeline([
        ("preprocessor", make_preprocessor(False)),
        ("model", DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # 모델 학습
    logistic_model.fit(X_train, y_train)
    tree_model.fit(X_train, y_train)

    # 테스트 데이터 예측
    pred_logistic = logistic_model.predict(X_test)
    pred_tree = tree_model.predict(X_test)

    predictions = {
        "로지스틱 회귀": pred_logistic,
        "의사결정나무": pred_tree
    }

    # 평가 지표 계산
    metrics = {}

    for name, prediction in predictions.items():

        metrics[name] = {
            "정확도": accuracy_score(y_test, prediction),
            "정밀도": precision_score(
                y_test, prediction, zero_division=0
            ),
            "재현율": recall_score(
                y_test, prediction, zero_division=0
            ),
            "F1 점수": f1_score(
                y_test, prediction, zero_division=0
            )
        }

    # 혼동 행렬 계산
    cm_logistic = confusion_matrix(
        y_test,
        pred_logistic,
        labels=[0, 1]
    ).tolist()

    cm_tree = confusion_matrix(
        y_test,
        pred_tree,
        labels=[0, 1]
    ).tolist()

    # 실제 정답과 예측 결과 비교
    prediction_table = pd.DataFrame({
        "실제 당뇨병 여부": y_test.to_numpy(),
        "로지스틱 회귀 예측": pred_logistic,
        "의사결정나무 예측": pred_tree
    })

    return {
        "metrics": metrics,
        "cm_logistic": cm_logistic,
        "cm_tree": cm_tree,
        "predictions": prediction_table,
        "train_count": len(X_train),
        "test_count": len(X_test),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features
    }


# ==================================================
# 7. 사이드바 메뉴
# ==================================================

st.sidebar.title("🩺 당뇨병 예측 실습실")

st.sidebar.caption("HEALTH DATA ANALYSIS LAB")

menu = st.sidebar.radio(
    "페이지 선택",
    [
        "🏠 메인 화면",
        "📊 데이터 탐색",
        "🧹 데이터 전처리",
        "🤖 분류 모델",
        "📈 모델 평가"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "건강 데이터를 분석하고 머신러닝 모델로 "
    "당뇨병 여부를 예측하는 프로젝트입니다."
)


# ==================================================
# 8. 메인 화면
# ==================================================

if menu == "🏠 메인 화면":

    st.markdown(
        '<div class="main-title">🩺 당뇨병 예측 실습실</div>',
        unsafe_allow_html=True
    )

    st.write(
        "건강 데이터를 활용하여 당뇨병과 관련된 특징을 분석하고 "
        "인공지능 모델로 당뇨병 여부를 예측하는 실습 공간입니다."
    )

    st.subheader("🔍 탐구 질문")

    st.info(
        "나이, BMI, 혈당, 당화혈색소 등의 건강 데이터로 "
        "당뇨병 여부를 예측할 수 있을까?"
    )

    st.divider()

    st.subheader("📊 데이터 한눈에 보기")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "전체 데이터 수",
        f"{total_people:,}명"
    )

    col2.metric(
        "전체 열 개수",
        f"{total_columns}개"
    )

    col3.metric(
        "당뇨병 환자 수",
        f"{diabetes_count:,}명"
    )

    col4.metric(
        "당뇨병 비율",
        f"{diabetes_rate:.2f}%"
    )

    st.divider()

    # 열 정보 표
    st.subheader("📋 데이터 열 정보")

    st.write(
        "각 열의 의미를 교재에서 찾아 우리말 뜻에 입력해 보세요."
    )

    column_info = []

    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):
            value_type = "숫자형"
        else:
            value_type = "문자형"

        missing_count = int(df[col].isna().sum())

        unique_count = int(df[col].nunique())

        column_info.append({
            "열 이름": col,
            "우리말 뜻": saved_meanings.get(col, ""),
            "값의 종류": f"{value_type} (고유값 {unique_count}개)",
            "빈 값 개수": missing_count
        })

    info_df = pd.DataFrame(column_info)

    with st.form("meaning_form"):

        edited_df = st.data_editor(
            info_df,
            hide_index=True,
            use_container_width=True,
            disabled=[
                "열 이름",
                "값의 종류",
                "빈 값 개수"
            ],
            column_config={
                "우리말 뜻": st.column_config.TextColumn(
                    "우리말 뜻",
                    help="교재를 참고하여 각 열의 의미를 입력하세요.",
                    width="large"
                )
            }
        )

        st.markdown("### 📚 데이터 출처")

        source_text = st.text_area(
            "데이터 출처를 입력하세요.",
            value=saved_source,
            height=100
        )

        save_button = st.form_submit_button(
            "💾 기록 저장하기",
            use_container_width=True
        )

    if save_button:

        meanings = {}

        for _, row in edited_df.iterrows():
            meanings[row["열 이름"]] = str(row["우리말 뜻"])

        save_content = {
            "column_meanings": meanings,
            "source": source_text
        }

        try:
            with open(SAVE_FILE, "w", encoding="utf-8") as file:
                json.dump(
                    save_content,
                    file,
                    ensure_ascii=False,
                    indent=4
                )

            st.success("입력한 기록을 JSON 파일에 저장했습니다!")

        except OSError as e:
            st.error(f"기록 저장 중 오류가 발생했습니다: {e}")

    st.divider()

    st.subheader("🔎 데이터 미리 보기")

    st.write("실제 데이터의 처음 다섯 행입니다.")

    st.dataframe(
        df.head(5),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.caption("데이터 출처: Kaggle 당뇨병 예측 데이터셋")

    st.caption(
        "※ 이 앱은 데이터 분석 및 학습용이며 "
        "실제 의료 진단에 사용할 수 없습니다."
    )


# ==================================================
# 9. 데이터 탐색 페이지
# ==================================================

elif menu == "📊 데이터 탐색":

    st.title("📊 데이터 탐색")

    st.write(
        "건강 관련 변수의 분포를 살펴보고 "
        "당뇨병 여부에 따른 차이를 비교합니다."
    )

    st.divider()

    st.subheader("1. 숫자형 건강 데이터 분포")

    numeric_options = [
        "age",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level"
    ]

    korean_names = {
        "age": "나이",
        "bmi": "BMI",
        "HbA1c_level": "당화혈색소",
        "blood_glucose_level": "혈당 수치"
    }

    selected_numeric = st.selectbox(
        "분석할 변수 선택",
        numeric_options,
        format_func=lambda x: korean_names[x]
    )

    hist_fig = px.histogram(
        df,
        x=selected_numeric,
        color=df["diabetes"].astype(str),
        barmode="overlay",
        opacity=0.7,
        title=f"{korean_names[selected_numeric]} 분포",
        labels={
            selected_numeric: korean_names[selected_numeric],
            "color": "당뇨병 여부"
        }
    )

    st.plotly_chart(
        hist_fig,
        use_container_width=True
    )

    st.caption(
        "0은 당뇨병이 없는 경우, 1은 당뇨병이 있는 경우입니다."
    )

    st.divider()

    st.subheader("2. 당뇨병 여부에 따른 건강 지표 비교")

    selected_box = st.selectbox(
        "비교할 건강 지표 선택",
        numeric_options,
        key="box_select",
        format_func=lambda x: korean_names[x]
    )

    box_fig = px.box(
        df,
        x="diabetes",
        y=selected_box,
        color="diabetes",
        title=f"당뇨병 여부에 따른 {korean_names[selected_box]} 비교",
        labels={
            "diabetes": "당뇨병 여부",
            selected_box: korean_names[selected_box]
        }
    )

    st.plotly_chart(
        box_fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("3. 문자형 데이터 비교")

    category = st.selectbox(
        "비교할 항목 선택",
        ["gender", "smoking_history"],
        format_func=lambda x: {
            "gender": "성별",
            "smoking_history": "흡연 이력"
        }[x]
    )

    category_data = (
        df.groupby([category, "diabetes"])
        .size()
        .reset_index(name="인원")
    )

    category_fig = px.bar(
        category_data,
        x=category,
        y="인원",
        color="diabetes",
        barmode="group",
        title="항목별 당뇨병 여부 비교"
    )

    st.plotly_chart(
        category_fig,
        use_container_width=True
    )

    st.info(
        "그래프는 변수별 분포와 당뇨병 여부에 따른 차이를 보여 줍니다. "
        "두 변수 사이에 차이가 나타나더라도 직접적인 원인이라고 "
        "단정할 수는 없습니다."
    )


# ==================================================
# 10. 데이터 전처리 페이지
# ==================================================

elif menu == "🧹 데이터 전처리":

    st.title("🧹 데이터 전처리")

    st.write(
        "머신러닝 모델에 데이터를 입력하기 전에 "
        "데이터의 상태를 확인하고 전처리 과정을 살펴봅니다."
    )

    st.divider()

    st.subheader("1. 중복 데이터 확인")

    col1, col2 = st.columns(2)

    col1.metric(
        "전처리 전 데이터",
        f"{len(df):,}행"
    )

    col2.metric(
        "중복 데이터",
        f"{duplicate_count:,}행"
    )

    st.write(
        f"중복 데이터를 제거하면 {len(clean_df):,}행이 남습니다."
    )

    st.dataframe(
        clean_df.head(5),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("2. 결측치 확인")

    missing_df = pd.DataFrame({
        "열 이름": df.columns,
        "결측치 개수": [
            int(df[col].isna().sum())
            for col in df.columns
        ]
    })

    st.dataframe(
        missing_df,
        use_container_width=True,
        hide_index=True
    )

    total_missing = int(df.isna().sum().sum())

    st.metric(
        "전체 결측치 개수",
        f"{total_missing:,}개"
    )

    st.write(
        "숫자형 결측치는 중앙값으로, 문자형 결측치는 "
        "최빈값으로 대체하도록 모델의 전처리 과정을 구성했습니다."
    )

    st.divider()

    st.subheader("3. 문자형 데이터 인코딩")

    st.markdown("""
    **원-핫 인코딩이란?**

    문자로 이루어진 범주형 데이터를 여러 개의 0과 1로 된 열로
    변환하는 방법입니다.

    예를 들어 성별이나 흡연 이력처럼 여러 범주로 나뉘는
    데이터를 머신러닝 모델이 처리할 수 있도록 바꿉니다.
    """)

    st.write("원-핫 인코딩 적용 예시")

    encoded_preview = pd.get_dummies(
        clean_df.drop(columns=["diabetes"])
    ).head(5)

    st.dataframe(
        encoded_preview,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "위 표는 인코딩 결과의 미리 보기입니다. "
        "실제 모델에서는 학습 데이터에 맞춰 전처리 과정을 학습하고 "
        "테스트 데이터에 적용합니다."
    )

    st.divider()

    st.subheader("4. 전처리 과정 정리")

    st.markdown("""
    1. 중복 데이터 제거
    2. 결측치 확인
    3. 숫자형 결측치는 중앙값으로 대체
    4. 문자형 결측치는 최빈값으로 대체
    5. 문자형 데이터 원-핫 인코딩
    6. 로지스틱 회귀에 사용할 숫자형 데이터 표준화
    """)


# ==================================================
# 11. 분류 모델 페이지
# ==================================================

elif menu == "🤖 분류 모델":

    st.title("🤖 분류 모델")

    st.write(
        "두 가지 머신러닝 알고리즘을 사용하여 "
        "당뇨병 여부를 예측합니다."
    )

    st.divider()

    try:
        results = train_models(df)

    except Exception as e:
        st.error(f"모델 학습 중 오류가 발생했습니다: {e}")
        st.stop()

    st.subheader("1. 학습 데이터와 테스트 데이터")

    col1, col2 = st.columns(2)

    col1.metric(
        "학습 데이터",
        f"{results['train_count']:,}개"
    )

    col2.metric(
        "테스트 데이터",
        f"{results['test_count']:,}개"
    )

    st.write(
        "전체 데이터를 학습용 80%, 테스트용 20%로 나누었습니다. "
        "층화 추출을 사용하여 당뇨병 여부의 비율이 "
        "비슷하게 유지되도록 했습니다."
    )

    st.divider()

    st.subheader("2. 사용한 머신러닝 모델")

    with st.expander("① 로지스틱 회귀"):

        st.write(
            "입력된 건강 데이터를 바탕으로 당뇨병일 확률을 계산하고 "
            "그 확률에 따라 0 또는 1로 분류하는 모델입니다."
        )

    with st.expander("② 의사결정나무"):

        st.write(
            "건강 데이터의 조건을 단계적으로 나누어 "
            "최종적으로 당뇨병 여부를 판단하는 모델입니다."
        )

    st.divider()

    st.subheader("3. 모델 예측 결과 비교")

    comparison = pd.DataFrame(results["metrics"]).T
    comparison.index.name = "모델"

    st.dataframe(
        comparison.style.format("{:.4f}"),
        use_container_width=True
    )

    st.divider()

    st.subheader("4. 테스트 데이터 예측 미리 보기")

    prediction_df = results["predictions"].head(20).copy()

    prediction_df = prediction_df.replace({
        0: "당뇨병 없음",
        1: "당뇨병 있음"
    })

    st.dataframe(
        prediction_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "실제 정답과 두 모델의 예측 결과를 비교한 표입니다."
    )


# ==================================================
# 12. 모델 평가 페이지
# ==================================================

elif menu == "📈 모델 평가":

    st.title("📈 모델 평가")

    st.write(
        "학습한 두 모델의 예측 결과를 "
        "여러 평가 지표로 비교합니다."
    )

    st.divider()

    try:
        results = train_models(df)

    except Exception as e:
        st.error(f"모델 평가 중 오류가 발생했습니다: {e}")
        st.stop()

    metrics_df = pd.DataFrame(results["metrics"]).T
    metrics_df.index.name = "모델"

    st.subheader("1. 평가 지표 비교")

    st.dataframe(
        metrics_df.style.format("{:.4f}"),
        use_container_width=True
    )

    chart_df = metrics_df.reset_index().melt(
        id_vars="모델",
        var_name="평가 지표",
        value_name="점수"
    )

    metrics_fig = px.bar(
        chart_df,
        x="평가 지표",
        y="점수",
        color="모델",
        barmode="group",
        title="두 모델의 평가 지표 비교",
        range_y=[0, 1]
    )

    st.plotly_chart(
        metrics_fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("2. 혼동 행렬")

    selected_model = st.selectbox(
        "확인할 모델 선택",
        ["로지스틱 회귀", "의사결정나무"]
    )

    if selected_model == "로지스틱 회귀":
        cm = results["cm_logistic"]
    else:
        cm = results["cm_tree"]

    cm_fig = px.imshow(
        cm,
        text_auto=True,
        x=["예측: 당뇨병 없음", "예측: 당뇨병 있음"],
        y=["실제: 당뇨병 없음", "실제: 당뇨병 있음"],
        color_continuous_scale="Blues",
        title=f"{selected_model} 혼동 행렬"
    )

    st.plotly_chart(
        cm_fig,
        use_container_width=True
    )

    st.markdown("""
    **혼동 행렬의 의미**

    - TN: 실제 당뇨병이 없고, 없다고 예측한 경우
    - FP: 실제 당뇨병이 없지만, 있다고 예측한 경우
    - FN: 실제 당뇨병이 있지만, 없다고 예측한 경우
    - TP: 실제 당뇨병이 있고, 있다고 예측한 경우
    """)

    st.divider()

    st.subheader("3. 평가 지표의 의미")

    with st.expander("정확도 (Accuracy)"):

        st.write(
            "전체 예측 중 정답을 맞힌 비율입니다."
        )

    with st.expander("정밀도 (Precision)"):

        st.write(
            "당뇨병이라고 예측한 사람 중 실제 당뇨병인 사람의 비율입니다."
        )

    with st.expander("재현율 (Recall)"):

        st.write(
            "실제 당뇨병인 사람 중 모델이 당뇨병이라고 찾아낸 사람의 비율입니다."
        )

    with st.expander("F1 점수 (F1-score)"):

        st.write(
            "정밀도와 재현율을 함께 고려한 지표입니다."
        )

    st.info(
        "당뇨병 환자 수가 상대적으로 적을 수 있으므로 "
        "정확도만으로 모델을 평가하기 어렵습니다. "
        "당뇨병 환자를 얼마나 찾아내는지 보여 주는 "
        "재현율도 함께 확인해야 합니다."
    )

    st.caption(
        "※ 이 앱은 교육 및 데이터 분석 실습용이며 "
        "실제 의료 진단이나 치료 판단에 사용할 수 없습니다."
    )
