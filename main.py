
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. 기본 페이지 설정
# --------------------------------------------------

st.set_page_config(
    page_title="당뇨병 예측 실습실",
    page_icon="🩺",
    layout="wide"
)

# --------------------------------------------------
# 2. 디자인 설정
# --------------------------------------------------

st.markdown("""
<style>
    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #173B63;
        margin-bottom: 5px;
    }

    .sub-title {
        color: #64748b;
        font-size: 15px;
        margin-bottom: 25px;
    }

    div[data-testid="stMetric"] {
        background-color: #F0F5FA;
        padding: 17px;
        border-radius: 12px;
        border: 1px solid #DCE6F0;
    }

    .info-box {
        background-color: #EFF6FF;
        padding: 16px;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
        margin-bottom: 15px;
    }

    .warning-box {
        background-color: #FFF7ED;
        padding: 16px;
        border-radius: 10px;
        border-left: 5px solid #EA580C;
        margin-bottom: 15px;
    }

    h1, h2, h3 {
        color: #173B63;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. 데이터 불러오기
# --------------------------------------------------

@st.cache_data
def load_data():

    file_path = Path(__file__).parent / "diabetes_prediction_dataset.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            "CSV 파일을 찾을 수 없습니다. "
            "main.py와 같은 폴더에 CSV 파일을 넣어 주세요."
        )

    data = pd.read_csv(file_path, encoding="utf-8")

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

    missing_columns = [
        col for col in required_columns if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"필수 열이 없습니다: {missing_columns}")

    return data


try:
    original_df = load_data()

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 4. 데이터 전처리
# --------------------------------------------------

# 중복 데이터 제거
df = original_df.drop_duplicates().copy()

# 정답 데이터와 입력 데이터 분리
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# 숫자형 변수와 문자형 변수 구분
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()


# --------------------------------------------------
# 5. 학습용 데이터와 테스트용 데이터 분리
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 6. 머신러닝 모델 생성
# --------------------------------------------------

@st.cache_resource
def train_models(X_train, y_train):

    # 숫자형 데이터 전처리
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 문자형 데이터 전처리
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # 로지스틱 회귀용 전처리
    logistic_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # 의사결정나무용 전처리
    tree_numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("num", tree_numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # 로지스틱 회귀 모델
    logistic_model = Pipeline(steps=[
        ("preprocessor", logistic_preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # 의사결정나무 모델
    tree_model = Pipeline(steps=[
        ("preprocessor", tree_preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # 모델 학습
    logistic_model.fit(X_train, y_train)
    tree_model.fit(X_train, y_train)

    return logistic_model, tree_model


logistic_model, tree_model = train_models(X_train, y_train)


# --------------------------------------------------
# 7. 모델 예측 및 평가
# --------------------------------------------------

logistic_pred = logistic_model.predict(X_test)
tree_pred = tree_model.predict(X_test)


def calculate_metrics(y_true, y_pred):

    return {
        "정확도": accuracy_score(y_true, y_pred),
        "정밀도": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "재현율": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1-score": f1_score(
            y_true, y_pred, zero_division=0
        )
    }


logistic_metrics = calculate_metrics(y_test, logistic_pred)
tree_metrics = calculate_metrics(y_test, tree_pred)


# --------------------------------------------------
# 8. 사이드바 메뉴
# --------------------------------------------------

st.sidebar.markdown(
    "<h2 style='color:#173B63;'>🩺 당뇨병 예측 실습실</h2>",
    unsafe_allow_html=True
)

st.sidebar.caption("DATA SCIENCE LAB")

st.sidebar.divider()

menu = st.sidebar.radio(
    "메뉴 선택",
    [
        "🏠 메인 페이지",
        "📊 데이터 탐색",
        "🧹 데이터 전처리",
        "🤖 분류 모델",
        "📈 모델 평가"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Kaggle 당뇨병 데이터를 활용한 "
    "머신러닝 분석 프로젝트입니다."
)

st.sidebar.caption("교육용 분석 프로그램")


# ==================================================
# PAGE 1. 메인 페이지
# ==================================================

if menu == "🏠 메인 페이지":

    st.markdown(
        "<div class='main-title'>🩺 당뇨병 예측 실습실</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='sub-title'>"
        "데이터를 활용하여 당뇨병 예측 모델을 만들어 보는 "
        "머신러닝 프로젝트"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("## 프로젝트 소개")

    st.markdown("""
    <div class='info-box'>
    이 프로젝트는 Kaggle에서 수집한 건강 데이터를 활용하여
    당뇨병 여부를 예측하는 인공지능 분류 모델을 만드는 프로젝트입니다.

    나이, BMI, 혈당, 당화혈색소 등의 변수를 분석하고
    머신러닝 모델의 예측 성능을 비교합니다.
    </div>
    """, unsafe_allow_html=True)

    # 데이터 통계
    total_count = len(original_df)
    patient_count = int((original_df["diabetes"] == 1).sum())
    normal_count = int((original_df["diabetes"] == 0).sum())
    diabetes_rate = patient_count / total_count * 100

    st.markdown("## 데이터 현황")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("전체 데이터", f"{total_count:,}개")
    col2.metric("당뇨병 환자", f"{patient_count:,}명")
    col3.metric("변수 개수", f"{original_df.shape[1]}개")
    col4.metric("당뇨병 비율", f"{diabetes_rate:.1f}%")

    st.divider()

    st.markdown("## 당뇨병 데이터 분포")

    distribution = pd.DataFrame({
        "구분": ["당뇨병 없음", "당뇨병 있음"],
        "인원": [normal_count, patient_count]
    })

    fig = px.pie(
        distribution,
        names="구분",
        values="인원",
        hole=0.45,
        title="당뇨병 유무에 따른 데이터 비율",
        color="구분",
        color_discrete_map={
            "당뇨병 없음": "#4C9BE8",
            "당뇨병 있음": "#F28E8E"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("## 데이터 미리보기")

    st.dataframe(
        original_df.head(5),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("## 데이터 변수 설명")

    variable_info = pd.DataFrame({
        "변수": [
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "smoking_history",
            "bmi",
            "HbA1c_level",
            "blood_glucose_level",
            "diabetes"
        ],
        "설명": [
            "성별",
            "나이",
            "고혈압 여부 (0: 없음, 1: 있음)",
            "심장 질환 여부 (0: 없음, 1: 있음)",
            "흡연 이력",
            "체질량지수",
            "당화혈색소 수치",
            "혈당 수치",
            "당뇨병 여부 (0: 없음, 1: 있음)"
        ]
    })

    st.dataframe(
        variable_info,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown("## 프로젝트 진행 과정")

    st.markdown("""
    1. 데이터 수집: Kaggle에서 당뇨병 데이터 수집
    2. 데이터 탐색: 변수의 분포와 특징 분석
    3. 데이터 전처리: 중복 제거 및 데이터 변환
    4. 모델링: 두 가지 머신러닝 모델 학습
    5. 모델 평가: 예측 결과와 성능 비교
    """)

    st.warning(
        "이 프로그램은 교육 및 데이터 분석 목적으로 제작되었으며, "
        "실제 의료 진단을 대신할 수 없습니다."
    )


# ==================================================
# PAGE 2. 데이터 탐색
# ==================================================

elif menu == "📊 데이터 탐색":

    st.markdown("# 📊 데이터 탐색")

    st.write(
        "당뇨병 데이터에 포함된 변수들의 분포와 특징을 살펴봅니다."
    )

    st.divider()

    st.markdown("## 1. 성별 분포")

    gender_data = original_df["gender"].value_counts().reset_index()
    gender_data.columns = ["성별", "인원"]

    fig = px.bar(
        gender_data,
        x="성별",
        y="인원",
        color="성별",
        title="성별에 따른 데이터 분포",
        color_discrete_sequence=["#4C9BE8", "#F28E8E", "#A78BFA"]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("## 2. 나이 분포")

    fig = px.histogram(
        original_df,
        x="age",
        color="diabetes",
        nbins=30,
        barmode="overlay",
        opacity=0.7,
        title="나이에 따른 당뇨병 데이터 분포",
        labels={
            "age": "나이",
            "diabetes": "당뇨병 여부"
        },
        color_discrete_map={
            0: "#4C9BE8",
            1: "#F28E8E"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("## 3. BMI 분포")

    fig = px.histogram(
        original_df,
        x="bmi",
        color="diabetes",
        nbins=40,
        barmode="overlay",
        opacity=0.7,
        title="BMI에 따른 당뇨병 데이터 분포",
        labels={
            "bmi": "BMI",
            "diabetes": "당뇨병 여부"
        },
        color_discrete_map={
            0: "#4C9BE8",
            1: "#F28E8E"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("## 4. 혈당 수치 비교")

    fig = px.box(
        original_df,
        x="diabetes",
        y="blood_glucose_level",
        color="diabetes",
        title="당뇨병 유무에 따른 혈당 수치 비교",
        labels={
            "diabetes": "당뇨병 여부",
            "blood_glucose_level": "혈당 수치"
        },
        color_discrete_map={
            0: "#4C9BE8",
            1: "#F28E8E"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("## 5. 당화혈색소 수치 비교")

    fig = px.box(
        original_df,
        x="diabetes",
        y="HbA1c_level",
        color="diabetes",
        title="당뇨병 유무에 따른 당화혈색소 비교",
        labels={
            "diabetes": "당뇨병 여부",
            "HbA1c_level": "당화혈색소 수치"
        },
        color_discrete_map={
            0: "#4C9BE8",
            1: "#F28E8E"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "그래프에서 당뇨병 유무에 따른 변수의 분포 차이를 확인할 수 있습니다. "
        "다만 분포 차이가 있다고 해서 해당 변수가 질병의 직접적인 원인이라는 "
        "뜻은 아닙니다."
    )


# ==================================================
# PAGE 3. 데이터 전처리
# ==================================================

elif menu == "🧹 데이터 전처리":

    st.markdown("# 🧹 데이터 전처리")

    st.write(
        "머신러닝 모델이 데이터를 학습할 수 있도록 "
        "데이터를 확인하고 변환하는 과정입니다."
    )

    st.divider()

    st.markdown("## 1. 중복 데이터 확인")

    duplicate_count = int(original_df.duplicated().sum())

    col1, col2 = st.columns(2)

    col1.metric("전처리 전 데이터", f"{len(original_df):,}개")
    col2.metric("중복 데이터", f"{duplicate_count:,}개")

    st.write(
        f"중복 제거 후 데이터: **{len(df):,}개**"
    )

    st.caption(
        "중복된 행은 완전히 동일한 정보를 가진 데이터입니다. "
        "중복 데이터는 제거하여 분석에 같은 사례가 반복 반영되는 것을 줄입니다."
    )

    st.divider()

    st.markdown("## 2. 결측값 확인")

    missing_data = original_df.isnull().sum().reset_index()

    missing_data.columns = ["변수", "결측값 개수"]

    st.dataframe(
        missing_data,
        use_container_width=True,
        hide_index=True
    )

    if original_df.isnull().sum().sum() == 0:
        st.success("결측값이 없습니다.")
    else:
        st.warning("결측값이 존재합니다. 모델 학습 과정에서 처리합니다.")

    st.divider()

    st.markdown("## 3. 문자형 데이터 확인")

    categorical_data = original_df.select_dtypes(
        include="object"
    )

    st.dataframe(
        categorical_data.head(5),
        use_container_width=True
    )

    st.write(
        "gender와 smoking_history는 문자형 변수이므로 "
        "원-핫 인코딩을 이용하여 숫자로 변환합니다."
    )

    # 전처리 결과 예시
    encoded_example = pd.get_dummies(
        df.drop("diabetes", axis=1),
        columns=["gender", "smoking_history"],
        dtype=int
    )

    st.markdown("### 원-핫 인코딩 결과 미리보기")

    st.dataframe(
        encoded_example.head(5),
        use_container_width=True
    )

    st.caption(
        "위 표는 원-핫 인코딩의 결과를 보여주는 예시입니다. "
        "실제 모델 학습에서는 학습 데이터에만 인코더를 맞추고 "
        "테스트 데이터에는 학습된 인코더를 적용합니다."
    )

    st.divider()

    st.markdown("## 4. 전처리 전후 비교")

    comparison = pd.DataFrame({
        "구분": ["전처리 전", "중복 제거 후"],
        "데이터 수": [len(original_df), len(df)],
        "변수 수": [
            original_df.shape[1],
            df.shape[1]
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "데이터 확인과 중복 제거가 완료되었습니다. "
        "모델 학습 시 문자형 데이터 변환과 수치형 데이터 표준화가 "
        "자동으로 진행됩니다."
    )


# ==================================================
# PAGE 4. 분류 모델
# ==================================================

elif menu == "🤖 분류 모델":

    st.markdown("# 🤖 분류 모델")

    st.write(
        "두 가지 머신러닝 모델을 활용하여 당뇨병 여부를 예측합니다."
    )

    st.divider()

    st.markdown("## 1. 데이터 분리")

    col1, col2 = st.columns(2)

    col1.metric(
        "학습 데이터",
        f"{len(X_train):,}개",
        "80%"
    )

    col2.metric(
        "테스트 데이터",
        f"{len(X_test):,}개",
        "20%"
    )

    st.markdown("""
    <div class='info-box'>
    학습 데이터는 모델이 패턴을 배우는 데 사용하고,
    테스트 데이터는 학습에 사용하지 않은 새로운 데이터에 대한
    예측 성능을 평가하는 데 사용합니다.
    </div>
    """, unsafe_allow_html=True)

    st.write(
        "층화 추출(stratify)을 적용하여 학습용과 테스트용 데이터의 "
        "당뇨병 유무 비율이 비슷하게 유지되도록 했습니다."
    )

    st.divider()

    st.markdown("## 2. 사용할 머신러닝 모델")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 로지스틱 회귀")

        st.write("""
        입력 변수와 당뇨병 여부 사이의 관계를 학습하여
        당뇨병에 해당할 확률을 예측하는 분류 모델입니다.
        """)

    with col2:
        st.markdown("### 의사결정나무")

        st.write("""
        여러 조건을 차례로 나누어 데이터를 분류하는 모델입니다.
        나무 형태의 구조로 예측 과정을 이해하기 쉽습니다.
        """)

    st.divider()

    st.markdown("## 3. 클래스 불균형 처리")

    st.write("""
    이 데이터는 당뇨병이 없는 사람의 비율이 훨씬 높습니다.
    이를 고려하여 두 모델에 `class_weight='balanced'`를 적용했습니다.
    """)

    st.info(
        "balanced 옵션은 적은 수의 당뇨병 데이터도 "
        "학습 과정에서 중요하게 반영하도록 가중치를 조정합니다."
    )

    st.divider()

    st.markdown("## 4. 모델 학습 결과")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 로지스틱 회귀")
        st.success("학습 완료")
        st.write("테스트 데이터 예측 완료")

        st.metric(
            "예측 정확도",
            f"{logistic_metrics['정확도'] * 100:.2f}%"
        )

    with col2:
        st.markdown("### 의사결정나무")
        st.success("학습 완료")
        st.write("테스트 데이터 예측 완료")

        st.metric(
            "예측 정확도",
            f"{tree_metrics['정확도'] * 100:.2f}%"
        )

    st.divider()

    st.markdown("## 5. 실제값과 예측값 비교")

    result_df = pd.DataFrame({
        "실제 당뇨병 여부": y_test.values[:20],
        "로지스틱 회귀 예측": logistic_pred[:20],
        "의사결정나무 예측": tree_pred[:20]
    })

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "실제값과 예측값이 같으면 해당 사례를 올바르게 분류한 것입니다."
    )


# ==================================================
# PAGE 5. 모델 평가
# ==================================================

elif menu == "📈 모델 평가":

    st.markdown("# 📈 모델 평가")

    st.write(
        "테스트 데이터를 이용하여 두 모델의 예측 성능을 비교합니다."
    )

    st.divider()

    st.markdown("## 1. 성능 지표 비교")

    metrics_df = pd.DataFrame({
        "평가 지표": [
            "정확도",
            "정밀도",
            "재현율",
            "F1-score"
        ],
        "로지스틱 회귀": [
            logistic_metrics["정확도"],
            logistic_metrics["정밀도"],
            logistic_metrics["재현율"],
            logistic_metrics["F1-score"]
        ],
        "의사결정나무": [
            tree_metrics["정확도"],
            tree_metrics["정밀도"],
            tree_metrics["재현율"],
            tree_metrics["F1-score"]
        ]
    })

    st.dataframe(
        metrics_df.style.format({
            "로지스틱 회귀": "{:.4f}",
            "의사결정나무": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        metrics_df,
        x="평가 지표",
        y=["로지스틱 회귀", "의사결정나무"],
        barmode="group",
        title="두 모델의 성능 비교",
        labels={
            "value": "평가 점수",
            "variable": "모델"
        },
        color_discrete_sequence=["#4C9BE8", "#F28E8E"]
    )

    fig.update_yaxes(range=[0, 1])

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("## 2. 혼동행렬")

    model_choice = st.selectbox(
        "평가할 모델 선택",
        ["로지스틱 회귀", "의사결정나무"]
    )

    if model_choice == "로지스틱 회귀":
        selected_pred = logistic_pred
    else:
        selected_pred = tree_pred

    cm = confusion_matrix(
        y_test,
        selected_pred,
        labels=[0, 1]
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["예측: 당뇨병 없음", "예측: 당뇨병 있음"],
            y=["실제: 당뇨병 없음", "실제: 당뇨병 있음"],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            showscale=True
        )
    )

    fig.update_layout(
        title=f"{model_choice} 혼동행렬",
        xaxis_title="모델의 예측",
        yaxis_title="실제 정답"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    - TN: 당뇨병이 없는 사람을 없다고 예측
    - FP: 당뇨병이 없는 사람을 있다고 예측
    - FN: 당뇨병이 있는 사람을 없다고 예측
    - TP: 당뇨병이 있는 사람을 있다고 예측
    """)

    st.divider()

    st.markdown("## 3. 평가 지표의 의미")

    st.markdown("""
    | 평가 지표 | 의미 |
    |---|---|
    | 정확도 | 전체 데이터 중 올바르게 예측한 비율 |
    | 정밀도 | 당뇨병이라고 예측한 사람 중 실제 당뇨병인 비율 |
    | 재현율 | 실제 당뇨병인 사람 중 올바르게 찾아낸 비율 |
    | F1-score | 정밀도와 재현율을 함께 고려한 점수 |
    """)

    st.info(
        "당뇨병 데이터는 클래스 불균형이 있으므로 "
        "정확도뿐 아니라 재현율과 F1-score도 중요하게 살펴봐야 합니다."
    )

    st.divider()

    st.markdown("## 4. 최종 결과 해석")

    best_f1_model = (
        "로지스틱 회귀"
        if logistic_metrics["F1-score"] >= tree_metrics["F1-score"]
        else "의사결정나무"
    )

    best_recall_model = (
        "로지스틱 회귀"
        if logistic_metrics["재현율"] >= tree_metrics["재현율"]
        else "의사결정나무"
    )

    st.markdown(
        f"""
        - F1-score가 더 높은 모델: **{best_f1_model}**
        - 재현율이 더 높은 모델: **{best_recall_model}**
        """
    )

    st.write("""
    두 모델은 서로 다른 방식으로 데이터를 학습하기 때문에
    평가 지표에 따라 성능 차이가 나타날 수 있습니다.

    특히 당뇨병 예측에서는 실제 당뇨병 환자를 얼마나 잘 찾아내는지를
    보여주는 재현율도 중요하게 살펴볼 필요가 있습니다.
    """)

    st.warning(
        "이 결과는 특정 Kaggle 데이터셋을 이용한 교육용 분석 결과이며, "
        "실제 환자의 당뇨병 진단이나 치료 판단에 사용할 수 없습니다."
    )


# --------------------------------------------------
# 9. 하단 안내
# --------------------------------------------------

st.sidebar.divider()

st.sidebar.caption(
    "🧬 당뇨병 예측 실습실 | 머신러닝 데이터 분석 프로젝트"
)

st.sidebar.caption(
    "본 프로그램은 교육용으로 제작되었습니다."
)
