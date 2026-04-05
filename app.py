from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"

PATIENT_FEATURES = ["Age", "BMI", "Diabetes", "Hypertension"]
DIET_TO_FOOD_TYPES = {
    "Diabetic": ["Healthy Food", "Japanese", "Thai"],
    "DASH": ["Healthy Food", "Japanese", "Korean", "Thai"],
    "Weight_Loss": ["Healthy Food", "Japanese", "Korean", "Thai", "Vietnames"],
    "Weight_Gain": ["Indian", "Italian", "Mexican", "French"],
    "Standard": ["Healthy Food", "Indian", "Chinese", "Italian", "Japanese", "Thai", "Vietnames"],
}
DIET_GUIDANCE = {
    "Diabetic": "Ưu tiên các món có xu hướng cân bằng đường huyết, nhiều rau và hạn chế đường đơn.",
    "DASH": "Ưu tiên các món ít muối, cân bằng chất béo và tăng cường rau củ để hỗ trợ huyết áp.",
    "Weight_Loss": "Tập trung vào các món giàu chất xơ, đạm vừa đủ và có mật độ năng lượng hợp lý.",
    "Weight_Gain": "Chọn các món giàu năng lượng hơn nhưng vẫn cần giữ cân bằng dinh dưỡng.",
    "Standard": "Duy trì chế độ ăn cân bằng, đa dạng thực phẩm và ưu tiên món lành mạnh mỗi ngày.",
}
STUDENT_NAME = "Lê Ngọc Ánh"
STUDENT_ID = "22T1020019"
PROJECT_TITLE = (
    "Phân loại chế độ ăn và đề xuất thực đơn dựa trên chỉ số BMI của người mắc "
    "tiểu đường, cao huyết áp bằng mô hình Random Forest nhằm giảm thiểu rủi ro dinh dưỡng."
)
BLUE_PALETTE = ["#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa", "#3b82f6", "#1d4ed8"]

st.set_page_config(page_title="Hệ thống gợi ý dinh dưỡng", page_icon="🥗", layout="wide")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f5f7fb;
            --paper: #ffffff;
            --panel: #f8fbff;
            --ink: #152033;
            --muted: #64748b;
            --accent: #2563eb;
            --accent-soft: #e8f0ff;
            --accent-2: #0f766e;
            --warm: #1d4ed8;
            --line: rgba(37, 99, 235, 0.10);
            --shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(219, 234, 254, 0.95), transparent 30%),
                radial-gradient(circle at top left, rgba(240, 249, 255, 0.95), transparent 24%),
                linear-gradient(180deg, #f8fbff 0%, var(--bg) 100%);
            color: var(--ink);
            font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 2.5rem;
        }

        [data-testid="stSidebar"] {
background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
            border-right: 1px solid rgba(37, 99, 235, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #152033 !important;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label {
            background: rgba(255,255,255,0.75);
            border: 1px solid rgba(37,99,235,0.08);
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 10px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.03);
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            border-color: rgba(37,99,235,0.20);
            background: #ffffff;
        }

        .hero-card {
            position: relative;
            background:
                radial-gradient(circle at top right, rgba(255,255,255,0.28), transparent 30%),
                linear-gradient(135deg, #2563eb 0%, #1d4ed8 48%, #0f766e 100%);
            border-radius: 24px;
            padding: 34px 36px;
            box-shadow: var(--shadow);
            color: #f8fbff;
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 20px;
            overflow: hidden;
        }

        .hero-card::after {
            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            top: -90px;
            right: -60px;
            background: radial-gradient(circle, rgba(255,255,255,0.20), transparent 68%);
            pointer-events: none;
        }

        .hero-badge {
            display: inline-block;
            padding: 7px 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.10);
            color: #f4fbff;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 12px;
            font-weight: 700;
        }

        .hero-title {
            font-size: 2.3rem;
            line-height: 1.08;
            margin: 0 0 10px 0;
            font-weight: 700;
            letter-spacing: -0.03em;
        }

        .hero-text {
            max-width: 780px;
            color: rgba(244, 251, 247, 0.88);
            font-size: 1rem;
            line-height: 1.75;
            margin: 0;
        }

        .glass-card {
            background: linear-gradient(180deg, var(--paper) 0%, #fbfdff 100%);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: var(--shadow);
            margin-bottom: 14px;
        }

        .stat-card {
            background: var(--paper);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: var(--shadow);
            min-height: 132px;
        }
.stat-label {
            color: var(--muted);
            font-size: 0.82rem;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }

        .stat-value {
            font-size: 1.8rem;
            color: var(--ink);
            margin-bottom: 8px;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .stat-note {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.5;
        }

        .section-kicker {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--accent-2);
            font-size: 0.78rem;
            margin-bottom: 6px;
            font-weight: 700;
        }

        .section-title {
            font-size: 1.35rem;
            color: var(--ink);
            margin: 0 0 8px 0;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .section-copy {
            color: var(--muted);
            line-height: 1.65;
            margin-bottom: 0;
        }

        .info-chip {
            display: inline-block;
            padding: 7px 11px;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            margin-right: 8px;
            margin-bottom: 8px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .stButton>button, .stFormSubmitButton>button {
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.78rem 1rem;
            font-weight: 700;
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.18);
            transition: 0.2s ease;
        }

        .stButton>button:hover, .stFormSubmitButton>button:hover {
            background: linear-gradient(135deg, #1d4ed8, #1e40af);
            transform: translateY(-1px);
        }

        div[data-testid="stMetric"] {
            background: var(--paper);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 12px 14px;
            box-shadow: var(--shadow);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--muted);
        }

        div[data-testid="stMetricValue"] {
            color: var(--ink);
        }

        div[data-baseweb="input"], div[data-baseweb="select"], textarea {
            border-radius: 12px !important;
        }

        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, textarea {
            background: #fbfdff !important;
            border: 1px solid rgba(37, 99, 235, 0.12) !important;
            min-height: 46px;
        }
.stNumberInput label, .stSelectbox label, .stRadio label {
            color: var(--ink);
            font-weight: 600;
        }

        .stDataFrame, div[data-testid="stExpander"], div[data-testid="stForm"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid var(--line);
            box-shadow: var(--shadow);
            background: var(--paper);
        }

        div[data-testid="stForm"] {
            padding: 14px 16px 6px 16px;
        }

        div[data-testid="stAlert"] {
            border-radius: 16px;
            border: 1px solid rgba(37, 99, 235, 0.08);
        }

        div[data-testid="stExpander"] summary {
            font-weight: 700;
            color: var(--ink);
        }

        @media (max-width: 768px) {
            .hero-card {
                padding: 24px 22px;
                border-radius: 20px;
            }

            .hero-title {
                font-size: 1.8rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, badge: str) -> None:
    st.markdown(
        f"""
        <section class="hero-card">
            <div class="hero-badge">{badge}</div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-text">{subtitle}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="section-kicker">{kicker}</div>
            <h2 class="section-title">{title}</h2>
            <p class="section-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_food_data(file_path: Path, file_mtime: float) -> pd.DataFrame:
    food_df = pd.read_csv(file_path, nrows=5000)  # Limit to 5000 rows for performance
    category_path = file_path.parent / "food_category.csv"
    category_df = pd.read_csv(category_path)
    
    # Merge food with categories
    df = food_df.merge(category_df, left_on="food_category_id", right_on="id", how="left", suffixes=("_food", "_category"))
    
    # Rename columns
    df = df.rename(columns={"description_food": "Name", "description_category": "Category_Name"})
    
    # Create C_Type by mapping category names
    category_mapping = {
        "Fruits and Fruit Juices": "Healthy Food",
        "Vegetables and Vegetable Products": "Healthy Food",
        "Cereal Grains and Pasta": "Healthy Food",
        "Legumes and Legume Products": "Healthy Food",
        "Nut and Seed Products": "Healthy Food",
        "Beef Products": "Indian",
        "Pork Products": "French",
        "Poultry Products": "Italian",
        "Finfish and Shellfish Products": "Japanese",
        "Dairy and Egg Products": "Healthy Food",
        "Sausages and Luncheon Meats": "French",
        "Soups, Sauces, and Gravies": "Italian",
        "Baked Products": "Healthy Food",
        "Snacks": "Healthy Food",
        "Sweets": "Healthy Food",
        "Beverages": "Healthy Food",
        "Fats and Oils": "Healthy Food",
        "Spices and Herbs": "Thai",
        "Baby Foods": "Healthy Food",
        "Breakfast Cereals": "Healthy Food",
        "American Indian/Alaska Native Foods": "Vietnames",
        "Restaurant Foods": "Chinese",
    }
    
    df["C_Type"] = df["Category_Name"].fillna("Unknown").map(category_mapping).fillna("Healthy Food")
    
    # Set Veg_Non based on category
    df["Veg_Non"] = df["Category_Name"].fillna("").apply(lambda x: "veg" if "Vegetable" in x or "Fruit" in x else "non-veg")
    
    # Keep only necessary columns and remove duplicates
    df = df[["Name", "C_Type", "Veg_Non"]].drop_duplicates(subset=["Name"])
    
    return df


@st.cache_data
def load_patient_data(file_path: Path, file_mtime: float) -> pd.DataFrame:
    return pd.read_csv(file_path)


@st.cache_resource
def load_model_bundle() -> dict:
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


def recommend_foods(food_df: pd.DataFrame, diet_class: str, veg_choice: str, sample_size: int = 5) -> pd.DataFrame:
    allowed_types = DIET_TO_FOOD_TYPES.get(diet_class, ["Healthy Food"])
    filtered = food_df[food_df["C_Type"].isin(allowed_types)].copy()

    if veg_choice != "Tất cả":
        target = "veg" if veg_choice == "Món chay" else "non-veg"
        filtered = filtered[filtered["Veg_Non"].fillna("").str.lower() == target]

    if filtered.empty:
        filtered = food_df[food_df["C_Type"].isin(["Healthy Food"])].copy()

    if filtered.empty:
        filtered = food_df.copy()

    sample_size = min(sample_size, len(filtered))
    return filtered.sample(n=sample_size, random_state=42).reset_index(drop=True)


def plot_confusion_matrix(confusion_matrix: list[list[int]], labels: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    matrix_df = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    image = ax.imshow(matrix_df.values, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thực tế")
    ax.set_title("Ma trận nhầm lẫn")

    for row in range(len(labels)):
        for col in range(len(labels)):
            value = matrix_df.iloc[row, col]
            text_color = "white" if value > matrix_df.values.max() * 0.5 else "#152033"
            ax.text(col, row, value, ha="center", va="center", color=text_color, fontsize=10, fontweight="bold")

    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_class_distribution(patient_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = patient_df["Diet_Class"].value_counts().sort_values(ascending=False)
    colors = BLUE_PALETTE[: len(counts)]
    ax.bar(counts.index, counts.values, color=colors, edgecolor="#1d4ed8")
    ax.set_title("Phân bố nhãn Diet_Class")
    ax.set_xlabel("Nhãn")
    ax.set_ylabel("Số lượng")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    fig.tight_layout()
    return fig


def plot_bmi_by_class(patient_df: pd.DataFrame) -> plt.Figure:
    labels = list(patient_df["Diet_Class"].dropna().unique())
    series = [patient_df.loc[patient_df["Diet_Class"] == label, "BMI"].dropna() for label in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    box = ax.boxplot(series, tick_labels=labels, patch_artist=True)
    for patch, color in zip(box["boxes"], BLUE_PALETTE):
        patch.set_facecolor(color)
        patch.set_edgecolor("#1d4ed8")
    for median in box["medians"]:
        median.set_color("#0f172a")
        median.set_linewidth(2)
    ax.set_title("Phân bố BMI theo nhóm chế độ ăn")
    ax.set_xlabel("Diet_Class")
    ax.set_ylabel("BMI")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    fig.tight_layout()
    return fig
def plot_feature_importance(feature_names: list[str], importances: list[float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    chart_df = pd.DataFrame({"Đặc trưng": feature_names, "Mức độ quan trọng": importances}).sort_values(
        "Mức độ quan trọng", ascending=True
    )
    ax.barh(chart_df["Đặc trưng"], chart_df["Mức độ quan trọng"], color="#3b82f6", edgecolor="#1d4ed8")
    ax.set_title("Mức độ quan trọng của đặc trưng")
    ax.set_xlabel("Mức độ quan trọng")
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    fig.tight_layout()
    return fig


def plot_probability_chart(confidence_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    chart_df = confidence_df.sort_values("Probability", ascending=True)
    colors = BLUE_PALETTE[-len(chart_df) :]
    ax.barh(chart_df["Diet_Class"], chart_df["Probability"], color=colors, edgecolor="#1d4ed8")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Xác suất")
    ax.set_title("Phân bố xác suất dự đoán")
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    for index, value in enumerate(chart_df["Probability"]):
        ax.text(min(value + 0.02, 0.98), index, f"{value:.1%}", va="center", color="#152033")
    fig.tight_layout()
    return fig


inject_styles()

food_path = DATA_DIR / "Nguon_1" / "food.csv"
patient_path = DATA_DIR / "Nguon_2.csv"

food_df = load_food_data(food_path, food_path.stat().st_mtime)
patient_df = load_patient_data(patient_path, patient_path.stat().st_mtime)
model_bundle = load_model_bundle()

sidebar_options = [
    "1. Giới thiệu & Khám phá dữ liệu",
    "2. Triển khai mô hình",
    "3. Đánh giá & Hiệu năng",
]

st.sidebar.markdown("### Nutrition AI")
st.sidebar.caption("Phân loại chế độ ăn và gợi ý thực đơn cho người có bệnh lý nền.")
if st.sidebar.button("Tải lại dữ liệu"):
    st.cache_data.clear()
    st.rerun()
page = st.sidebar.radio("Chọn trang", sidebar_options)
st.sidebar.caption("Ứng dụng được xây dựng theo hướng dẫn Streamlit 2026.")


if page == "1. Giới thiệu & Khám phá dữ liệu":
    render_hero(
        "Hệ thống phân loại chế độ ăn và gợi ý thực đơn",
        "Ứng dụng hỗ trợ người dùng có bệnh lý nền nhận khuyến nghị dinh dưỡng rõ ràng, có dữ liệu và có thể tương tác trực tiếp trên web app.",
        "Streamlit Web App • Random Forest • Nutrition AI",
    )
    st.markdown(
        '<span class="info-chip">Đề tài bám theo yêu cầu PDF 2026</span>'
        '<span class="info-chip">Tên đề tài đã đồng bộ với hồ sơ</span>'
        '<span class="info-chip">Dữ liệu bệnh nhân và thực phẩm tách riêng</span>',
        unsafe_allow_html=True,
    )

    stats_col1, stats_col2, stats_col3 = st.columns(3)
    with stats_col1:
        render_stat_card("Sinh viên", STUDENT_NAME, f"MSSV: {STUDENT_ID}")
    with stats_col2:
        render_stat_card(
            "Nguồn bệnh nhân",
            f"{len(patient_df):,} mẫu",
            "Dữ liệu dùng để huấn luyện mô hình phân loại chế độ ăn.",
        )
    with stats_col3:
        render_stat_card(
            "Nguồn món ăn",
            f"{len(food_df):,} món",
            "Kho món ăn dùng để lọc và gợi ý thực đơn sau dự đoán.",
        )

    render_section_header(
        "Tổng quan dự án",
        "Thông tin đề tài",
        f"<strong>Tên đề tài:</strong> {PROJECT_TITLE}",
    )

    render_section_header(
        "Dữ liệu đầu vào",
        "Hai nguồn dữ liệu phục vụ hai nhiệm vụ khác nhau",
        "Nguồn 1 lưu danh sách món ăn để gợi ý thực đơn. Nguồn 2 lưu hồ sơ bệnh nhân để huấn luyện và đánh giá mô hình Random Forest.",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.write("Nguồn 1: danh sách món ăn từ Kaggle")
        st.dataframe(food_df.head(10), width='stretch')
    with col2:
        st.write("Nguồn 2: dữ liệu hồ sơ bệnh nhân tự sinh")
        st.dataframe(patient_df.head(10), width='stretch')

    render_section_header(
        "EDA",
        "Khám phá dữ liệu",
        "Hai biểu đồ bên dưới giúp nhìn nhanh phân bố nhãn và vai trò của BMI trong bài toán phân loại.",
    )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.pyplot(plot_class_distribution(patient_df))
    with chart_col2:
        st.pyplot(plot_bmi_by_class(patient_df))

    st.markdown(
        """
        **Nhận xét nhanh**

        - Dữ liệu huấn luyện gồm 1.000 mẫu và 5 nhãn, phân bố không quá lệch nhưng nhãn `Weight_Gain` ít hơn các nhãn còn lại.
        - BMI là đặc trưng tách nhóm khá rõ, đặc biệt ở các trường hợp `Weight_Loss` và `Weight_Gain`.
        - `Diabetes` và `Hypertension` là hai cờ bệnh lý quan trọng, phù hợp với bài toán phân loại phác đồ dinh dưỡng.
        """
    )


elif page == "2. Triển khai mô hình":
    render_hero(
        "Dự đoán chế độ ăn phù hợp",
        "Trang này là phần tương tác chính của ứng dụng: người dùng nhập thông tin cơ bản, hệ thống tính BMI, chạy mô hình Random Forest và gợi ý món ăn phù hợp.",
        "Interactive Inference",
    )

    render_section_header(
        "Nhập liệu",
        "Điền hồ sơ sức khỏe",
        "Các trường bên dưới được dùng để tính BMI và tạo đầu vào đúng định dạng với lúc huấn luyện mô hình.",
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Tuổi", min_value=18, max_value=100, value=30)
            height_cm = st.number_input("Chiều cao (cm)", min_value=120, max_value=220, value=165)
            weight_kg = st.number_input("Cân nặng (kg)", min_value=25, max_value=250, value=60)
        with col2:
            diabetes_text = st.selectbox("Tình trạng tiểu đường", ["Không", "Có"])
            hypertension_text = st.selectbox("Tình trạng cao huyết áp", ["Không", "Có"])
            veg_choice = st.selectbox("Loại món ưu tiên", ["Tất cả", "Món chay", "Món không chay"])

        submitted = st.form_submit_button("Dự đoán và gợi ý")

    if submitted:
        bmi = round(weight_kg / ((height_cm / 100) ** 2), 2)
        input_df = pd.DataFrame(
            [
                {
                    "Age": age,
                    "BMI": bmi,
                    "Diabetes": 1 if diabetes_text == "Có" else 0,
                    "Hypertension": 1 if hypertension_text == "Có" else 0,
                }
            ]
        )

        model = model_bundle["model"]
        prediction = model.predict(input_df[PATIENT_FEATURES])[0]
        probabilities = model.predict_proba(input_df[PATIENT_FEATURES])[0]
        class_names = list(model.classes_)

        confidence_df = (
            pd.DataFrame({"Diet_Class": class_names, "Probability": probabilities})
            .sort_values("Probability", ascending=False)
            .reset_index(drop=True)
        )
        recommendations = recommend_foods(food_df, prediction, veg_choice)

        result_col1, result_col2, result_col3 = st.columns(3)
        with result_col1:
            st.metric("BMI", bmi)
        with result_col2:
            st.metric("Nhãn dự đoán", prediction)
        with result_col3:
            st.metric("Độ tin cậy cao nhất", f"{float(confidence_df.loc[0, 'Probability']):.1%}")

        st.success(f"Phác đồ dinh dưỡng đề xuất: **{prediction}**")
        st.caption(DIET_GUIDANCE[prediction])

        chart_col1, chart_col2 = st.columns([1.1, 0.9])
        with chart_col1:
            st.pyplot(plot_probability_chart(confidence_df))
        with chart_col2:
            display_df = confidence_df.rename(
                columns={"Diet_Class": "Nhóm chế độ ăn", "Probability": "Xác suất"}
            ).copy()
            display_df["Xác suất"] = display_df["Xác suất"].map(lambda value: f"{value:.1%}")
            st.dataframe(display_df, width='stretch', hide_index=True)

        render_section_header(
            "Gợi ý món ăn",
            "Thực đơn phù hợp với nhãn dự đoán",
            "Các món bên dưới được lọc theo nhóm thực phẩm phù hợp với kết quả phân loại và ưu tiên ăn uống bạn đã chọn.",
        )

        if len(recommendations) == 0:
            st.warning("Không tìm thấy món ăn phù hợp. Vui lòng thử lại với lựa chọn khác.")
        else:
            for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                with st.expander(f"{idx}. {row['Name']} ({row['C_Type']})"):
                    st.write(f"**Tên món:** {row['Name']}")
                    st.write(f"**Loại:** {row['C_Type']}")
                    st.caption(f"Loại món: {row['Veg_Non']}")


else:
    metrics = model_bundle["metrics"]
    report_df = pd.DataFrame(model_bundle["classification_report"]).transpose()

    render_hero(
        "Đánh giá và hiệu năng mô hình",
        "Trang này chứng minh mô hình hoạt động đủ tốt cho bài toán: có chỉ số đánh giá, ma trận nhầm lẫn và phân tích sai số ngắn gọn.",
        "Model Evaluation",
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    metric_col2.metric("F1-score weighted", f"{metrics['f1_weighted']:.3f}")
    metric_col3.metric("Số mẫu test", str(metrics["test_size"]))

    render_section_header(
        "Hiệu năng",
        "Chỉ số đánh giá chính",
        "Kết quả hiện tại đến từ Random Forest đã được huấn luyện với tập dữ liệu bệnh nhân và quy trình tiền xử lý cố định.",
    )

    st.subheader("Ma trận nhầm lẫn")
    st.pyplot(plot_confusion_matrix(model_bundle["confusion_matrix"], model_bundle["labels"]))

    st.subheader("Mức độ quan trọng của đặc trưng")
    st.pyplot(
        plot_feature_importance(
            model_bundle["feature_names_after_preprocessing"], model_bundle["feature_importances"]
        )
    )

    st.subheader("Báo cáo phân loại")
    report_display = report_df.rename(
        columns={
            "precision": "precision",
            "recall": "recall",
            "f1-score": "f1-score",
            "support": "support",
        }
    )
    st.dataframe(report_display, width='stretch')

    st.markdown(
        """
        **Phân tích sai số**

        - Mô hình dễ nhầm nhất ở các nhóm có logic gần nhau, đặc biệt `Standard` với `Weight_Loss` khi BMI nằm sát ngưỡng.
        - `Diabetic` và `DASH` thường được nhận diện tốt hơn vì có tín hiệu bệnh lý rõ ràng ngay từ input.
        - Hướng cải thiện: bổ sung dữ liệu thật, thêm đặc trưng về mức độ vận động và lịch sử ăn uống, đồng thời chuẩn hóa rule mapping sang thực đơn chi tiết hơn.
        """
    )