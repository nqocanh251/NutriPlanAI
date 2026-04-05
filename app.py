from __future__ import annotations

import base64
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
ASSETS_DIR = BASE_DIR / "assets"
MODEL_PATH = MODELS_DIR / "model.pkl"

PATIENT_FEATURES = ["Age", "BMI", "Diabetes", "Hypertension"]
BLUE_PALETTE = ["#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8"]
MEAL_ACCENT_MAP = {
    "Bữa sáng": "#60a5fa",
    "Bữa trưa": "#3b82f6",
    "Bữa tối": "#2563eb",
    "Bữa phụ": "#93c5fd",
}
NUTRIENT_NAME_MAP = {
    "Energy": "Calories_kcal",
    "Protein": "Protein_g",
    "Total lipid (fat)": "Fat_g",
    "Carbohydrate, by difference": "Carb_g",
    "Fiber, total dietary": "Fiber_g",
    "Sodium, Na": "Sodium_mg",
    "Total Sugars": "Sugar_g",
    "Sugars, Total": "Sugar_g",
}
VEG_CATEGORIES = {
    "Fruits and Fruit Juices",
    "Vegetables and Vegetable Products",
    "Cereal Grains and Pasta",
    "Legumes and Legume Products",
    "Nut and Seed Products",
    "Baked Products",
    "Beverages",
    "Fats and Oils",
    "Spices and Herbs",
    "Breakfast Cereals",
    "Baby Foods",
    "Soups, Sauces, and Gravies",
    "Sweets",
}
DIET_GUIDANCE = {
    "Diabetic": "Tập trung kiểm soát đường huyết bằng cách chọn thực phẩm giàu chất xơ, đạm vừa đủ và hạn chế đường đơn.",
    "DASH": "Ưu tiên thực phẩm ít natri, giàu rau củ và kiểm soát chất béo để hỗ trợ huyết áp ổn định.",
    "Weight_Loss": "Tăng tỷ lệ rau xanh, đạm nạc và các món có mật độ năng lượng hợp lý để hỗ trợ giảm cân an toàn.",
    "Weight_Gain": "Ưu tiên thực phẩm giàu năng lượng, đạm và chất béo tốt để hỗ trợ tăng cân có kiểm soát.",
    "Standard": "Duy trì khẩu phần cân bằng giữa tinh bột, đạm, rau củ và trái cây cho sinh hoạt hằng ngày.",
}
DIET_DISPLAY_MAP = {
    "Diabetic": "Kiểm soát đường huyết",
    "DASH": "DASH giảm natri",
    "Weight_Loss": "Hỗ trợ giảm cân",
    "Weight_Gain": "Tăng cân lành mạnh",
    "Standard": "Chế độ ăn cân bằng",
}
STUDENT_NAME = "Lê Ngọc Ánh"
STUDENT_ID = "22T1020019"
PROJECT_TITLE = (
    "Phân loại chế độ ăn và đề xuất thực đơn dựa trên chỉ số BMI của người mắc "
    "tiểu đường, cao huyết áp bằng mô hình Random Forest nhằm giảm thiểu rủi ro dinh dưỡng."
)

st.set_page_config(page_title="Gợi ý thực đơn theo BMI", page_icon="🥗", layout="wide")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');
        :root {
            --bg: #f6faff;
            --paper: #ffffff;
            --ink: #0f172a;
            --muted: #6b7280;
            --accent: #1d4ed8;
            --accent-soft: #eff6ff;
            --border: #dbeafe;
            --shadow: 0 14px 30px rgba(37,99,235,0.08);
        }
        html, body, [class*="css"] {
            font-family: 'Be Vietnam Pro', sans-serif;
        }
        .stApp, .main {
            background: var(--bg);
            color: var(--ink);
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 2.4rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f2f78 0%, #1d4ed8 62%, #173677 100%);
        }
        [data-testid="stSidebar"] * {
            color: #eff6ff !important;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(191,219,254,0.24);
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 10px;
        }
        .hero-card {
            background: linear-gradient(135deg, #0f3ba8 0%, #1d4ed8 48%, #38bdf8 100%);
            border-radius: 18px;
            padding: 2.5rem 3rem;
            margin-bottom: 2rem;
            color: white;
            position: relative;
            overflow: hidden;
            box-shadow: var(--shadow);
        }
        .hero-card::before {
            content: '';
            position: absolute;
            top: -40px;
            right: -40px;
            width: 200px;
            height: 200px;
            background: rgba(255,255,255,0.05);
            border-radius: 50%;
        }
        .hero-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.18);
            color: #eff6ff;
            font-size: 0.78rem;
            font-weight: 600;
            margin-bottom: 10px;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .hero-title {
            font-family: 'Playfair Display', serif;
            font-size: 2.2rem;
            line-height: 1.2;
            margin: 0 0 0.5rem 0;
        }
        .hero-text {
            font-size: 1rem;
            opacity: 0.88;
            margin: 0;
            max-width: 840px;
            line-height: 1.7;
        }
        .glass-card, .meal-card, .stat-card, .metric-card, .result-card, .food-card,
        .stDataFrame, div[data-testid="stExpander"], div[data-testid="stForm"] {
            background: white;
            border-radius: 14px;
            box-shadow: var(--shadow);
        }
        .glass-card {
            padding: 1.15rem 1.3rem;
            border: 1px solid var(--border);
            margin-bottom: 1rem;
        }
        .section-header {
            font-family: 'Playfair Display', serif;
            font-size: 1.5rem;
            color: #0f3ba8;
            border-bottom: 3px solid #60a5fa;
            padding-bottom: 8px;
            margin: 1.8rem 0 0.5rem 0;
        }
        .section-copy {
            color: var(--muted);
            line-height: 1.65;
            margin: 0.35rem 0 0.8rem 0;
            font-size: 0.95rem;
        }
        .metric-card {
            padding: 1.4rem 1.6rem;
            border: 1px solid #dbeafe;
            border-left: 5px solid #2563eb;
            margin-bottom: 1rem;
        }
        .metric-card .label {
            font-size: 0.8rem;
            color: #6b7280;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-card .value {
            font-size: 2rem;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.1;
        }
        .metric-card .sub {
            font-size: 0.85rem;
            color: #9ca3af;
            margin-top: 2px;
            line-height: 1.5;
        }
        .result-card {
            border-radius: 16px;
            padding: 1.6rem 1.8rem;
            margin: 1rem 0;
            border: 2px solid;
        }
        .result-positive { background: #eff6ff; border-color: #2563eb; }
        .result-warning  { background: #fffbeb; border-color: #f59e0b; }
        .result-danger   { background: #fef2f2; border-color: #ef4444; }
        .badge {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            margin: 3px 6px 3px 0;
            background: #dbeafe;
            color: #1e40af;
        }
        .info-chip {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            margin: 3px 6px 3px 0;
            background: #dbeafe;
            color: #1d4ed8;
        }
        .food-card {
            padding: 1.2rem;
            margin-bottom: 0.8rem;
            border-top: 4px solid;
        }
        .meal-image-wrap {
            background: linear-gradient(180deg, #ffffff 0%, #eff6ff 100%);
            border: 1px solid #dbeafe;
            border-radius: 18px;
            padding: 0.75rem;
            box-shadow: var(--shadow);
            margin-bottom: 0.35rem;
        }
        .meal-image-wrap img {
            width: 100%;
            height: 190px;
            object-fit: cover;
            border-radius: 14px;
            display: block;
            background: #eff6ff;
        }
        .meal-image-caption {
            font-size: 0.9rem;
            color: #64748b;
            text-align: center;
            margin-top: 0.55rem;
        }
        .food-card-breakfast { border-color: #60a5fa; }
        .food-card-lunch { border-color: #3b82f6; }
        .food-card-dinner { border-color: #2563eb; }
        .food-card-snack { border-color: #93c5fd; }
        .food-title {
            font-weight: 700;
            font-size: 1rem;
            color: #0f172a;
            margin-bottom: 6px;
        }
        .food-desc {
            font-size: 0.88rem;
            color: #6b7280;
            line-height: 1.55;
            margin-bottom: 8px;
        }
        .nut-tag {
            display: inline-block;
            background: #f3f4f6;
            border-radius: 8px;
            padding: 3px 10px;
            font-size: 0.78rem;
            color: #374151;
            margin: 3px 6px 3px 0;
        }
        .info-box, .warn-box {
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin: 0.8rem 0;
            font-size: 0.9rem;
        }
        .info-box {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #1e40af;
        }
        .warn-box {
            background: #fffbeb;
            border: 1px solid #fde68a;
            color: #92400e;
        }
        .stButton>button, .stFormSubmitButton>button {
            background: linear-gradient(135deg, #1d4ed8, #38bdf8);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.78rem 1rem;
            font-weight: 700;
            box-shadow: 0 10px 22px rgba(37, 99, 235, 0.2);
        }
        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, textarea {
            background: #ffffff !important;
            border: 1px solid #d1d5db !important;
            min-height: 46px;
            border-radius: 10px !important;
        }
        div[data-testid="stForm"] {
            padding: 14px 16px 6px 16px;
            border: 1px solid #e5e7eb;
        }
        div[data-testid="stMetric"] {
            background: white;
            border-left: 5px solid #2563eb;
            border-radius: 14px;
            box-shadow: var(--shadow);
            padding: 0.9rem 1rem;
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
        <div class="section-header">{title}</div>
        <p class="section-copy"><strong>{kicker}.</strong> {copy}</p>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(title: str, value: str, description: str, confidence_text: str, variant: str = "positive") -> None:
    st.markdown(
        f"""
        <div class="result-card result-{variant}">
            <div style="font-size:0.85rem; font-weight:600; color:#6b7280; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;">{title}</div>
            <div style="font-size:1.6rem; font-weight:800; color:#1a3a2a; margin-bottom:8px;">{value}</div>
            <div style="font-size:0.95rem; color:#374151; margin-bottom:12px;">{description}</div>
            <div style="font-size:0.85rem; color:#6b7280;">{confidence_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_box(message: str, warning: bool = False) -> None:
    css_class = "warn-box" if warning else "info-box"
    st.markdown(f'<div class="{css_class}">{message}</div>', unsafe_allow_html=True)


def translate_diet_class(label: str) -> str:
    return DIET_DISPLAY_MAP.get(label, label)


@st.cache_data
def svg_path_to_data_uri(path_str: str) -> str:
    path = Path(path_str)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def prepare_figure(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#f6faff")
    ax.set_facecolor("#f6faff")
    return fig, ax


def style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors="#334155")
    ax.title.set_color("#0f172a")
    if grid_axis:
        ax.grid(axis=grid_axis, linestyle="--", alpha=0.18, color="#60a5fa")


def add_bar_labels(
    ax: plt.Axes,
    bars,
    value_format: str = "{:.0f}",
    padding: float = 3.0,
    orientation: str = "vertical",
) -> None:
    for bar in bars:
        if orientation == "horizontal":
            value = bar.get_width()
            ax.text(
                bar.get_width() + padding,
                bar.get_y() + bar.get_height() / 2,
                value_format.format(value),
                va="center",
                ha="left",
                fontsize=9,
                color="#0f172a",
                fontweight="bold",
            )
            continue

        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + padding,
            value_format.format(value),
            va="bottom",
            ha="center",
            fontsize=9,
            color="#0f172a",
            fontweight="bold",
        )


def render_meal_image(meal_name: str, fallback_path: Path) -> None:
    photo_url = MEAL_PHOTO_MAP.get(meal_name, "")
    fallback_uri = svg_path_to_data_uri(str(fallback_path))
    st.markdown(
        f"""
        <div class="meal-image-wrap">
            <img src="{photo_url}" alt="Minh họa {meal_name.lower()}" onerror="this.onerror=null;this.src='{fallback_uri}';">
            <div class="meal-image-caption">Minh họa {meal_name.lower()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


DIET_RULES = {
    "Diabetic": {
        "goal": "Kiểm soát đường huyết và giảm tải đường đơn trong ngày.",
        "filters": {"max_sugar": 12, "max_carb": 35, "max_sodium": 220, "min_fiber": 1.0},
        "template": {
            "Bữa sáng": ["whole_grain", "dairy_egg", "fruit"],
            "Bữa trưa": ["seafood", "vegetable", "legume"],
            "Bữa tối": ["poultry", "vegetable", "whole_grain"],
            "Bữa phụ": ["fruit", "nut_seed"],
        },
    },
    "DASH": {
        "goal": "Giảm natri, cân bằng chất béo và tăng lượng rau củ giàu vi chất.",
        "filters": {"max_sodium": 140, "max_fat": 12, "min_fiber": 1.0},
        "template": {
            "Bữa sáng": ["fruit", "whole_grain", "dairy_egg"],
            "Bữa trưa": ["legume", "vegetable", "seafood"],
            "Bữa tối": ["poultry", "vegetable", "whole_grain"],
            "Bữa phụ": ["fruit", "nut_seed"],
        },
    },
    "Weight_Loss": {
        "goal": "Giảm mật độ năng lượng, ưu tiên no lâu và kiểm soát khẩu phần.",
        "filters": {"max_calories": 180, "max_fat": 10, "min_fiber": 1.5},
        "template": {
            "Bữa sáng": ["fruit", "dairy_egg"],
            "Bữa trưa": ["poultry", "vegetable", "legume"],
            "Bữa tối": ["seafood", "vegetable", "whole_grain"],
            "Bữa phụ": ["fruit"],
        },
    },
    "Weight_Gain": {
        "goal": "Tăng năng lượng và đạm tốt nhưng vẫn giữ cấu trúc bữa ăn cân bằng.",
        "filters": {"min_calories": 70, "min_protein": 2.0},
        "template": {
            "Bữa sáng": ["whole_grain", "dairy_egg", "nut_seed"],
            "Bữa trưa": ["poultry", "whole_grain", "vegetable"],
            "Bữa tối": ["seafood", "legume", "whole_grain"],
            "Bữa phụ": ["fruit", "nut_seed"],
        },
    },
    "Standard": {
        "goal": "Giữ cấu trúc dinh dưỡng ổn định cho nhu cầu hằng ngày.",
        "filters": {"max_sodium": 280},
        "template": {
            "Bữa sáng": ["fruit", "whole_grain", "dairy_egg"],
            "Bữa trưa": ["poultry", "vegetable", "whole_grain"],
            "Bữa tối": ["seafood", "vegetable", "legume"],
            "Bữa phụ": ["fruit", "nut_seed"],
        },
    },
}
COMPONENT_LIBRARY = {
    "fruit": {
        "label": "Trái cây",
        "categories": ["Fruits and Fruit Juices"],
        "description": "Bổ sung vitamin, khoáng chất và chất xơ cho bữa ăn.",
    },
    "whole_grain": {
        "label": "Tinh bột nền",
        "categories": ["Cereal Grains and Pasta"],
        "description": "Cung cấp năng lượng nền và giúp bữa ăn ổn định hơn.",
    },
    "vegetable": {
        "label": "Rau củ",
        "categories": ["Vegetables and Vegetable Products"],
        "description": "Tăng chất xơ, thể tích bữa ăn và hỗ trợ kiểm soát chuyển hóa.",
    },
    "legume": {
        "label": "Đậu và họ đậu",
        "categories": ["Legumes and Legume Products"],
        "description": "Bổ sung đạm thực vật và chất xơ hòa tan.",
    },
    "nut_seed": {
        "label": "Hạt và quả hạch",
        "categories": ["Nut and Seed Products"],
        "description": "Bổ sung chất béo tốt và tăng độ no cho bữa phụ.",
    },
    "seafood": {
        "label": "Cá và hải sản",
        "categories": ["Finfish and Shellfish Products"],
        "description": "Nguồn đạm nạc phù hợp với nhiều nhóm bệnh nền.",
    },
    "poultry": {
        "label": "Thịt gia cầm",
        "categories": ["Poultry Products"],
        "description": "Nguồn đạm dễ sắp xếp vào bữa chính.",
    },
    "dairy_egg": {
        "label": "Sữa hoặc trứng",
        "categories": ["Dairy and Egg Products"],
        "description": "Bổ sung đạm và canxi ở mức vừa phải.",
    },
}
MEAL_IMAGE_MAP = {
    "Bữa sáng": ASSETS_DIR / "meal-breakfast.svg",
    "Bữa trưa": ASSETS_DIR / "meal-lunch.svg",
    "Bữa tối": ASSETS_DIR / "meal-dinner.svg",
    "Bữa phụ": ASSETS_DIR / "meal-snack.svg",
}
MEAL_PHOTO_MAP = {
    "Bữa sáng": "https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=1200&q=80&auto=format&fit=crop",
    "Bữa trưa": "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=1200&q=80&auto=format&fit=crop",
    "Bữa tối": "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=1200&q=80&auto=format&fit=crop",
    "Bữa phụ": "https://images.unsplash.com/photo-1490645935967-10de6ba17061?w=1200&q=80&auto=format&fit=crop",
}


@st.cache_data
def load_food_data(
    food_path: str,
    category_path: str,
    foundation_path: str,
    nutrient_path: str,
    food_nutrient_path: str,
    signature: tuple[float, ...],
) -> pd.DataFrame:
    food_df = pd.read_csv(food_path, usecols=["fdc_id", "description", "food_category_id"])
    foundation_df = pd.read_csv(foundation_path, usecols=["fdc_id"])
    category_df = pd.read_csv(category_path, usecols=["id", "description"]).rename(
        columns={"description": "Category_Name"}
    )
    nutrient_df = pd.read_csv(nutrient_path, usecols=["id", "name"])

    foundation_food_df = (
        food_df.merge(foundation_df, on="fdc_id", how="inner")
        .merge(category_df, left_on="food_category_id", right_on="id", how="left")
        .rename(columns={"description": "Name"})
    )

    selected_nutrients = nutrient_df[nutrient_df["name"].isin(NUTRIENT_NAME_MAP)].copy()
    nutrient_ids = set(selected_nutrients["id"].tolist())
    foundation_ids = set(foundation_food_df["fdc_id"].tolist())

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(food_nutrient_path, usecols=["fdc_id", "nutrient_id", "amount"], chunksize=200000):
        filtered = chunk[
            chunk["fdc_id"].isin(foundation_ids) & chunk["nutrient_id"].isin(nutrient_ids)
        ]
        if not filtered.empty:
            chunks.append(filtered)

    nutrient_values = pd.concat(chunks, ignore_index=True)
    nutrient_values = nutrient_values.merge(
        selected_nutrients.rename(columns={"id": "nutrient_id"}), on="nutrient_id", how="left"
    )
    nutrient_values["Metric"] = nutrient_values["name"].map(NUTRIENT_NAME_MAP)
    pivot_df = (
        nutrient_values.pivot_table(index="fdc_id", columns="Metric", values="amount", aggfunc="first")
        .reset_index()
        .fillna(0)
    )

    merged_df = foundation_food_df.merge(pivot_df, on="fdc_id", how="left").fillna(0)
    merged_df["Veg_Non"] = merged_df["Category_Name"].apply(
        lambda value: "veg" if value in VEG_CATEGORIES else "non-veg"
    )

    numeric_columns = ["Calories_kcal", "Protein_g", "Fat_g", "Carb_g", "Fiber_g", "Sodium_mg", "Sugar_g"]
    for column in numeric_columns:
        merged_df[column] = merged_df[column].astype(float)

    return merged_df[
        [
            "fdc_id",
            "Name",
            "Category_Name",
            "Veg_Non",
            "Calories_kcal",
            "Protein_g",
            "Fat_g",
            "Carb_g",
            "Fiber_g",
            "Sodium_mg",
            "Sugar_g",
        ]
    ].drop_duplicates(subset=["fdc_id"])


@st.cache_data
def load_patient_data(file_path: str, file_mtime: float) -> pd.DataFrame:
    return pd.read_csv(file_path)


@st.cache_resource
def load_model_bundle() -> dict:
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


def apply_veg_filter(food_df: pd.DataFrame, veg_choice: str) -> pd.DataFrame:
    if veg_choice == "Món chay":
        return food_df[food_df["Veg_Non"] == "veg"].copy()
    if veg_choice == "Món không chay":
        return food_df[food_df["Veg_Non"] == "non-veg"].copy()
    return food_df.copy()


def apply_diet_filters(food_df: pd.DataFrame, diet_class: str) -> pd.DataFrame:
    filtered = food_df.copy()
    rules = DIET_RULES[diet_class]["filters"]

    if "max_sugar" in rules:
        filtered = filtered[filtered["Sugar_g"] <= rules["max_sugar"]]
    if "max_carb" in rules:
        filtered = filtered[filtered["Carb_g"] <= rules["max_carb"]]
    if "max_sodium" in rules:
        filtered = filtered[filtered["Sodium_mg"] <= rules["max_sodium"]]
    if "max_fat" in rules:
        filtered = filtered[filtered["Fat_g"] <= rules["max_fat"]]
    if "max_calories" in rules:
        filtered = filtered[filtered["Calories_kcal"] <= rules["max_calories"]]
    if "min_fiber" in rules:
        filtered = filtered[filtered["Fiber_g"] >= rules["min_fiber"]]
    if "min_calories" in rules:
        filtered = filtered[filtered["Calories_kcal"] >= rules["min_calories"]]
    if "min_protein" in rules:
        filtered = filtered[filtered["Protein_g"] >= rules["min_protein"]]

    return filtered if not filtered.empty else food_df.copy()


def score_foods(food_df: pd.DataFrame, diet_class: str) -> pd.Series:
    if diet_class == "Diabetic":
        return (
            food_df["Protein_g"] * 1.3
            + food_df["Fiber_g"] * 1.6
            - food_df["Sugar_g"] * 1.7
            - food_df["Carb_g"] * 0.15
            - food_df["Sodium_mg"] * 0.01
        )
    if diet_class == "DASH":
        return (
            food_df["Protein_g"] * 1.0
            + food_df["Fiber_g"] * 1.3
            - food_df["Sodium_mg"] * 0.015
            - food_df["Fat_g"] * 0.18
        )
    if diet_class == "Weight_Loss":
        return (
            food_df["Protein_g"] * 1.4
            + food_df["Fiber_g"] * 1.8
            - food_df["Calories_kcal"] * 0.03
            - food_df["Fat_g"] * 0.20
        )
    if diet_class == "Weight_Gain":
        return (
            food_df["Calories_kcal"] * 0.04
            + food_df["Protein_g"] * 1.1
            + food_df["Fat_g"] * 0.35
            + food_df["Carb_g"] * 0.08
        )
    return (
        food_df["Protein_g"] * 1.1
        + food_df["Fiber_g"] * 0.8
        - (food_df["Sodium_mg"] / 300.0)
        - (food_df["Sugar_g"] / 12.0)
    )


def pick_food_for_component(
    food_df: pd.DataFrame,
    component_key: str,
    diet_class: str,
    veg_choice: str,
    used_fdc_ids: set[int],
) -> pd.Series | None:
    component = COMPONENT_LIBRARY[component_key]
    candidate_df = apply_veg_filter(food_df, veg_choice)
    candidate_df = apply_diet_filters(candidate_df, diet_class)
    candidate_df = candidate_df[candidate_df["Category_Name"].isin(component["categories"])]
    candidate_df = candidate_df[~candidate_df["fdc_id"].isin(used_fdc_ids)]

    if candidate_df.empty:
        candidate_df = apply_veg_filter(food_df, veg_choice)
        candidate_df = candidate_df[candidate_df["Category_Name"].isin(component["categories"])]
        candidate_df = candidate_df[~candidate_df["fdc_id"].isin(used_fdc_ids)]

    if candidate_df.empty:
        return None

    candidate_df = candidate_df.assign(Score=score_foods(candidate_df, diet_class)).sort_values(
        "Score", ascending=False
    )
    return candidate_df.iloc[0]


def build_meal_plan(food_df: pd.DataFrame, diet_class: str, veg_choice: str) -> list[dict]:
    used_fdc_ids: set[int] = set()
    plan: list[dict] = []

    for meal_name, component_keys in DIET_RULES[diet_class]["template"].items():
        selected_rows: list[pd.Series] = []
        notes: list[str] = []

        for component_key in component_keys:
            row = pick_food_for_component(food_df, component_key, diet_class, veg_choice, used_fdc_ids)
            if row is None:
                continue
            used_fdc_ids.add(int(row["fdc_id"]))
            selected_rows.append(row)
            notes.append(COMPONENT_LIBRARY[component_key]["description"])

        if not selected_rows:
            continue

        meal_df = pd.DataFrame(selected_rows)
        totals = meal_df[["Calories_kcal", "Protein_g", "Fat_g", "Carb_g", "Fiber_g", "Sodium_mg", "Sugar_g"]].sum()
        plan.append(
            {
                "meal_name": meal_name,
                "goal": DIET_RULES[diet_class]["goal"],
                "image_path": MEAL_IMAGE_MAP[meal_name],
                "items": meal_df.reset_index(drop=True),
                "totals": totals,
                "rationale": " ".join(dict.fromkeys(notes)),
            }
        )

    return plan


def summarize_meal_plan(meal_plan: list[dict]) -> pd.DataFrame:
    records = []
    for meal in meal_plan:
        totals = meal["totals"]
        records.append(
            {
                "Bữa ăn": meal["meal_name"],
                "Calories_kcal": totals["Calories_kcal"],
                "Protein_g": totals["Protein_g"],
                "Carb_g": totals["Carb_g"],
                "Fiber_g": totals["Fiber_g"],
                "Sodium_mg": totals["Sodium_mg"],
            }
        )
    return pd.DataFrame(records)


def plot_class_distribution(patient_df: pd.DataFrame) -> plt.Figure:
    counts = patient_df["Diet_Class"].value_counts().sort_values(ascending=False)
    labels = [translate_diet_class(label) for label in counts.index]
    fig, ax = prepare_figure((7, 4))
    bars = ax.bar(labels, counts.values, color=BLUE_PALETTE[: len(counts)], edgecolor="#1d4ed8", linewidth=1.5)
    ax.set_title("Phân bố các nhóm chế độ ăn", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Nhóm chế độ ăn")
    ax.set_ylabel("Số lượng mẫu")
    ax.tick_params(axis="x", rotation=20)
    style_axes(ax, "y")
    add_bar_labels(ax, bars, "{:.0f}", 6.0)
    fig.tight_layout()
    return fig


def plot_bmi_by_class(patient_df: pd.DataFrame) -> plt.Figure:
    labels = list(patient_df["Diet_Class"].dropna().unique())
    series = [patient_df.loc[patient_df["Diet_Class"] == label, "BMI"].dropna() for label in labels]
    fig, ax = prepare_figure((7, 4))
    box = ax.boxplot(series, tick_labels=[translate_diet_class(label) for label in labels], patch_artist=True)
    for patch, color in zip(box["boxes"], BLUE_PALETTE):
        patch.set_facecolor(color)
        patch.set_edgecolor("#1d4ed8")
        patch.set_linewidth(1.6)
    for median in box["medians"]:
        median.set_color("#0f172a")
        median.set_linewidth(2)
    ax.set_title("Phân bố BMI theo từng nhóm chế độ ăn", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Nhóm chế độ ăn")
    ax.set_ylabel("BMI")
    ax.tick_params(axis="x", rotation=20)
    style_axes(ax, "y")
    fig.tight_layout()
    return fig


def plot_condition_rate(patient_df: pd.DataFrame) -> plt.Figure:
    grouped = patient_df.groupby("Diet_Class")[["Diabetes", "Hypertension"]].mean().mul(100)
    fig, ax = prepare_figure((7.2, 4))
    positions = list(range(len(grouped)))
    bars_diabetes = ax.bar(
        [position - 0.18 for position in positions],
        grouped["Diabetes"],
        width=0.36,
        color="#60a5fa",
        label="Tiểu đường",
    )
    bars_hypertension = ax.bar(
        [position + 0.18 for position in positions],
        grouped["Hypertension"],
        width=0.36,
        color="#1d4ed8",
        label="Cao huyết áp",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([translate_diet_class(label) for label in grouped.index], rotation=20)
    ax.set_ylabel("Tỷ lệ (%)")
    ax.set_title("Tỷ lệ bệnh lý theo nhóm chế độ ăn", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=False)
    style_axes(ax, "y")
    add_bar_labels(ax, bars_diabetes, "{:.0f}%", 1.2)
    add_bar_labels(ax, bars_hypertension, "{:.0f}%", 1.2)
    fig.tight_layout()
    return fig


def plot_food_category_distribution(food_df: pd.DataFrame) -> plt.Figure:
    counts = food_df["Category_Name"].value_counts().head(8).sort_values()
    fig, ax = prepare_figure((7.2, 4.2))
    bars = ax.barh(counts.index, counts.values, color="#3b82f6", edgecolor="#1d4ed8", linewidth=1.4)
    ax.set_title("Phân bố nhóm thực phẩm trong USDA", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Số lượng mẫu")
    style_axes(ax, "x")
    add_bar_labels(ax, bars, "{:.0f}", 12.0, orientation="horizontal")
    fig.tight_layout()
    return fig


def plot_user_profile(user_profile: dict, patient_df: pd.DataFrame) -> plt.Figure:
    chart_df = pd.DataFrame(
        {
            "Chỉ số": ["Tuổi", "BMI", "Tiểu đường", "Cao huyết áp"],
            "Người dùng": [
                user_profile["Age"],
                user_profile["BMI"],
                user_profile["Diabetes"] * 100,
                user_profile["Hypertension"] * 100,
            ],
            "Trung bình dữ liệu": [
                patient_df["Age"].mean(),
                patient_df["BMI"].mean(),
                patient_df["Diabetes"].mean() * 100,
                patient_df["Hypertension"].mean() * 100,
            ],
        }
    )
    fig, ax = prepare_figure((7.2, 4.2))
    positions = list(range(len(chart_df)))
    bars_user = ax.bar(
        [position - 0.18 for position in positions],
        chart_df["Người dùng"],
        width=0.36,
        color="#60a5fa",
        label="Người dùng",
    )
    bars_avg = ax.bar(
        [position + 0.18 for position in positions],
        chart_df["Trung bình dữ liệu"],
        width=0.36,
        color="#1d4ed8",
        label="Trung bình dữ liệu",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(chart_df["Chỉ số"])
    ax.set_title("Hồ sơ đầu vào so với dữ liệu tham chiếu", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=False)
    style_axes(ax, "y")
    add_bar_labels(ax, bars_user, "{:.1f}", 1.4)
    add_bar_labels(ax, bars_avg, "{:.1f}", 1.4)
    fig.tight_layout()
    return fig


def plot_probability_chart(confidence_df: pd.DataFrame) -> plt.Figure:
    chart_df = confidence_df.sort_values("Probability", ascending=True).copy()
    chart_df["Diet_Class_VI"] = chart_df["Diet_Class"].map(translate_diet_class)
    fig, ax = prepare_figure((7, 3.8))
    colors = BLUE_PALETTE[-len(chart_df) :]
    bars = ax.barh(chart_df["Diet_Class_VI"], chart_df["Probability"], color=colors, edgecolor="#1d4ed8", linewidth=1.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Xác suất")
    ax.set_title("Phân bố xác suất dự đoán", fontsize=12, fontweight="bold", pad=12)
    style_axes(ax, "x")
    for bar, value in zip(bars, chart_df["Probability"]):
        ax.text(min(value + 0.02, 0.98), bar.get_y() + bar.get_height() / 2, f"{value:.1%}", va="center", color="#152033")
    fig.tight_layout()
    return fig


def plot_meal_calories(meal_summary_df: pd.DataFrame) -> plt.Figure:
    fig, ax = prepare_figure((7.2, 4.2))
    colors = [MEAL_ACCENT_MAP.get(name, "#60a5fa") for name in meal_summary_df["Bữa ăn"]]
    calories = meal_summary_df["Calories_kcal"]
    labels = meal_summary_df["Bữa ăn"]
    wedges, _, autotexts = ax.pie(
        calories,
        labels=labels,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=lambda pct: f"{pct:.0f}%",
        pctdistance=0.78,
        wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2),
        textprops=dict(color="#334155", fontsize=10, fontweight="semibold"),
    )
    total = float(calories.sum())
    ax.text(0, 0.08, "Tổng năng lượng", ha="center", va="center", fontsize=10, color="#64748b")
    ax.text(0, -0.08, f"{total:.0f} kcal", ha="center", va="center", fontsize=15, fontweight="bold", color="#0f172a")
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(10)
    ax.set_title("Tỷ lệ năng lượng theo từng bữa", fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def plot_daily_nutrients(meal_summary_df: pd.DataFrame) -> plt.Figure:
    totals = meal_summary_df[["Protein_g", "Carb_g", "Fiber_g", "Sodium_mg"]].sum()
    label_map = {
        "Protein_g": "Đạm (g)",
        "Carb_g": "Bột đường (g)",
        "Fiber_g": "Chất xơ (g)",
        "Sodium_mg": "Natri (mg)",
    }
    chart_df = pd.DataFrame(
        {"Dưỡng chất": [label_map[column] for column in totals.index], "Giá trị": totals.values}
    )
    fig, ax = prepare_figure((7.2, 4.2))
    bars = ax.bar(chart_df["Dưỡng chất"], chart_df["Giá trị"], color=["#bfdbfe", "#93c5fd", "#60a5fa", "#1d4ed8"], edgecolor="white", linewidth=1.6)
    ax.set_title("Tổng dưỡng chất của thực đơn gợi ý", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Giá trị cộng dồn")
    ax.tick_params(axis="x", rotation=12)
    style_axes(ax, "y")
    add_bar_labels(ax, bars, "{:.0f}", 8.0)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(confusion_matrix: list[list[int]], labels: list[str]) -> plt.Figure:
    matrix_df = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    fig, ax = prepare_figure((8, 5))
    image = ax.imshow(matrix_df.values, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([translate_diet_class(label) for label in labels], rotation=30, ha="right")
    ax.set_yticklabels([translate_diet_class(label) for label in labels])
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thực tế")
    ax.set_title("Ma trận nhầm lẫn", fontsize=12, fontweight="bold", pad=12)
    threshold = matrix_df.values.max() * 0.5 if matrix_df.values.size else 0
    for row in range(len(labels)):
        for col in range(len(labels)):
            value = matrix_df.iloc[row, col]
            text_color = "white" if value > threshold else "#152033"
            ax.text(col, row, value, ha="center", va="center", color=text_color, fontsize=10, fontweight="bold")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_feature_importance(feature_names: list[str], importances: list[float]) -> plt.Figure:
    feature_df = pd.DataFrame({"Đặc trưng": feature_names, "Mức độ quan trọng": importances}).sort_values(
        "Mức độ quan trọng", ascending=True
    )
    translate_map = {"Age": "Tuổi", "BMI": "BMI", "Diabetes": "Tiểu đường", "Hypertension": "Cao huyết áp"}
    feature_df["Đặc trưng"] = feature_df["Đặc trưng"].map(lambda value: translate_map.get(value, value))
    fig, ax = prepare_figure((8, 4.5))
    bars = ax.barh(feature_df["Đặc trưng"], feature_df["Mức độ quan trọng"], color="#3b82f6", edgecolor="#1d4ed8", linewidth=1.4)
    ax.set_title("Mức độ quan trọng của đặc trưng", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Mức độ quan trọng")
    style_axes(ax, "x")
    add_bar_labels(ax, bars, "{:.2f}", 0.01, orientation="horizontal")
    fig.tight_layout()
    return fig


def plot_class_metrics(report_df: pd.DataFrame, labels: list[str]) -> plt.Figure:
    class_report = report_df.loc[labels, ["precision", "recall", "f1-score"]]
    fig, ax = prepare_figure((8, 4.5))
    positions = list(range(len(class_report)))
    bars_precision = ax.bar([position - 0.22 for position in positions], class_report["precision"], width=0.22, color="#bfdbfe", label="Độ chuẩn")
    bars_recall = ax.bar([position for position in positions], class_report["recall"], width=0.22, color="#60a5fa", label="Độ bao phủ")
    bars_f1 = ax.bar([position + 0.22 for position in positions], class_report["f1-score"], width=0.22, color="#1d4ed8", label="F1-score")
    ax.set_xticks(positions)
    ax.set_xticklabels([translate_diet_class(label) for label in class_report.index], rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_title("So sánh chất lượng theo từng lớp", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=False, ncol=3)
    style_axes(ax, "y")
    add_bar_labels(ax, bars_precision, "{:.2f}", 0.02)
    add_bar_labels(ax, bars_recall, "{:.2f}", 0.02)
    add_bar_labels(ax, bars_f1, "{:.2f}", 0.02)
    fig.tight_layout()
    return fig


def plot_top_errors(confusion_matrix: list[list[int]], labels: list[str]) -> plt.Figure:
    error_records = []
    for actual_index, actual_label in enumerate(labels):
        for predicted_index, predicted_label in enumerate(labels):
            if actual_index == predicted_index:
                continue
            value = confusion_matrix[actual_index][predicted_index]
            if value > 0:
                error_records.append(
                    {
                        "Cặp nhầm lẫn": f"{translate_diet_class(actual_label)} → {translate_diet_class(predicted_label)}",
                        "Số lượng": value,
                    }
                )
    if not error_records:
        fig, ax = prepare_figure((7, 3.5))
        ax.text(0.5, 0.5, "Không có lỗi ngoài đường chéo", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig
    error_df = pd.DataFrame(error_records).sort_values("Số lượng", ascending=True).tail(6)
    fig, ax = prepare_figure((7.8, 4.2))
    bars = ax.barh(error_df["Cặp nhầm lẫn"], error_df["Số lượng"], color="#60a5fa", edgecolor="#1d4ed8", linewidth=1.4)
    ax.set_title("Các cặp nhầm lẫn xuất hiện nhiều nhất", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Số lượng")
    style_axes(ax, "x")
    add_bar_labels(ax, bars, "{:.0f}", 0.25, orientation="horizontal")
    fig.tight_layout()
    return fig


def render_meal_plan(meal_plan: list[dict]) -> None:
    meal_style_map = {
        "Bữa sáng": "food-card-breakfast",
        "Bữa trưa": "food-card-lunch",
        "Bữa tối": "food-card-dinner",
        "Bữa phụ": "food-card-snack",
    }
    for start_index in range(0, len(meal_plan), 2):
        columns = st.columns(2)
        for column, meal in zip(columns, meal_plan[start_index : start_index + 2]):
            with column:
                img_col, info_col = st.columns([1, 2.2])
                with img_col:
                    render_meal_image(meal["meal_name"], meal["image_path"])
                with info_col:
                    totals = meal["totals"]
                    nutrition_tags = (
                        f'<span class="nut-tag">{totals["Calories_kcal"]:.0f} kcal</span>'
                        f'<span class="nut-tag">{totals["Protein_g"]:.1f}g đạm</span>'
                        f'<span class="nut-tag">{totals["Fiber_g"]:.1f}g chất xơ</span>'
                        f'<span class="nut-tag">{totals["Sodium_mg"]:.0f}mg natri</span>'
                    )
                    items_html = "".join(
                        f"<li><strong>{row['Name']}</strong> <span style='color:#6b7280;'>({row['Category_Name']})</span></li>"
                        for _, row in meal["items"].iterrows()
                    )
                    st.markdown(
                        f"""
                        <div class="food-card {meal_style_map.get(meal['meal_name'], 'food-card-lunch')}">
                            <div style="color:#6b7280; font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">{meal['meal_name']}</div>
                            <div class="food-title">{meal['goal']}</div>
                            <div class="food-desc">{meal['rationale']}</div>
                            <div class="food-desc" style="margin-top:10px;">
                                <ul style="padding-left:18px; margin:0;">{items_html}</ul>
                            </div>
                            <div style="margin-top:10px;">{nutrition_tags}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f"""
                    <span class="badge">{totals['Calories_kcal']:.0f} kcal</span>
                    <span class="badge">{totals['Protein_g']:.1f}g đạm</span>
                    <span class="badge">{totals['Carb_g']:.1f}g bột đường</span>
                    <span class="badge">{totals['Fiber_g']:.1f}g chất xơ</span>
                    <span class="badge">{totals['Sodium_mg']:.0f}mg natri</span>
                    """,
                    unsafe_allow_html=True,
                )


inject_styles()

food_files = {
    "food": DATA_DIR / "Nguon_1" / "food.csv",
    "category": DATA_DIR / "Nguon_1" / "food_category.csv",
    "foundation": DATA_DIR / "Nguon_1" / "foundation_food.csv",
    "nutrient": DATA_DIR / "Nguon_1" / "nutrient.csv",
    "food_nutrient": DATA_DIR / "Nguon_1" / "food_nutrient.csv",
}
food_signature = tuple(path.stat().st_mtime for path in food_files.values())
patient_path = DATA_DIR / "Nguon_2.csv"

food_df = load_food_data(
    str(food_files["food"]),
    str(food_files["category"]),
    str(food_files["foundation"]),
    str(food_files["nutrient"]),
    str(food_files["food_nutrient"]),
    food_signature,
)
patient_df = load_patient_data(str(patient_path), patient_path.stat().st_mtime)
model_bundle = load_model_bundle()

sidebar_options = [
    "1. Giới thiệu & Khám phá dữ liệu",
    "2. Triển khai mô hình",
    "3. Đánh giá & Hiệu năng",
]

st.sidebar.markdown("### Dinh dưỡng AI")
st.sidebar.caption("Phân loại chế độ ăn và gợi ý thực đơn cho người có bệnh lý nền.")
if st.sidebar.button("Tải lại dữ liệu"):
    st.cache_data.clear()
    st.rerun()
page = st.sidebar.radio("Chọn trang", sidebar_options)


if page == "1. Giới thiệu & Khám phá dữ liệu":
    render_hero(
        "Phân loại chế độ ăn và gợi ý thực đơn dựa trên chỉ số BMI",
        "Ứng dụng hỗ trợ người dùng có bệnh lý nền nhận khuyến nghị dinh dưỡng rõ ràng, có dữ liệu, có biểu đồ mô tả và có thể tương tác trực tiếp trên web app.",
        "Ứng dụng Streamlit • Random Forest • Dinh dưỡng thông minh",
    )
    st.markdown(
        '<span class="info-chip">Tiếng Việt đầy đủ có dấu</span>'
        '<span class="info-chip">Biểu đồ đồng bộ tông xanh dương</span>'
        '<span class="info-chip">Dữ liệu thực phẩm USDA + hồ sơ bệnh nhân</span>',
        unsafe_allow_html=True,
    )

    stats_col1, stats_col2, stats_col3 = st.columns(3)
    with stats_col1:
        render_stat_card("Sinh viên", STUDENT_NAME, f"MSSV: {STUDENT_ID}")
    with stats_col2:
        render_stat_card("Nguồn bệnh nhân", f"{len(patient_df):,} mẫu", "Dữ liệu tự sinh phục vụ huấn luyện mô hình.")
    with stats_col3:
        render_stat_card("Nguồn thực phẩm", f"{len(food_df):,} mẫu", "Dữ liệu USDA Foundation Foods đã được làm sạch và chuẩn hóa.")

    render_section_header("Tổng quan dự án", "Thông tin đề tài", f"<strong>Tên đề tài:</strong> {PROJECT_TITLE}")
    render_section_header(
        "Dữ liệu đầu vào",
        "Hai nguồn dữ liệu phục vụ hai nhiệm vụ khác nhau",
        "Nguồn 1 cung cấp dữ liệu thực phẩm và dinh dưỡng để gợi ý thực đơn. Nguồn 2 là hồ sơ bệnh nhân dùng để huấn luyện và đánh giá mô hình Random Forest.",
    )

    preview_col1, preview_col2 = st.columns(2)
    with preview_col1:
        st.write("Nguồn 1: dữ liệu thực phẩm USDA")
        food_preview = (
            food_df[["Name", "Category_Name", "Calories_kcal", "Protein_g", "Carb_g", "Sodium_mg"]]
            .head(10)
            .rename(
                columns={
                    "Name": "Tên thực phẩm",
                    "Category_Name": "Nhóm thực phẩm",
                    "Calories_kcal": "Năng lượng (kcal)",
                    "Protein_g": "Đạm (g)",
                    "Carb_g": "Bột đường (g)",
                    "Sodium_mg": "Natri (mg)",
                }
            )
        )
        st.dataframe(
            food_preview,
            use_container_width=True,
        )
    with preview_col2:
        st.write("Nguồn 2: dữ liệu hồ sơ bệnh nhân tự sinh")
        patient_preview = (
            patient_df.head(10)
            .rename(
                columns={
                    "Age": "Tuổi",
                    "BMI": "BMI",
                    "Diabetes": "Tiểu đường",
                    "Hypertension": "Cao huyết áp",
                    "Diet_Class": "Nhóm chế độ ăn",
                }
            )
            .copy()
        )
        if "Nhóm chế độ ăn" in patient_preview.columns:
            patient_preview["Nhóm chế độ ăn"] = patient_preview["Nhóm chế độ ăn"].map(translate_diet_class)
        st.dataframe(patient_preview, use_container_width=True)

    render_section_header(
        "EDA",
        "Khám phá dữ liệu chi tiết",
        "Các biểu đồ dưới đây mô tả phân bố nhãn, vai trò của BMI, tỷ lệ bệnh lý và cấu trúc nhóm thực phẩm đang dùng trong dự án.",
    )
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.pyplot(plot_class_distribution(patient_df))
    with row1_col2:
        st.pyplot(plot_bmi_by_class(patient_df))
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.pyplot(plot_condition_rate(patient_df))
    with row2_col2:
        st.pyplot(plot_food_category_distribution(food_df))

    render_info_box(
        "📌 Dữ liệu huấn luyện gồm 1.000 mẫu và 5 nhãn, phân bố tương đối cân bằng. BMI cùng hai cờ bệnh lý là tín hiệu mạnh giúp mô hình phân biệt các nhóm chế độ ăn, còn dữ liệu thực phẩm USDA hỗ trợ nâng cấp từ mức gợi ý món đơn lẻ sang thực đơn theo từng bữa."
    )


elif page == "2. Triển khai mô hình":
    render_hero(
        "Phân loại chế độ ăn và sinh thực đơn chi tiết",
        "Người dùng nhập hồ sơ sức khỏe, hệ thống tính BMI, dự đoán nhóm chế độ ăn và sinh thực đơn gợi ý gồm bữa sáng, trưa, tối và bữa phụ.",
        "Dự đoán tương tác trực tiếp",
    )
    render_section_header(
        "Nhập liệu",
        "Điền hồ sơ sức khỏe",
        "Các trường bên dưới được dùng để tính BMI và tạo đầu vào đúng định dạng với lúc huấn luyện mô hình.",
    )

    with st.form("prediction_form"):
        form_col1, form_col2 = st.columns(2)
        with form_col1:
            age = st.number_input("Tuổi", min_value=18, max_value=100, value=30)
            height_cm = st.number_input("Chiều cao (cm)", min_value=120, max_value=220, value=165)
            weight_kg = st.number_input("Cân nặng (kg)", min_value=25, max_value=250, value=60)
        with form_col2:
            diabetes_text = st.selectbox("Tình trạng tiểu đường", ["Không", "Có"])
            hypertension_text = st.selectbox("Tình trạng cao huyết áp", ["Không", "Có"])
            veg_choice = st.selectbox("Loại món ưu tiên", ["Tất cả", "Món chay", "Món không chay"])
        submitted = st.form_submit_button("Dự đoán và tạo thực đơn")

    if submitted:
        bmi = round(weight_kg / ((height_cm / 100) ** 2), 2)
        user_profile = {
            "Age": age,
            "BMI": bmi,
            "Diabetes": 1 if diabetes_text == "Có" else 0,
            "Hypertension": 1 if hypertension_text == "Có" else 0,
        }
        input_df = pd.DataFrame([user_profile])
        model = model_bundle["model"]
        prediction = model.predict(input_df[PATIENT_FEATURES])[0]
        probabilities = model.predict_proba(input_df[PATIENT_FEATURES])[0]
        class_names = list(model.classes_)
        confidence_df = (
            pd.DataFrame({"Diet_Class": class_names, "Probability": probabilities})
            .sort_values("Probability", ascending=False)
            .reset_index(drop=True)
        )
        meal_plan = build_meal_plan(food_df, prediction, veg_choice)
        meal_summary_df = summarize_meal_plan(meal_plan)

        result_col1, result_col2, result_col3 = st.columns(3)
        with result_col1:
            render_stat_card("BMI", f"{bmi:.2f}", "Chỉ số khối cơ thể sau khi tính từ chiều cao và cân nặng.")
        with result_col2:
            render_stat_card(
                "Nhóm chế độ ăn",
                translate_diet_class(prediction),
                "Nhãn được mô hình Random Forest dự đoán phù hợp nhất.",
            )
        with result_col3:
            render_stat_card(
                "Độ tin cậy",
                f"{float(confidence_df.loc[0, 'Probability']):.1%}",
                "Xác suất lớn nhất trong toàn bộ phân bố dự đoán.",
            )

        render_result_card(
            "Chế độ ăn được đề xuất",
            translate_diet_class(prediction),
            DIET_GUIDANCE[prediction],
            f"Độ tin cậy dự đoán: <b>{float(confidence_df.loc[0, 'Probability']):.1%}</b>",
            "positive" if float(confidence_df.loc[0, 'Probability']) >= 0.75 else "warning",
        )

        prob_col1, prob_col2 = st.columns([1.05, 0.95])
        with prob_col1:
            st.pyplot(plot_probability_chart(confidence_df))
        with prob_col2:
            display_df = confidence_df.rename(columns={"Diet_Class": "Nhóm chế độ ăn", "Probability": "Xác suất"}).copy()
            display_df["Nhóm chế độ ăn"] = display_df["Nhóm chế độ ăn"].map(translate_diet_class)
            display_df["Xác suất"] = display_df["Xác suất"].map(lambda value: f"{value:.1%}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        render_section_header(
            "Giải thích hồ sơ",
            "So sánh nhanh đầu vào với dữ liệu tham chiếu",
            "Biểu đồ giúp người dùng thấy hồ sơ hiện tại lệch thế nào so với mặt bằng dữ liệu huấn luyện.",
        )
        st.pyplot(plot_user_profile(user_profile, patient_df))

        render_section_header(
            "Thực đơn gợi ý",
            "Thực đơn cụ thể theo nhóm bệnh",
            "Hệ thống sinh gợi ý theo từng bữa và kèm giải thích dinh dưỡng phù hợp với nhóm bệnh được dự đoán.",
        )

        if meal_plan:
            render_meal_plan(meal_plan)
            nutrient_col1, nutrient_col2 = st.columns(2)
            with nutrient_col1:
                st.pyplot(plot_meal_calories(meal_summary_df))
            with nutrient_col2:
                st.pyplot(plot_daily_nutrients(meal_summary_df))
            summary_view = meal_summary_df.rename(
                columns={
                    "Calories_kcal": "Năng lượng (kcal)",
                    "Protein_g": "Đạm (g)",
                    "Carb_g": "Bột đường (g)",
                    "Fiber_g": "Chất xơ (g)",
                    "Sodium_mg": "Natri (mg)",
                }
            ).copy()
            for column in ["Năng lượng (kcal)", "Đạm (g)", "Bột đường (g)", "Chất xơ (g)", "Natri (mg)"]:
                summary_view[column] = summary_view[column].map(lambda value: round(float(value), 1))
            st.dataframe(summary_view, use_container_width=True, hide_index=True)
        else:
            render_info_box(
                "⚠️ Chưa tạo được thực đơn phù hợp từ dữ liệu hiện có. Vui lòng thử lại với lựa chọn khác.",
                warning=True,
            )


else:
    metrics = model_bundle["metrics"]
    report_df = pd.DataFrame(model_bundle["classification_report"]).transpose()
    labels = model_bundle["labels"]

    render_hero(
        "Đánh giá và hiệu năng mô hình",
        "Trang này tổng hợp chỉ số đánh giá, ma trận nhầm lẫn, chất lượng theo từng lớp và các cặp lỗi xuất hiện nhiều nhất.",
        "Đánh giá mô hình",
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        render_stat_card("Độ chính xác", f"{metrics['accuracy']:.3f}", "Kết quả trên tập kiểm tra.")
    with metric_col2:
        render_stat_card("F1-score có trọng số", f"{metrics['f1_weighted']:.3f}", "Mức cân bằng giữa độ chuẩn và độ bao phủ.")
    with metric_col3:
        render_stat_card("Số mẫu kiểm tra", str(metrics["test_size"]), "Quy mô tập dữ liệu dùng để đánh giá.")

    render_section_header(
        "Hiệu năng",
        "Chỉ số đánh giá chính",
        "Ngoài ma trận nhầm lẫn và mức độ quan trọng của đặc trưng, trang này bổ sung thêm biểu đồ chất lượng theo từng lớp để việc giải thích mô hình rõ ràng hơn.",
    )
    eval_row1_col1, eval_row1_col2 = st.columns(2)
    with eval_row1_col1:
        st.pyplot(plot_confusion_matrix(model_bundle["confusion_matrix"], labels))
    with eval_row1_col2:
        st.pyplot(plot_class_metrics(report_df, labels))
    eval_row2_col1, eval_row2_col2 = st.columns(2)
    with eval_row2_col1:
        st.pyplot(plot_feature_importance(model_bundle["feature_names_after_preprocessing"], model_bundle["feature_importances"]))
    with eval_row2_col2:
        st.pyplot(plot_top_errors(model_bundle["confusion_matrix"], labels))

    st.subheader("Báo cáo phân loại")
    report_view = report_df.copy()
    report_view.index = [
        translate_diet_class(index)
        if index in labels
        else {"accuracy": "Độ chính xác", "macro avg": "Trung bình macro", "weighted avg": "Trung bình có trọng số"}.get(index, index)
        for index in report_view.index
    ]
    report_view = report_view.rename(
        columns={"precision": "Độ chuẩn", "recall": "Độ bao phủ", "f1-score": "F1-score", "support": "Số mẫu"}
    )
    st.dataframe(report_view, use_container_width=True)
    render_info_box(
        "📌 Các cặp nhầm lẫn chủ yếu xuất hiện ở những nhóm có ngưỡng BMI gần nhau hoặc hồ sơ bệnh lý không quá khác biệt. Nhóm kiểm soát đường huyết và DASH thường ổn định hơn vì có cờ bệnh lý rõ ngay từ đầu vào."
    )
    render_info_box(
        "⚠️ Hướng cải thiện: mở rộng dữ liệu thật, thêm đặc trưng lâm sàng và tinh chỉnh luật sinh thực đơn chi tiết theo từng bệnh nền.",
        warning=True,
    )
