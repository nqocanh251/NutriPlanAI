# Phân Loại Chế Độ Ăn Và Đề Xuất Thực Đơn Bằng Random Forest

## Giới thiệu

Đây là đồ án xây dựng ứng dụng web bằng **Streamlit** để hỗ trợ:

- phân loại chế độ ăn phù hợp cho người dùng dựa trên **BMI**, **tuổi**, **tiểu đường** và **cao huyết áp**
- sinh **thực đơn gợi ý theo bữa sáng, trưa, tối, bữa phụ** sau khi mô hình dự đoán
- trực quan hóa dữ liệu và đánh giá hiệu năng mô hình ngay trên giao diện web

Ứng dụng sử dụng mô hình **Random Forest Classifier** cho bài toán phân loại và kết hợp thêm **rule-based filtering** để sinh thực đơn từ cơ sở dữ liệu thực phẩm USDA.

## Tên đề tài

**Phân loại chế độ ăn và đề xuất thực đơn dựa trên chỉ số BMI của người mắc tiểu đường, cao huyết áp bằng mô hình Random Forest nhằm giảm thiểu rủi ro dinh dưỡng.**

## Mục tiêu dự án

- Tự sinh bộ dữ liệu hồ sơ bệnh nhân gồm tuổi, chiều cao, cân nặng, BMI và cờ bệnh lý.
- Huấn luyện mô hình Random Forest để phân loại người dùng vào nhóm chế độ ăn phù hợp.
- Hiển thị các chỉ số đánh giá như `Accuracy`, `F1-score`, `Confusion Matrix`, `Feature Importance`.
- Xây dựng giao diện Streamlit trực quan, dễ thao tác và hỗ trợ nhập liệu trực tiếp.
- Kết hợp dữ liệu thực phẩm để sinh thực đơn theo từng bữa sau khi dự đoán.

## Công nghệ sử dụng

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Cấu trúc thư mục

```text
DOANCUOIKY/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   ├── Nguon_2.csv
│   └── Nguon_1/
│       ├── food.csv
│       ├── food_category.csv
│       └── ...
├── models/
│   ├── model.pkl
│   └── metrics.json
└── scripts/
    ├── code_tusinhdata.py
    └── train_model.py
```

## Nguồn dữ liệu

### 1. Dữ liệu hồ sơ bệnh nhân tự sinh

File: `data/Nguon_2.csv`

Được tạo bởi script `scripts/code_tusinhdata.py` với:

- `Age`
- `Height_cm`
- `Weight_kg`
- `BMI`
- `Diabetes`
- `Hypertension`
- `Diet_Class`

Bộ dữ liệu này gồm **1.000 mẫu** và được gán nhãn theo rule y khoa đơn giản để phục vụ huấn luyện mô hình.

### 2. Dữ liệu thực phẩm

Thư mục: `data/Nguon_1/`

Nguồn thực phẩm hiện tại được tổ chức theo dạng dữ liệu USDA Foundation Foods / FoodData style CSV. Trong code hiện tại, ứng dụng sử dụng chủ yếu:

- `food.csv`
- `food_category.csv`
- `foundation_food.csv`
- `nutrient.csv`
- `food_nutrient.csv`

Sau đó hệ thống chuẩn hóa dữ liệu dinh dưỡng, gom nhóm thực phẩm và dùng luật lọc để chọn nguyên liệu phù hợp cho từng nhóm chế độ ăn.

Lưu ý:
- Ứng dụng sử dụng dữ liệu USDA Foundation Foods đã được làm sạch để phục vụ sinh thực đơn.
- Phần thực đơn được tạo theo hướng **rule-based** từ nhóm thực phẩm và chỉ số dinh dưỡng, sau đó hiển thị theo từng bữa ăn.

## Pipeline xử lý

### Bước 1. Sinh dữ liệu bệnh nhân

Script `scripts/code_tusinhdata.py` sẽ:

- sinh tuổi, chiều cao, cân nặng ngẫu nhiên
- tính BMI
- mô phỏng xác suất mắc tiểu đường và cao huyết áp
- gán nhãn chế độ ăn:
  - `Diabetic`
  - `DASH`
  - `Weight_Loss`
  - `Weight_Gain`
  - `Standard`

### Bước 2. Tiền xử lý và huấn luyện mô hình

Script `scripts/train_model.py` thực hiện:

- đọc dữ liệu từ `data/Nguon_2.csv`
- loại bỏ giá trị không hợp lệ
- tiền xử lý bằng:
  - `StandardScaler` cho `Age`, `BMI`
  - `OneHotEncoder` cho `Diabetes`, `Hypertension`
- chia train/test theo `stratify`
- xử lý lệch lớp bằng `class_weight="balanced_subsample"`
- huấn luyện `RandomForestClassifier`
- lưu mô hình và metrics vào thư mục `models/`

### Bước 3. Gợi ý thực phẩm

Trong `app.py`, sau khi mô hình dự đoán nhãn chế độ ăn:

- hệ thống lọc dữ liệu thực phẩm theo nhóm dinh dưỡng và ngưỡng phù hợp với từng lớp dự đoán
- cho phép chọn ưu tiên:
  - `Tất cả`
  - `Món chay`
  - `Món không chay`
- sinh thực đơn gồm `Bữa sáng`, `Bữa trưa`, `Bữa tối`, `Bữa phụ`
- hiển thị ảnh minh họa, dưỡng chất theo bữa và tổng dưỡng chất trong ngày

## Giao diện ứng dụng

Ứng dụng gồm 3 trang chính:

### 1. Giới thiệu & Khám phá dữ liệu

- thông tin đề tài
- thông tin sinh viên
- xem trước dữ liệu
- biểu đồ phân bố nhãn
- biểu đồ phân bố BMI theo nhóm chế độ ăn
- biểu đồ tỷ lệ bệnh lý theo nhóm
- biểu đồ phân bố nhóm thực phẩm USDA

### 2. Triển khai mô hình

- nhập tuổi, chiều cao, cân nặng
- nhập tình trạng tiểu đường và cao huyết áp
- tính BMI tự động
- dự đoán nhãn chế độ ăn
- hiển thị xác suất dự đoán
- hiển thị thực đơn gợi ý theo từng bữa
- hiển thị ảnh minh họa món ăn
- biểu đồ donut tỷ lệ năng lượng theo bữa
- biểu đồ tổng dưỡng chất của thực đơn

### 3. Đánh giá & Hiệu năng

- `Accuracy`
- `F1-score weighted`
- `Confusion Matrix`
- `Feature Importance`
- `Classification Report`
- biểu đồ chất lượng theo từng lớp
- biểu đồ các cặp nhầm lẫn xuất hiện nhiều nhất

## Kết quả hiện tại

File `models/metrics.json`:

- `Accuracy`: **0.9698**
- `F1-score weighted`: **0.9696**
- `Train size`: **793**
- `Test size`: **199**

## Cách chạy dự án

### 1. Cài thư viện

```bash
pip install -r requirements.txt
```
### 2. Sinh lại dữ liệu bệnh nhân

```bash
python scripts/code_tusinhdata.py
```

### 3. Huấn luyện lại mô hình

```bash
python scripts/train_model.py
```

### 4. Chạy ứng dụng Streamlit

```bash
streamlit run app.py
```

Nếu lệnh `streamlit` không nhận, có thể dùng:

```bash
python -m streamlit run app.py
```

## Chạy nhanh trên Windows CMD

```bat
cd /d C:\Users\User\Downloads\DOANCUOIKY\DOANCUOIKY
python scripts\code_tusinhdata.py
python scripts\train_model.py
streamlit run app.py
```

## Điểm mạnh hiện tại

- Có pipeline ML đầy đủ từ dữ liệu -> huấn luyện -> đánh giá -> giao diện web
- Giao diện Streamlit đã được thiết kế lại theo hướng hiện đại, trực quan
- Có mô hình đóng gói sẵn trong `models/model.pkl`
- Có xử lý lệch lớp ở mức mô hình bằng `class_weight` và chia tập theo `stratify`
- Có thực đơn gợi ý theo từng bữa, biểu đồ trực quan và ảnh minh họa

## Hạn chế hiện tại

- Dữ liệu bệnh nhân là **synthetic data**, chưa phải dữ liệu lâm sàng thực tế
- Thực đơn hiện được sinh theo luật từ dữ liệu USDA, chưa phải thực đơn do chuyên gia dinh dưỡng xây dựng
- Tên thực phẩm trong USDA chủ yếu là tiếng Anh nên phần mô tả món vẫn mang tính kỹ thuật
- Ứng dụng hiện ưu tiên tính đơn giản và dễ trình bày trong đồ án

## Hướng phát triển

- Việt hóa sâu hơn tên thực phẩm hoặc ánh xạ sang tên món quen thuộc
- Tăng chất lượng dữ liệu đầu vào bằng dữ liệu thật hoặc dataset chuẩn hơn
- Mở rộng logic dinh dưỡng theo từng bệnh lý cụ thể
- Triển khai ứng dụng lên Streamlit Community Cloud hoặc Hugging Face Spaces

## Tác giả

- **Họ tên:** Lê Ngọc Ánh
- **MSSV:** 22T1020019
