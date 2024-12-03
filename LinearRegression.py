import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle  # Sử dụng pickle thay cho joblib

# Đọc dữ liệu từ tệp CSV
data = pd.read_csv("ENB2012_data.csv")

# Các đặc trưng (X1 đến X8)
X = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]

# Biến mục tiêu (y1 và y2)
y1 = data['Y1']  # Heating Load
y2 = data['Y2']  # Cooling Load

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% huấn luyện, 20% kiểm tra)
X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.2, random_state=42)
_, _, y2_train, y2_test = train_test_split(X, y2, test_size=0.2, random_state=42)

# Huấn luyện mô hình cho Heating Load
heating_model = LinearRegression()
heating_model.fit(X_train, y1_train)

# Huấn luyện mô hình cho Cooling Load
cooling_model = LinearRegression()
cooling_model.fit(X_train, y2_train)

# Kiểm tra độ chính xác của mô hình trên tập huấn luyện
print("Training score for heating model: ", heating_model.score(X_train, y1_train))
print("Training score for cooling model: ", cooling_model.score(X_train, y2_train))

# Lưu mô hình đã huấn luyện vào tệp bằng pickle
try:
    with open("heating_model.pkl", "wb") as f:
        pickle.dump(heating_model, f)
    
    with open("cooling_model.pkl", "wb") as f:
        pickle.dump(cooling_model, f)
    
    print("Mô hình đã được lưu thành công!")
except Exception as e:
    print(f"Lỗi khi lưu mô hình: {e}")
