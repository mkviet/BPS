import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Đọc dữ liệu từ file CSV
df = pd.read_csv("SaleData.csv")

# Hiển thị thông tin tổng quan về dữ liệu
print(df.info())

# Xem mẫu dữ liệu đầu tiên
print(df.head())

# Xử lý dữ liệu trống bằng cách điền giá trị trung bình
df.fillna(df.mean(), inplace=True)

# Lưu dữ liệu đã xử lý
df.to_csv("CleanedSaleData.csv", index=False)

# Biểu đồ cột của total_amount theo sale_date
df['sale_date'] = pd.to_datetime(df['sale_date'])
plt.figure(figsize=(12, 6))
plt.plot(df['sale_date'], df['total_amount'], marker='o', color='b')
plt.title('Biểu đồ doanh thu theo ngày')
plt.xlabel('Ngày')
plt.ylabel('Doanh thu')
plt.grid(True)
plt.show()

# Chọn features và target
X = df[['quantity', 'discount', 'tax_amount']]
y = df['total_amount']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo mô hình Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Đánh giá mô hình
print(f'R^2 Score: {model.score(X_test, y_test)}')

# Dự đoán doanh số bán hàng trong tương lai
future_data = [[5, 10, 15]]  # Số lượng, giảm giá, số tiền thuế
future_sales = model.predict(future_data)
print(f'Dự đoán doanh số: {future_sales[0]}')
