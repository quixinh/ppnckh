import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import model

def select_feature(X,y,top_n=40):
    # Train mô hình
    model = RandomForestClassifier()
    model.fit(X, y)


    feature_importance = model.feature_importances_
    feature_names = X.columns

    # Tạo DataFrame và sắp xếp
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    # Lấy Top N đặc trưng quan trọng nhất
    top_features = importance_df.head(top_n)

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_features, x='Importance', y='Feature', palette='viridis')
    plt.title(f'Top {top_n} Feature Importances (Random Forest)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    # plt.show()
    return top_features # return ve cai list co chua cac feature quan trong nhat
def remove_highly_correlated_columns(df, threshold=0.8, target_column=None):
    # Tính ma trận tương quan
    corr_matrix = df.corr().abs()  # Lấy giá trị tuyệt đối để xem xét cả tương quan âm
    
    # Tạo một tập để lưu các cột cần loại bỏ
    to_drop = set()
    
    # Duyệt qua ma trận tương quan
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] >= threshold:
                colname_i = corr_matrix.columns[i]
                colname_j = corr_matrix.columns[j]
                
                # Không loại bỏ cột mục tiêu
                if target_column and (colname_i == target_column or colname_j == target_column):
                    continue
                
                # Loại bỏ cột có tổng tương quan lớn hơn với các cột khác
                if colname_j not in to_drop:
                    to_drop.add(colname_j)
    
    # Loại bỏ các cột
    df_reduced = df.drop(columns=to_drop)
    dropped_columns = list(to_drop)
    
    print(f"Các cột bị loại bỏ: {dropped_columns}")
    
    return df_reduced, dropped_columns # return ve cai 
# df_reduced, dropped_columns = remove_highly_correlated_columns(data, threshold=0.8, target_column='attack_type')
    
# print("\nDataFrame sau khi loại bỏ:")
# print(df_reduced)
# print("\nDanh sách các cột bị loại bỏ:")
# print(dropped_columns)
import pandas as pd
def remove_highly_correlated_columns(df, threshold=0.8, target_column=None):
    # Tính ma trận tương quan
    corr_matrix = df.corr().abs()  # Lấy giá trị tuyệt đối để xem xét cả tương quan âm
    
    # Tạo một tập để lưu các cột cần loại bỏ
    to_drop = set()
    
    # Duyệt qua ma trận tương quan
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] >= threshold:
                colname_i = corr_matrix.columns[i]
                colname_j = corr_matrix.columns[j]
                
                # Không loại bỏ cột mục tiêu
                if target_column and (colname_i == target_column or colname_j == target_column):
                    continue
                
                # Loại bỏ cột có tổng tương quan lớn hơn với các cột khác
                if colname_j not in to_drop:
                    to_drop.add(colname_j)
    
    # Loại bỏ các cột
    df_reduced = df.drop(columns=to_drop)
    dropped_columns = list(to_drop)
    
    print(f"Các cột bị loại bỏ: {dropped_columns}")
    
    return df_reduced, dropped_columns
# df_reduced, dropped_columns = remove_highly_correlated_columns(data, threshold=0.8, target_column='attack_type')
    
# print("\nDataFrame sau khi loại bỏ:")
# print(df_reduced)
# print("\nDanh sách các cột bị loại bỏ:")
# print(dropped_columns)
import shap
import matplotlib.pyplot as plt  # Đảm bảo có thư viện vẽ

def shap_ex(X_train, y_train, X_test):
    # Giả sử bạn đã định nghĩa hàm random_forest trong model.py
    modelr = model.model_random_forest(X_train, y_train)
    
    # Train model
    modelr.fit(X_train, y_train)

    # Đảm bảo dữ liệu đầu vào là float
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    # Khởi tạo SHAP explainer
    explainer = shap.Explainer(modelr, X_train)

    # Tính SHAP values và tắt kiểm tra additivity để tránh lỗi
    shap_values = explainer(X_test, check_additivity=False)

    # Vẽ biểu đồ SHAP
    shap.summary_plot(shap_values, X_test)

#cachs kahc khac de ve shap
# from lime import lime_tabular

# explainer = lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=['Normal', 'Attack'], mode='classification')
# exp = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba)
# exp.show_in_notebook()
#pp khch nhau
# target = 'payload_bytes_per_second'  # hoặc 'flow_duration'
# X = df.drop(columns=[target, 'Attack_type'])
# y = df[target]

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# reg = RandomForestRegressor()
# reg.fit(X_train, y_train)

# y_pred = reg.predict(X_test)
# print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
# print("R² Score:", r2_score(y_test, y_pred))

# import matplotlib.pyplot as plt

# plt.scatter(y_test, y_pred, alpha=0.3)
# plt.xlabel("True Values")
# plt.ylabel("Predicted Values")
# plt.title("Regression Results")
# plt.show()
