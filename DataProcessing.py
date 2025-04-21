import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris


def select_feature(X,y,top_n=20):
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
    plt.show()
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