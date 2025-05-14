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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pca_analysis(df, n_components=2, plot=True):
    """
    Thực hiện PCA trên dataframe đầu vào.
    
    Tham số:
    - df: DataFrame chứa dữ liệu số
    - n_components: Số thành phần chính mong muốn (mặc định = 2)
    - plot: Nếu True, vẽ biểu đồ phương sai và scatter plot

    Trả về:
    - pca_df: DataFrame chứa các thành phần chính
    - explained_variance: Tỷ lệ phương sai của các thành phần chính
    """
    # Chỉ lấy các cột số
    features = df.select_dtypes(include=['float64', 'int64']).columns
    x = df[features].dropna().values  # loại bỏ NaN nếu có
    x_scaled = x
    
    # PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(x_scaled)
    
    # Tạo DataFrame kết quả
    pc_columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(data=principal_components, columns=pc_columns)

    # Hiển thị biểu đồ
    if plot:
        # Biểu đồ phương sai tích lũy
        pca_all = PCA().fit(x_scaled)
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(pca_all.explained_variance_ratio_)+1),
                 pca_all.explained_variance_ratio_.cumsum(), marker='o')
        plt.xlabel('Số thành phần chính')
        plt.ylabel('Tổng % phương sai')
        plt.title('Biểu đồ tích lũy phương sai PCA')
        plt.grid(True)
        plt.show()

        # Biểu đồ scatter plot nếu n_components >= 2
        if n_components >= 2:
            sns.scatterplot(data=pca_df, x='PC1', y='PC2')
            plt.title("Biểu đồ PCA (2 thành phần chính)")
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.show()

    return pca_df, pca.explained_variance_ratio_
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_pca_3d(X_pca, y=None, title="Biểu đồ PCA 3D"):
    """
    Vẽ biểu đồ PCA 3 thành phần chính trong không gian 3D.
    
    Parameters:
        X_pca: ndarray đã được PCA với n_components >= 3
        y: Nhãn (tùy chọn) để tô màu
        title: Tiêu đề biểu đồ
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    if y is not None:
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', edgecolor='k')
        plt.colorbar(sc)
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], edgecolor='k')

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.tight_layout()
    plt.show()

def plot_pca_4d(X_pca, y=None, title="Biểu đồ PCA 4D (3D + màu)"):
    """
    Vẽ PCA 4D: dùng PC1, PC2, PC3 cho không gian và PC4 cho màu sắc.

    Parameters:
        X_pca: ndarray đã được PCA với n_components >= 4
        y: Nhãn (tuỳ chọn), dùng nếu muốn tô màu theo lớp
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    color = X_pca[:, 3] if y is None else y
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                    c=color, cmap='plasma', edgecolor='k')
    plt.colorbar(sc)
    
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.tight_layout()
    plt.show()

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
