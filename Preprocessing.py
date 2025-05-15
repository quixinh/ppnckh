import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from imblearn.over_sampling import SMOTE

# Hàm load_data: Tải và xử lý cơ bản
def load_data(file_path, target_col='target'):
    data = pd.read_csv(file_path)
    
    # Tách X, y
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    return data  # <-- Phải là chỉ return data
def delete_columns(data, columns_to_delete):
    for col in columns_to_delete:
        if col in data.columns:
            data.drop(col, axis=1, inplace=True)
    return data
def check_types(data, categorical_cols):
    for col in data.columns:
        if data[col].dtype == categorical_cols:
            print(col)
def one_hot_encode(data, categorical_cols):
    for col in categorical_cols:
        dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
        data = pd.concat([data, dummies], axis=1)
        data.drop(col, axis=1, inplace=True)
    return data
def label_encode(data, categorical_cols):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    return data
# Hàm scale dữ liệu
def scale_data(X, scaler_type='standard'):
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns), scaler

# Hàm áp dụng SMOTE
def apply_smote(X, y, random_state=42):
    sampling_strategy = {label: 500 for label, count in y.value_counts().items() if count < 500}

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    return X_balanced, y_balanced

# Hàm chọn đặc trưng bằng SVM
def select_features(X, y, n_features=5):
    svm = SVC(kernel='linear')
    rfe = RFE(estimator=svm, n_features_to_select=n_features)
    rfe.fit(X, y)
    selected_features = X.columns[rfe.support_]
    return X[selected_features], selected_features

# Hàm vẽ heatmap
def plot_heatmap(X, y, title="Correlation Heatmap"):
    """
    Vẽ heatmap cho các đặc trưng đã chọn.
    Input: X, y
    """
    data = X.copy()
    data['Attack_type'] = y
    corr_matrix = data.corr()
    
    plt.figure(figsize=(20, 25))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(title)
    plt.show()

# Pipeline chính
def main():
    # Bước 1: Tải dữ liệu
    X, y = load_data('your_data.csv')
    
    # Bước 2: Scale dữ liệu
    X_scaled, scaler = scale_data(X, scaler_type='standard')
    
    # Bước 3: Áp dụng SMOTE
    X_balanced, y_balanced = apply_smote(X_scaled, y)
    
    # Bước 4: Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
    
    # Bước 5: Chọn đặc trưng
    X_train_selected, selected_features = select_features(X_train, y_train, n_features=5)
    X_test_selected = X_test[selected_features]
    
    # Bước 6: Vẽ heatmap
    plot_heatmap(X_train_selected, y_train, title="Correlation Heatmap of Selected Features")
    
    # Bước 7: Huấn luyện và đánh giá
    svm = SVC(kernel='linear')
    svm.fit(X_train_selected, y_train)
    accuracy = svm.score(X_test_selected, y_test)
    print(f"Accuracy on test set: {accuracy:.4f}")

if __name__ == "__main__":
    main()