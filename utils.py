from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
def report(y_test, y_pred):
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#Vẽ Confusion Matrix
def confusion_matric(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix của Random Forest')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)  # Lưu hình cho báo cáo
    plt.show()
import pandas as pd
import matplotlib.pyplot as plt

def label_stats_plot(df, target_col):
    """
    Thống kê số lượng và tỷ lệ nhãn trong target + vẽ biểu đồ.

    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu
        target_col (str): Tên cột target

    Returns:
        pd.DataFrame: Bảng thống kê nhãn
    """
    value_counts = df[target_col].value_counts()
    percentages = df[target_col].value_counts(normalize=True) * 100

    stats_df = pd.DataFrame({
        'Count': value_counts,
        'Percentage (%)': percentages.round(2)
    })

    print(f"Tổng số mẫu: {len(df)}")
    print(stats_df)

    # Vẽ biểu đồ thanh
    plt.figure(figsize=(10, 4))
    value_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Phân phối nhãn trong "{target_col}"')
    plt.xlabel('Label')
    plt.ylabel('Số lượng')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return stats_df
import time

def measure_time(func, *args, **kwargs):
    """Đo thời gian thực thi của một hàm."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    print(f"Thời gian chạy: {end_time - start_time:.4f} giây")
    return result
