from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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
    plt.figure(figsize=(8, 5))  # Tăng chiều rộng để có không gian
    ax = value_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Phân phối nhãn trong "{target_col}"')
    plt.xlabel('Label')
    plt.ylabel('Số lượng')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Căn chỉnh nhãn với cột và lệch sang trái
    ax.set_xticks(range(len(value_counts)))  # Đặt vị trí tick theo số lượng cột
    ax.set_xticklabels(value_counts.index, rotation=45, ha='right', rotation_mode='anchor')  # Căn phải và xoay
    plt.subplots_adjust(bottom=0.2)  # Tăng khoảng cách dưới để tránh chồng lấn
    
    # Dịch nhãn sang trái
    for label in ax.get_xticklabels():
        x_coord = label.get_position()[0]
        label.set_x(x_coord - 0.2)  # Dịch sang trái (tăng giá trị âm để lệch nhiều hơn)

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


def plot_confusion_matrix(y_true, y_pred, labels=None, tex="Ma trận nhầm lẫn sau giảm chiều - Mô hình Random Forest", figsize=(10, 8)):
    """
    Vẽ ma trận nhầm lẫn giữa nhãn thực tế và nhãn dự đoán.

    Args:
        y_true (array-like): Nhãn thực tế
        y_pred (array-like): Nhãn dự đoán
        labels (list, optional): Danh sách tên nhãn (nếu có)
        tex (str, optional): Tiêu đề biểu đồ và tên file
        figsize (tuple, optional): Kích thước biểu đồ (mặc định: (10, 8))
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Tính ma trận nhầm lẫn
    cm = confusion_matrix(y_true, y_pred)

    # Vẽ ma trận nhầm lẫn
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, square=True,
                xticklabels=labels if labels is not None else 'auto',
                yticklabels=labels if labels is not None else 'auto')

    plt.title(f'{tex}')
    plt.xlabel('Nhãn dự đoán')
    plt.ylabel('Nhãn thực tế')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Tạo thư mục nếu chưa có
    os.makedirs('result_confusion_matrix', exist_ok=True)

    # Đổi tên file hợp lệ (loại bỏ kí tự đặc biệt)
    from re import sub
    filename = sub(r'[\\/*?:"<>|]', "_", tex) + '.png'

    # Lưu hình
    plt.savefig(f'result_confusion_matrix/{filename}', dpi=300)
    plt.show()
