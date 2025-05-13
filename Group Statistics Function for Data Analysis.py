import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def group_statistics_analysis(df, features, group_by_col, output_dir='stats_output'):
    """
    Hàm thực hiện thống kê nhóm trên dataframe, phân tích các đặc trưng theo cột nhóm.
    
    Parameters:
    - df (pd.DataFrame): Dataframe chứa dữ liệu (ví dụ: RT-IoT2022).
    - features (list): Danh sách các cột đặc trưng để phân tích (ví dụ: ['fwd_pkts_tot', 'flow_duration']).
    - group_by_col (str): Tên cột để nhóm dữ liệu (ví dụ: 'Attack_type').
    - output_dir (str): Thư mục lưu kết quả (mặc định: 'stats_output').
    
    Returns:
    - None: In kết quả, vẽ biểu đồ, và lưu file CSV.
    """
    # 1. Kiểm tra dữ liệu đầu vào
    if not all(feat in df.columns for feat in features):
        raise ValueError("Một hoặc nhiều đặc trưng không tồn tại trong dataframe.")
    if group_by_col not in df.columns:
        raise ValueError(f"Cột {group_by_col} không tồn tại trong dataframe.")
    
    print("Kiểm tra giá trị khuyết:")
    print(df[features + [group_by_col]].isnull().sum())
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 2. Thống kê nhóm cho từng đặc trưng
    for feat in features:
        # print(f"\n=== Phân tích đặc trưng: {feat} ===")
        
        # Trung bình
        group_mean = df.groupby(group_by_col)[feat].mean()
        # print(f"\nTrung bình {feat} theo {group_by_col}:")
        # print(group_mean)
        
        # Trung vị
        group_median = df.groupby(group_by_col)[feat].median()
        # print(f"\nTrung vị {feat} theo {group_by_col}:")
        # print(group_median)
        
        # Thống kê chi tiết
        group_stats = df.groupby(group_by_col)[feat].describe()
        # print(f"\nThống kê chi tiết {feat} theo {group_by_col}:")
        # print(group_stats)
        
        # Lưu thống kê chi tiết
        # group_stats.to_csv(os.path.join(output_dir, f'{feat}_stats.csv'))
        # print(f"\nĐã lưu thống kê vào {output_dir}/{feat}_stats.csv")
        
        # 3. Trực quan hóa
        # Biểu đồ cột cho trung bình
        plt.figure(figsize=(10, 6))
        sns.barplot(x=group_mean.index, y=group_mean.values)

        # Cải thiện căn chỉnh nhãn trục X
        plt.xticks(rotation=45, ha='right')  # Thêm ha='right' để nhãn không đè lên nhau

        plt.title(f'Trung bình {feat} theo {group_by_col}')
        plt.xlabel(group_by_col)
        plt.ylabel(f'Trung bình {feat}')
        plt.yscale('log')  # Dùng log-scale để xử lý giá trị lớn

        plt.tight_layout()  # Tránh cắt nhãn
        plt.savefig(os.path.join(output_dir, f'{feat}_mean_barplot.png'))
        # plt.show()

        
        # Boxplot cho phân bố
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=group_by_col, y=feat, data=df)
        plt.title(f'Phân bố {feat} theo {group_by_col}')
        plt.xlabel(group_by_col)
        plt.ylabel(feat)
        plt.xticks(rotation=45)
        plt.yscale('log')  # Dùng log-scale để dễ quan sát
        # plt.savefig(os.path.join(output_dir, f'{feat}_boxplot.png'))
        # plt.show()
    
    # 4. Thống kê nhóm cho nhiều đặc trưng
    group_multi = df.groupby(group_by_col)[features].agg(['mean', 'median', 'std'])
    # print("\nThống kê nhiều đặc trưng theo {}:".format(group_by_col))
    # print(group_multi)
    
    # Lưu thống kê nhiều đặc trưng
    # group_multi.to_csv(os.path.join(output_dir, 'multi_features_stats.csv'))
    # print(f"\nĐã lưu thống kê nhiều đặc trưng vào {output_dir}/multi_features_stats.csv")
    
    # 5. Phân tích mẫu hình
    print("\n=== Phân tích mẫu hình ===")
    for feat in features:
        max_mean_group = group_mean.idxmax() if feat in group_mean else None
        min_mean_group = group_mean.idxmin() if feat in group_mean else None
        print(f"- {feat}: Nhóm {max_mean_group} có trung bình cao nhất, nhóm {min_mean_group} thấp nhất.")
        if 'fwd_pkts_tot' == feat:
            print(f"  => Mẫu hình: {feat} cao thường liên quan đến {max_mean_group} (tấn công mạnh).")
        if 'flow_duration' == feat:
            print(f"  => Mẫu hình: {feat} thấp có thể liên quan đến tấn công nhanh như DDoS.")
        if 'pkt_len_avg' == feat:
            print(f"  => Mẫu hình: {feat} cao có thể do gửi gói tin lớn trong tấn công.")

# Ví dụ sử dụng
if __name__ == "__main__":
    # Đọc dữ liệu RT-IoT2022
    try:
        data = pd.read_csv('data/RT_IOT2022')
        
        # Danh sách đặc trưng
#         features = [
#     'flow_duration',
#     'fwd_iat.min', 'fwd_iat.max', 'fwd_iat.tot', 'fwd_iat.avg', 'fwd_iat.std',
#     'bwd_iat.min', 'bwd_iat.max', 'bwd_iat.tot', 'bwd_iat.avg', 'bwd_iat.std',
#     'flow_iat.min', 'flow_iat.max', 'flow_iat.tot', 'flow_iat.avg', 'flow_iat.std',
#     'active.min', 'active.max', 'active.tot', 'active.avg', 'active.std',
#     'idle.min', 'idle.max', 'idle.tot', 'idle.avg', 'idle.std',
#     'fwd_pkts_per_sec', 'bwd_pkts_per_sec', 'flow_pkts_per_sec',
#     'payload_bytes_per_second'
# ]
        features = [
    "flow_duration",
    "flow_iat.min",
    "flow_iat.max",
    "flow_iat.avg",
    "flow_iat.tot",
    "active.min"
]


        # Gọi hàm
        group_statistics_analysis(
            df=data,
            features=features,
            group_by_col='Attack_type',
            output_dir='stats_output'
        )
    except FileNotFoundError:
        print("File RT_IoT2022.csv không tìm thấy. Vui lòng kiểm tra đường dẫn.")