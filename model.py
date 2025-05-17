import xgboost as xgb
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
def model_SVM(X_train, y_train, C=1.0, kernel='rbf', gamma='scale'):
    if kernel == 'linear':
        model = SVC(C=C, kernel='linear', class_weight='balanced', random_state=42)
    elif kernel == 'rbf':
        model = SVC(C=C, kernel='rbf', gamma=gamma, class_weight='balanced', random_state=42)
    
    model.fit(X_train, y_train)
    return model
def model_LinearSVC(X_train, y_train):
    svm = LinearSVC(class_weight='balanced', random_state=42, max_iter=10000)

    # Huấn luyện mô hình
    model = svm.fit(X_train, y_train)
    return model
# def xgb(X_train, y_train):

#     # Create the model with specified parameters
#     model = xgb.XGBClassifier(scale_pos_weight=5, use_label_encoder=False, eval_metric='mlogloss')
#     model.fit(X_train, y_train)
#     return model
def model_xgboost(X_train, y_train, NUM_CLASSES):
    # Tính trọng số lớp
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    
    # Tạo dictionary ánh xạ class -> weight
    weight_dict = {i: w for i, w in zip(classes, class_weights)}
    
    # Tạo trọng số cho từng mẫu
    sample_weights = np.array([weight_dict[label] for label in y_train])

    # Khởi tạo mô hình với cấu hình nhiều lớp
    model = XGBClassifier(objective='multi:softmax', num_class=NUM_CLASSES, eval_metric='mlogloss', use_label_encoder=False)
    
    # Huấn luyện mô hình với trọng số mẫu
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    return model

def model_logistic_regression(X_train, y_train):
    
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight='balanced', random_state=42, max_iter=10000)
    model.fit(X_train, y_train)
    return model
from sklearn.neighbors import KNeighborsClassifier

def model_knn(X_train, y_train, n_neighbors=5):

    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', algorithm='auto', n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def model_random_forest(X_train, y_train):

    model = RandomForestClassifier(n_estimators=100,class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    return model
#####
def get_random_forest_model(**kwargs):
    return RandomForestClassifier(**kwargs)

def get_xgboost_model(**kwargs):
    return XGBClassifier(**kwargs)

def get_logistic_regression_model(**kwargs):
    return LogisticRegression(**kwargs)

def get_knn_model(**kwargs):
    return KNeighborsClassifier(**kwargs)



class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, dropout_rate=0.3):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)  # Thêm Dropout
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)  # Thêm Dropout
        self.layer3 = nn.Linear(hidden_size2, hidden_size3)  # Thêm tầng ẩn thứ 3
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)  # Thêm Dropout
        self.output = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.dropout1(x)
        x = self.relu2(self.layer2(x))
        x = self.dropout2(x)
        x = self.relu3(self.layer3(x))
        x = self.dropout3(x)
        x = self.output(x)
        return x

def model_mlp(X_train, y_train, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Đang huấn luyện trên: {device}')

    # Chuyển dữ liệu sang Tensor
    X_train_tensor = torch.tensor(X_train.astype(np.float32).values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)

    # Tính trọng số cho CrossEntropyLoss
    class_counts = np.bincount(y_train)
    num_classes = len(class_counts)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * num_classes
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    # Khởi tạo mô hình
    model = MLP(input_size, hidden_size1, hidden_size2, hidden_size3, output_size).to(device)

    # Loss và optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Tăng lr
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    # Huấn luyện
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Điều chỉnh lr
        scheduler.step(loss)

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    return model
def predict_mlp(model, X):
    model.eval()  # Đặt mô hình sang chế độ đánh giá
    with torch.no_grad():
        device = next(model.parameters()).device  # Lấy thiết bị hiện tại của model
        X_tensor = torch.tensor(X.astype(np.float32).values, dtype=torch.float32).to(device)
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)  # Lấy chỉ số lớp có xác suất cao nhất
    return predicted.cpu().numpy()
def predict(model, X_test):
    predictions = model.predict(X_test)
    return predictions