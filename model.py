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

    model = XGBClassifier(objective='multi:softmax', num_class=NUM_CLASSES)
    model.fit(X_train, y_train)
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
def predict(model, X_test):
    predictions = model.predict(X_test)
    return predictions
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(hidden_size2, output_size)
        # Không cần Softmax vì CrossEntropyLoss đã bao gồm
    
    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.output(x)
        return x