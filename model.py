import xgboost as xgb
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def SVM(X_train, y_train, C=1.0, kernel='rbf', gamma='scale'):
    if kernel == 'linear':
        model = SVC(C=C, kernel='linear', class_weight='balanced', random_state=42)
    elif kernel == 'rbf':
        model = SVC(C=C, kernel='rbf', gamma=gamma, class_weight='balanced', random_state=42)
    
    model.fit(X_train, y_train)
    return model
def LinearSVC(X_train, y_train):
    svm = LinearSVC(class_weight='balanced', random_state=42, max_iter=10000)

    # Huấn luyện mô hình
    model = svm.fit(X_train, y_train)
    return model
def xgb(X_train, y_train):

    # Create the model with specified parameters
    model = xgb.XGBClassifier(scale_pos_weight=5, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    return model
def xgboost(X_train, y_train, NUM_CLASSES):

    model = XGBClassifier(objective='multi:softmax', num_class=NUM_CLASSES)
    model.fit(X_train, y_train)
    return model

def logistic_regression(X_train, y_train):
    
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight='balanced', random_state=42, max_iter=10000)
    model.fit(X_train, y_train)
    return model
from sklearn.neighbors import KNeighborsClassifier

def knn(X_train, y_train, n_neighbors=5):

    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', algorithm='auto', n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def random_forest(X_train, y_train):

    model = RandomForestClassifier(n_estimators=100,class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    return model
def predict(model, X_test):
    predictions = model.predict(X_test)
    return predictions