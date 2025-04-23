import xgboost as xgb
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

def SVM(X,y, C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0):

    model = svm.SVC(C=C, kernel=kernel, class_weight='balanced', random_state=42, degree=degree, gamma=gamma, coef0=coef0)
    model.fit(X, y)
    return model
def SVC(X_train, y_train):
    svm = LinearSVC(class_weight='balanced', random_state=42, max_iter=1000)

    # Huấn luyện mô hình
    model = svm.fit(X_train, y_train)
    return model
def xgb(X_train, y_train):

    # Create the model with specified parameters
    model = xgb.XGBClassifier(scale_pos_weight=5, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    return model
def random_forest(X_train, y_train):

    model = RandomForestClassifier(n_estimators=100,class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    return model
def predict(model, X_test):

    predictions = model.predict(X_test)
    return predictions