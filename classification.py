from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

classifiers = []
predicts = []


def svm(x_train, y_train, x_test, y_test, f):
    f.write("SVM\n")
    clf_svm = SVC(kernel='linear')
    clf_svm.fit(x_train, y_train)
    classifiers.append(clf_svm)
    pred1 = clf_svm.predict(x_test)
    predicts.append(pred1)
    text = f'{accuracy_score(y_test, pred1)} {precision_score(y_test, pred1)} {recall_score(y_test, pred1)}\n'
    f.write(text)
    f.write(f'{confusion_matrix(y_test, pred1)}\n')

    clf_svm = SVC(kernel='poly')
    clf_svm.fit(x_train, y_train)
    classifiers.append(clf_svm)
    pred2 = clf_svm.predict(x_test)
    predicts.append(pred2)
    text = f'{accuracy_score(y_test, pred2)} {precision_score(y_test, pred2)} {recall_score(y_test, pred2)}\n'
    f.write(text)
    f.write(f'{confusion_matrix(y_test, pred2)}\n')

    clf_svm = SVC(kernel='rbf')
    clf_svm.fit(x_train, y_train)
    classifiers.append(clf_svm)
    pred3 = clf_svm.predict(x_test)
    predicts.append(pred3)
    text = f'{accuracy_score(y_test, pred3)} {precision_score(y_test, pred3)} {recall_score(y_test, pred3)}\n'
    f.write(text)
    f.write(f'{confusion_matrix(y_test, pred3)}\n')

    clf_svm = SVC(kernel='sigmoid')
    clf_svm.fit(x_train, y_train)
    classifiers.append(clf_svm)
    pred4 = clf_svm.predict(x_test)
    predicts.append(pred4)
    text = f'{accuracy_score(y_test, pred4)} {precision_score(y_test, pred4)} {recall_score(y_test, pred4)}\n'
    f.write(text)
    f.write(f'{confusion_matrix(y_test, pred4)}\n')


def random_forest(x_train, y_train, x_test, y_test, f):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    f.write("Random Forest\n")
    clf_rf = RandomForestClassifier(criterion='gini')
    clf_rf.fit(x_train, y_train)
    classifiers.append(clf_rf)
    pred1 = clf_rf.predict(x_test)
    predicts.append(pred1)
    text = f'{accuracy_score(y_test, pred1)} {precision_score(y_test, pred1)} {recall_score(y_test, pred1)}\n'
    f.write(text)
    f.write(f'{confusion_matrix(y_test, pred1)}\n')

    clf_rf = RandomForestClassifier(criterion='entropy')
    clf_rf.fit(x_train, y_train)
    classifiers.append(clf_rf)
    pred2 = clf_rf.predict(x_test)
    predicts.append(pred2)
    text = f'{accuracy_score(y_test, pred2)} {precision_score(y_test, pred2)} {recall_score(y_test, pred2)}\n'
    f.write(text)
    f.write(f'{confusion_matrix(y_test, pred2)}\n')

    clf_rf = RandomForestClassifier(criterion='log_loss')
    clf_rf.fit(x_train, y_train)
    classifiers.append(clf_rf)
    pred3 = clf_rf.predict(x_test)
    predicts.append(pred3)
    text = f'{accuracy_score(y_test, pred3)} {precision_score(y_test, pred3)} {recall_score(y_test, pred3)}\n'
    f.write(text)
    f.write(f'{confusion_matrix(y_test, pred3)}\n')
    ConfusionMatrixDisplay.from_estimator(clf_rf, x_test, y_test)


def knn(x_train, y_train, x_test, y_test, f):
    f.write("KNN\n")
    for i in range(1, 100):
        clf_KNN = KNeighborsClassifier(n_neighbors=i)
        clf_KNN.fit(x_train, y_train)
        classifiers.append(clf_KNN)
        pred = clf_KNN.predict(x_test)
        text = f'{i} {accuracy_score(y_test, pred)} {precision_score(y_test, pred)} {recall_score(y_test, pred)}\n'
        f.write(text)
        predicts.append(pred)
        f.write(f'{confusion_matrix(y_test, pred)}\n')
