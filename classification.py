from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

classifiers = []
predicts = []


# [TN, FP]
# [FN, TP]


def svm(x_train, y_train, x_test, y_test, f):
    # f.write("SVM\n")
    clf_svm = SVC(kernel='linear')
    clf_svm.fit(x_train, y_train)
    classifiers.append(clf_svm)
    pred1 = clf_svm.predict(x_test)
    predicts.append(pred1)
    f.write('A2', f'{accuracy_score(y_test, pred1)}')
    f.write('B2', f'{precision_score(y_test, pred1)}')
    f.write('C2', f'{recall_score(y_test, pred1)}')
    f.write('D2', f'{confusion_matrix(y_test, pred1)[0][0]}')
    f.write('E2', f'{confusion_matrix(y_test, pred1)[0][1]}')
    f.write('F2', f'{confusion_matrix(y_test, pred1)[1][0]}')
    f.write('G2', f'{confusion_matrix(y_test, pred1)[1][1]}')

    clf_svm = SVC(kernel='poly')
    clf_svm.fit(x_train, y_train)
    classifiers.append(clf_svm)
    pred2 = clf_svm.predict(x_test)
    predicts.append(pred2)
    # text = f'{accuracy_score(y_test, pred2)} {precision_score(y_test, pred2)} {recall_score(y_test, pred2)}\n'
    # f.write(text)
    # f.write(f'{confusion_matrix(y_test, pred2)}\n')
    f.write('A3', f'{accuracy_score(y_test, pred1)}')
    f.write('B3', f'{precision_score(y_test, pred1)}')
    f.write('C3', f'{recall_score(y_test, pred1)}')
    f.write('D3', f'{confusion_matrix(y_test, pred1)[0][0]}')
    f.write('E3', f'{confusion_matrix(y_test, pred1)[0][1]}')
    f.write('F3', f'{confusion_matrix(y_test, pred1)[1][0]}')
    f.write('G3', f'{confusion_matrix(y_test, pred1)[1][1]}')

    clf_svm = SVC(kernel='rbf')
    clf_svm.fit(x_train, y_train)
    classifiers.append(clf_svm)
    pred3 = clf_svm.predict(x_test)
    predicts.append(pred3)
    # text = f'{accuracy_score(y_test, pred3)} {precision_score(y_test, pred3)} {recall_score(y_test, pred3)}\n'
    # f.write(text)
    # f.write(f'{confusion_matrix(y_test, pred3)}\n')
    f.write('A4', f'{accuracy_score(y_test, pred1)}')
    f.write('B4', f'{precision_score(y_test, pred1)}')
    f.write('C4', f'{recall_score(y_test, pred1)}')
    f.write('D4', f'{confusion_matrix(y_test, pred1)[0][0]}')
    f.write('E4', f'{confusion_matrix(y_test, pred1)[0][1]}')
    f.write('F4', f'{confusion_matrix(y_test, pred1)[1][0]}')
    f.write('G4', f'{confusion_matrix(y_test, pred1)[1][1]}')

    clf_svm = SVC(kernel='sigmoid')
    clf_svm.fit(x_train, y_train)
    classifiers.append(clf_svm)
    pred4 = clf_svm.predict(x_test)
    predicts.append(pred4)
    # text = f'{accuracy_score(y_test, pred4)} {precision_score(y_test, pred4)} {recall_score(y_test, pred4)}\n'
    # f.write(text)
    # f.write(f'{confusion_matrix(y_test, pred4)}\n')
    f.write('A5', f'{accuracy_score(y_test, pred1)}')
    f.write('B5', f'{precision_score(y_test, pred1)}')
    f.write('C5', f'{recall_score(y_test, pred1)}')
    f.write('D5', f'{confusion_matrix(y_test, pred1)[0][0]}')
    f.write('E5', f'{confusion_matrix(y_test, pred1)[0][1]}')
    f.write('F5', f'{confusion_matrix(y_test, pred1)[1][0]}')
    f.write('G5', f'{confusion_matrix(y_test, pred1)[1][1]}')


def random_forest(x_train, y_train, x_test, y_test, f):
    f.write('A6', 'Random forest')
    for i in range(1, 100):
        clf = RandomForestClassifier(max_depth=i)
        clf.fit(x_train, y_train)
        classifiers.append(clf)
        pred = clf.predict(x_test)
        text = f'{len(classifiers)} {i} {accuracy_score(y_test, pred)} {precision_score(y_test, pred)} {recall_score(y_test, pred)}\n'
        # f.write(text)
        predicts.append(pred)
        # f.write(f'{confusion_matrix(y_test, pred)}\n')
        f.write(f'A{i + 6}', f'{accuracy_score(y_test, pred)}')
        f.write(f'B{i + 6}', f'{precision_score(y_test, pred)}')
        f.write(f'C{i + 6}', f'{recall_score(y_test, pred)}')
        f.write(f'D{i + 6}', f'{confusion_matrix(y_test, pred)[0][0]}')
        f.write(f'E{i + 6}', f'{confusion_matrix(y_test, pred)[0][1]}')
        f.write(f'F{i + 6}', f'{confusion_matrix(y_test, pred)[1][0]}')
        f.write(f'G{i + 6}', f'{confusion_matrix(y_test, pred)[1][1]}')
        f.write(f'H{i + 6}', i)

    clf = RandomForestClassifier(criterion='gini', max_depth=10)
    clf.fit(x_train, y_train)
    classifiers.append(clf)
    pred = clf.predict(x_test)
    predicts.append(pred)
    f.write('A106', f'{accuracy_score(y_test, pred)}')
    f.write('B106', f'{precision_score(y_test, pred)}')
    f.write('C106', f'{recall_score(y_test, pred)}')
    f.write('D106', f'{confusion_matrix(y_test, pred)[0][0]}')
    f.write('E106', f'{confusion_matrix(y_test, pred)[0][1]}')
    f.write('F106', f'{confusion_matrix(y_test, pred)[1][0]}')
    f.write('G106', f'{confusion_matrix(y_test, pred)[1][1]}')

    clf = RandomForestClassifier(criterion='entropy', max_depth=10)
    clf.fit(x_train, y_train)
    classifiers.append(clf)
    pred = clf.predict(x_test)
    predicts.append(pred)
    f.write('A107', f'{accuracy_score(y_test, pred)}')
    f.write('B107', f'{precision_score(y_test, pred)}')
    f.write('C107', f'{recall_score(y_test, pred)}')
    f.write('D107', f'{confusion_matrix(y_test, pred)[0][0]}')
    f.write('E107', f'{confusion_matrix(y_test, pred)[0][1]}')
    f.write('F107', f'{confusion_matrix(y_test, pred)[1][0]}')
    f.write('G107', f'{confusion_matrix(y_test, pred)[1][1]}')

    clf = RandomForestClassifier(criterion='log_loss', max_depth=10)
    clf.fit(x_train, y_train)
    classifiers.append(clf)
    pred = clf.predict(x_test)
    predicts.append(pred)
    f.write('A108', f'{accuracy_score(y_test, pred)}')
    f.write('B108', f'{precision_score(y_test, pred)}')
    f.write('C108', f'{recall_score(y_test, pred)}')
    f.write('D108', f'{confusion_matrix(y_test, pred)[0][0]}')
    f.write('E108', f'{confusion_matrix(y_test, pred)[0][1]}')
    f.write('F108', f'{confusion_matrix(y_test, pred)[1][0]}')
    f.write('G108', f'{confusion_matrix(y_test, pred)[1][1]}')


def knn(x_train, y_train, x_test, y_test, f):
    f.write("A109", "KNN")
    for i in range(1, 100):
        clf_KNN = KNeighborsClassifier(n_neighbors=i)
        clf_KNN.fit(x_train, y_train)
        classifiers.append(clf_KNN)
        pred = clf_KNN.predict(x_test)
        # text = f' {len(classifiers)} {i} {accuracy_score(y_test, pred)} {precision_score(y_test, pred)} {recall_score(y_test, pred)}\n'
        # f.write(text)
        predicts.append(pred)
        # f.write(f'{confusion_matrix(y_test, pred)}\n')
        f.write(f'A{i + 109}', f'{accuracy_score(y_test, pred)}')
        f.write(f'B{i + 109}', f'{precision_score(y_test, pred)}')
        f.write(f'C{i + 109}', f'{recall_score(y_test, pred)}')
        f.write(f'D{i + 109}', f'{confusion_matrix(y_test, pred)[0][0]}')  # TN
        f.write(f'E{i + 109}', f'{confusion_matrix(y_test, pred)[0][1]}')  # FP
        f.write(f'F{i + 109}', f'{confusion_matrix(y_test, pred)[1][0]}')  # FN
        f.write(f'G{i + 109}', f'{confusion_matrix(y_test, pred)[1][1]}')  # TP
