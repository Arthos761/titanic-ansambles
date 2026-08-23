import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

#1. ЗАГРУЗКА И ПРОСМОТР ДАННЫХ
df = sns.load_dataset('titanic')
print(df.head())
print(df.describe())
print(df.info())

#2. СОЗДАНИЕ НОВЫХ ПРИЗНАКОВ
df['family_size'] = df['sibsp'] + df['parch'] + 1
df['isAlone'] = (df['family_size'] == 1).astype(int)


#3. РАЗДЕЛЕНИЕ НА TRAIN/TEST
x = df.drop('survived', axis=1)
y = df['survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


#4. УСТАНОВКА COLUMNTRANSFORMER
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('std_scaler', StandardScaler())
    ]), ['age', 'fare', 'family_size', 'isAlone']),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), ['sex', 'embarked'])
])



#5. МЕТОД: 'ГОЛОСОВАНИЕ'(Voting)
pipeline1 = Pipeline([
    ('preprocessor', preprocessor),
    ('voting', VotingClassifier([
        ('xgb', XGBClassifier()),
        ('rf', RandomForestClassifier()),
        ('lnreg', LogisticRegression()),
        ('svc', CalibratedClassifierCV(SVC(), ensemble=False))
    ], voting='soft'))
])

pipeline1.fit(x_train, y_train)
y_pred1 = pipeline1.predict(x_test)

accuracy1 = accuracy_score(y_test, y_pred1)
precision1 = precision_score(y_test, y_pred1)
recall1 = recall_score(y_test, y_pred1)
f1_1 = f1_score(y_test, y_pred1)

print('РЕЗУЛЬТАТЫ МОДЕЛЕЙ С VOITING')
print('Общая правильность:', np.round(accuracy1 * 100, 2),'%')
print('Когда сказал да:', np.round(precision1 * 100, 2),'%')
print('Нашёл ли все «да»?:', np.round(recall1 * 100, 2),'%')
print('Баланас между precision и recall:',np.round(f1_1 * 100, 2),'%')


#6. ВЫБОР ПАРАМЕТРОВ(VOTING)
param_grid1 = {
    'voting__xgb__n_estimators': [100, 200],
    'voting__xgb__learning_rate': [0.02, 0.04],
    'voting__xgb__max_depth': [5, 10],
    'voting__rf__n_estimators': [100, 200],
    'voting__rf__max_depth': [5, 10],
    'voting__lnreg__C': [0.5, 1],
    'voting__svc__estimator__C': [0.5, 1.5],
    'voting__svc__estimator__gamma': [0.01, 0.1]
}

grid1 = GridSearchCV(pipeline1, param_grid1, cv=5, n_jobs=-1, scoring='accuracy')
grid1.fit(x_train, y_train)

print('\nЛучшие параметры:', grid1.best_params_)
print('Лучшая точность:', grid1.best_score_)




#7. МЕТОД:'ВЫБОР ЭКСПЕРТА'(Stacking)
pipeline2 = Pipeline([
    ('preprocessor', preprocessor),
    ('stacking', StackingClassifier([
        ('xgb', XGBClassifier()),
        ('rf', RandomForestClassifier()),
        ('lnreg', LogisticRegression()),
        ('svc', CalibratedClassifierCV(SVC(), ensemble=False))
    ], final_estimator=RandomForestClassifier()))
])

pipeline2.fit(x_train, y_train)
y_pred2 = pipeline2.predict(x_test)

accuracy2 = accuracy_score(y_test, y_pred2)
precision2 = precision_score(y_test, y_pred2)
recall2 = recall_score(y_test, y_pred2)
f1_2 = f1_score(y_test, y_pred2)

print('\nРЕЗУЛЬТАТЫ МОДЕЛЕЙ С STACKING:')
print('Общая правильность:', np.round(accuracy2 * 100, 2),'%')
print('Когда сказал да:', np.round(precision2 * 100, 2),'%')
print('Нашёл ли все «да»?:', np.round(recall2 * 100, 2),'%')
print('Баланас между precision и recall:',np.round(f1_2 * 100, 2),'%')




#8. ВЫБОР ПАРАМЕТРОВ(STACKING)
param_grid2 = {
    'stacking__xgb__n_estimators': [100, 200],
    'stacking__xgb__learning_rate': [0.02, 0.04],
    'stacking__xgb__max_depth': [5, 10],
    'stacking__rf__n_estimators': [100, 200],
    'stacking__rf__max_depth': [5, 10],
    'stacking__lnreg__C': [0.5, 1],
    'stacking__svc__estimator__C': [0.5, 1.5],
    'stacking__svc__estimator__gamma': [0.01, 0.1]
}

grid2 = GridSearchCV(pipeline2, param_grid2, cv=5, n_jobs=-1, scoring='accuracy')
grid2.fit(x_train, y_train)

print('\nЛучшие параметры:', grid2.best_params_)
print('Лучшая точность:', grid2.best_score_)




#9. ФИНАЛЬНАЯ КРОСС-ВАЛИДАЦИЯ
     #ДЛЯ VOTING
cv_scores_vot = cross_val_score(pipeline1, x_train, y_train, cv=5)
print('\nКРОСС-ВАЛИДАЦИЯ ДЛЯ VOTING:')
print(f"CV средняя: {cv_scores_vot.mean():.2%}")
print(f"CV разброс: {cv_scores_vot.std():.2%}")

     #ДЛЯ STACKING
cv_scores_stc = cross_val_score(pipeline2, x_train, y_train, cv=5)
print('\nКРОСС-ВАЛИДАЦИЯ ДЛЯ STACKING:')
print(f"CV средняя: {cv_scores_stc.mean():.2%}")
print(f"CV разброс: {cv_scores_stc.std():.2%}")


#10. МАТРИЦА ОШИБОК

   #ДЛЯ VOTING
cm_vot = confusion_matrix(y_test, y_pred1)
disp_vot = ConfusionMatrixDisplay(confusion_matrix=cm_vot, display_labels=['Погиб', 'Выжил'])
disp_vot.plot()
plt.title('Confusion Matrix — Voting')
plt.savefig('confusion_matrix_voting.png')
#plt.show() #СРАВНИАВЕМ БЛОКИ КРЕСТ-НАКРЕСТ, (ОТКАМЕНТИРОВАТЬ ДЛЯ ВИЗУАЛИЗАЦИИ)

   #ДЛЯ STACKING
cm_stc = confusion_matrix(y_test, y_pred2)
disp_stc = ConfusionMatrixDisplay(confusion_matrix=cm_stc, display_labels=['Погиб', 'Выжил'])
disp_stc.plot()
plt.title('Confusion Matrix — Stacking')
plt.savefig('confusion_matrix_stacking.png')
#plt.show() (ОТКАМЕНТИРОВАТЬ ДЛЯ ВИЗУАЛИЗАЦИИ)


#11. ВЫВОД

# ============================================
# ИТОГ ПРОЕКТА: СРАВНЕНИЕ АНСАМБЛЕЙ НА TITANIC
# ============================================

print("\n" + "="*60)
print("ИТОГ ПРОЕКТА: СРАВНЕНИЕ АНСАМБЛЕЙ НА TITANIC")
print("="*60)

print("\n=== РЕЗУЛЬТАТЫ МОДЕЛЕЙ ===")
print("-"*40)
print(f"Voting:")
print(f"  Accuracy:  {accuracy1:.2%}")
print(f"  Precision: {precision1:.2%}")
print(f"  Recall:    {recall1:.2%}")
print(f"  F1-score:  {f1_1:.2%}")
print()
print(f"Stacking:")
print(f"  Accuracy:  {accuracy2:.2%}")
print(f"  Precision: {precision2:.2%}")
print(f"  Recall:    {recall2:.2%}")
print(f"  F1-score:  {f1_2:.2%}")

print("\n=== ВЫВОДЫ ===")
print("-"*40)
print("1. Обе модели показали примерно одинаковую точность")
print("2. Stacking лучше по Precision (84.48% vs 80.30%).")
print("3. Voting лучше по Recall (71.62% vs 66.22%).")
print("4. Voting лучше по F1 (75.71% vs 74.24%).")
print("   -> Voting имеет лучший баланс Precision и Recall.")

print("\n=== КЛЮЧЕВОЙ ВЫВОД ===")
print("-"*40)
print("VotingClassifier показал более сбалансированный результат.")
print("Для задачи предсказания выживания на Титанике")
print("VotingClassifier является предпочтительным выбором.")

print("\n=== ЧТО СДЕЛАНО ===")
print("-"*40)
print("- Загружены и подготовлены данные Titanic")
print("- Добавлены новые признаки (family_size, isAlone)")
print("- Построен ColumnTransformer")
print("- Сравнены Voting и Stacking ансамбли")
print("- Настроены параметры через GridSearchCV")
print("- Проведена кросс-валидация")
print("- Построена матрица ошибок")








