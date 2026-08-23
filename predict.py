import joblib
import pandas as pd

# 1. Загрузка модели
model = joblib.load('titanic_voting_model.pkl')


# 2. Функция предсказания
def predict_survival(age, fare, family_size, isAlone, sex, embarked):
    data = pd.DataFrame({
        'age': [age],
        'fare': [fare],
        'family_size': [family_size],
        'isAlone': [isAlone],
        'sex': [sex],
        'embarked': [embarked]
    })
    pred = model.predict(data)[0]
    return 'Выжил' if pred == 1 else 'Погиб'

# 3. Примеры
print(predict_survival(25, 50, 2, 0, 'female', 'S'))  # Выжил
print(predict_survival(30, 10, 1, 1, 'male', 'S'))    # Погиб