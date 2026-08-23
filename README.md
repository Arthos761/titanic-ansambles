# 🚢 Titanic Survival Prediction — Ensemble Methods

## Описание проекта
Проект по предсказанию выживаемости пассажиров Титаника с использованием ансамблевых методов машинного обучения.

Сравниваются два подхода:
- **VotingClassifier** — голосование нескольких моделей
- **StackingClassifier** — мета-модель поверх других моделей

---

## 📊 Результаты

| Метод | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Voting | 81.01% | 80.30% | 71.62% | 75.71% |
| Stacking | 81.01% | 84.48% | 66.22% | 74.24% |

### Выводы
- Обе модели показали приблизительно общую точность
- **Stacking** лучше по Precision (84.48% против 80.30%)
- **Voting** лучше по Recall и F1
- **VotingClassifier** — предпочтительный выбор для этой задачи

---

## 🧠 Используемые модели

В ансамбль входят:
- **XGBoost** — градиентный бустинг
- **RandomForest** — случайный лес
- **LogisticRegression** — логистическая регрессия
- **SVC** — метод опорных векторов (с калибровкой вероятностей)

---

## 🛠️ Технологии

- Python 3.x
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Seaborn
- Matplotlib

---

## 🚀 Запуск проекта

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Запустить код
python titanic_ensemble.py

---

👤 Автор
Abdulla — https://github.com/Arthos761