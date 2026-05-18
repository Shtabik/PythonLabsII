import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
data = pd.DataFrame(
    {
        "text": [
            "Добавить темную тему в приложение",
            "Реализовать выгрузку отчетов в Excel",
            "Сделать быструю авторизацию через Telegram",
            "Добавить кнопку повтора последнего заказа",
            "Интеграция с календарем Google для планирования задач",
            "Сортировка товаров по возрастанию цены",
            "Добавить видео-инструкцию на главный экран",
            "Автоматические уведомления о скидках",
            "Поддержка английского языка в интерфейсе",
            "Раздел Избранное для частых покупок",
            "Добавить фильтр по датам в историю коммитов",
            "Кастомизация профиля под пользователя",
            "Добавить поиск по артикулу товара",
            "Возможность привязать несколько карт для оплаты",
            "Сделать онлайн-чат техподдержки в приложении",
            "Кнопка Поделиться для карточки товара",
            "Добавить отзывы с фото в карточку товара",
            "Внедрить систему кэшбэка за покупки",
            "Создать раздел с часто задаваемыми вопросами FAQ",
            "Добавить сканер штрих-кодов для поиска в магазине",
            "Приложение падает при попытке открыть корзину",
            "Кнопка оплаты не нажимается белый экран",
            "Не приходит смс-код для подтверждения профиля",
            "Шрифт накладывается друг на друга в мобильной версии",
            "Пропали сохраненные данные после обновления",
            "Ошибка 500 при попытке загрузить аватарку",
            "Текст сообщения обрезается невозможно прочитать",
            "Форма отправки зависает намертво на этапе проверки",
            "Не работает поиск server недоступен",
            "Купон на скидку выдает ошибку неверный формат данных",
            "При попытке авторизации всё зависает и бесконечно крутится лоадер",
            "Ошибка 404 при переходе в каталог товаров",
            "Вылетает приложение при смене пароля в настройках",
            "Не обновляется баланс в личном кабинете после оплаты",
            "Дублируются товары в корзине при повторном клике",
            "Звук уведомлений не отключается в настройках смартфона",
            "Ссылка на оплату ведет на пустую страницу",
            "Ошибка сессии при попытке войти в аккаунт",
            "Кнопка назад закрывает приложение вместо возврата на экран",
            "Пропадает текст в поле ввода комментария при отправке",
        ],
        "category": [1] * 20 + [0] * 20,
    }
)

X = data["text"]
y = data["category"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model_lr = Pipeline(
    steps=[
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
        (
            "clf",
            LogisticRegression(
                max_iter=1000, random_state=42, class_weight="balanced"
            ),
        ),
    ]
)


model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

print("ОТЧЕТ О КЛАССИФИКАЦИИ: ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ ")
print(classification_report(y_test, y_pred_lr, target_names=["Баг", "Фича"]))


model_mlp = Pipeline(
    steps=[
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
        (
            "clf",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.01,
                max_iter=500,
                random_state=42,
            ),
        ),
    ]
)


model_mlp.fit(X_train, y_train)
y_pred_mlp = model_mlp.predict(X_test)

print("\n ОТЧЕТ О КЛАССИФИКАЦИИ: НЕЙРОСЕТЬ")
print(classification_report(y_test, y_pred_mlp, target_names=["Баг", "Фича"]))

new_ticket = [
    "При попытке авторизации всё зависает и бесконечно крутится лоадер"
]
print("ТЕСТ НА НОВОМ ТИКЕТЕ ")
print(f'Текст обращения: "{new_ticket[0]}"\n')

pred_lr = model_lr.predict(new_ticket)[0]
prob_lr = model_lr.predict_proba(new_ticket)[0]
print("Результат Логистической регрессии")
print("Тип тикета:", "Запрос фичи" if pred_lr == 1 else "Баг-репорт")
print(f"Уверенность модели: {max(prob_lr):.2%}\n")


pred_mlp = model_mlp.predict(new_ticket)[0]
prob_mlp = model_mlp.predict_proba(new_ticket)[0]
print("Результат Нейросети MLP")
print("Тип тикета:", "Запрос фичи" if pred_mlp == 1 else "Баг-репорт")
print(f"Уверенность модели: {max(prob_mlp):.2%}")
