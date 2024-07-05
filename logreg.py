import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np


class LogReg:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.coef_ = None

    def predict_logit(self, X):
        return X @ self.coef_[1:] + self.coef_[0]

    def predict_proba(self, X):
        return 1 / (1 + np.exp(-self.predict_logit(X)))

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    def derivative_w(self, X, y):
        y_pred = self.predict_proba(X)
        return X.T @ (y_pred - y) / len(y)

    def derivative_w0(self, X, y):
        y_pred = self.predict_proba(X)
        return np.sum(y_pred - y) / len(y)

    def grad(self, X, y):
        dw0 = self.derivative_w0(X, y)
        dw = self.derivative_w(X, y)
        return np.concatenate(([dw0], dw))

    def score(self, X, y):
        y_pred = self.predict_proba(X)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def fit(self, X, y, epochs=500):
        self.coef_ = np.random.uniform(low=-0.1, high=0.1, size=X.shape[1] + 1)

        for epoch in range(1, epochs + 1):
            self.coef_ -= self.learning_rate * self.grad(X, y)

    def matplot(self, X_train, y_train, X_test, y_test):
        if X_train.shape[1] != 2:
            return 'Ошибка размерности данных!'

        fig = plt.figure(figsize=(20, 12))

        sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train, palette={0: 'blue', 1: 'red'},
                        style=y_train, markers={0: 'X', 1: 'X'}, legend='full', s=100, label='Train')

        sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_test, palette={0: 'blue', 1: 'red'},
                        style=y_test, markers={0: 'o', 1: 'o'}, legend='brief', s=100, label='Test')

        x_values = [np.min(np.vstack((X_train, X_test))[:, 0] - 1), np.max(np.vstack((X_train, X_test))[:, 0] + 1)]
        y_values = -(self.coef_[1] * np.array(x_values) + self.coef_[0]) / self.coef_[2]
        plt.plot(x_values, y_values, label='Линия регрессии')

        plt.xlabel('Фича 1')
        plt.ylabel('Фича 2')
        plt.legend()
        return fig

    def plotly(self, X_train, y_train, X_test, y_test):
        if X_train.shape[1] != 2:
            return 'Ошибка размерности данных!'

        fig = make_subplots(rows=1, cols=1)

        size = 7

        fig.add_trace(go.Scatter(
            x=X_train[y_train == 0][:, 0], y=X_train[y_train == 0][:, 1],
            mode='markers', marker=dict(color='blue', symbol='x', size=size), name='Train Class 0'
        ))
        fig.add_trace(go.Scatter(
            x=X_train[y_train == 1][:, 0], y=X_train[y_train == 1][:, 1],
            mode='markers', marker=dict(color='red', symbol='x', size=size), name='Train Class 1'
        ))

        fig.add_trace(go.Scatter(
            x=X_test[y_test == 0][:, 0], y=X_test[y_test == 0][:, 1],
            mode='markers', marker=dict(color='blue', symbol='circle', size=size), name='Test Class 0'
        ))
        fig.add_trace(go.Scatter(
            x=X_test[y_test == 1][:, 0], y=X_test[y_test == 1][:, 1],
            mode='markers', marker=dict(color='red', symbol='circle', size=size), name='Test Class 1'
        ))

        x_values = [np.min(np.vstack((X_train, X_test))[:, 0] - 1), np.max(np.vstack((X_train, X_test))[:, 0] + 1)]
        y_values = -(self.coef_[1] * np.array(x_values) + self.coef_[0]) / self.coef_[2]
        fig.add_trace(go.Scatter(
            x=x_values, y=y_values,
            mode='lines', name='Линия регрессии'
        ))

        fig.update_layout(
            xaxis_title='Фича 1',
            yaxis_title='Фича 2',
            width=1000,
            height=800
        )
        return fig


if 'flag' not in st.session_state:
    st.session_state.flag = False
if 'submit' not in st.session_state:
    st.session_state.submit = False

st.title('Логистическая регрессия')
st.caption('Серёжи')
st.divider()

button = st.sidebar.button('Погонять на подготовленных данных')

if button:
    train = pd.read_csv('data/credit_train.csv')
    test = pd.read_csv('data/credit_test.csv')

    ss = StandardScaler()
    X_train_s = ss.fit_transform(train.iloc[:, :-1])
    X_test_s = ss.transform(test.iloc[:, :-1])

    st.session_state.ss = ss
    st.session_state.y_train = train.iloc[:, -1].values
    st.session_state.X_train = X_train_s

    st.session_state.y_test = test.iloc[:, -1].values
    st.session_state.X_test = X_test_s

    st.session_state.flag = True

if st.session_state.flag:
    with st.sidebar.form(key='compression_form'):
        st.header('Настройки')
        lr = st.radio('Скорость обучения', [0.1, 0.01, 0.001])
        epoch = st.slider('Количество эпох', 5, 4000)
        submit = st.form_submit_button(label="Обучаем")

        if 'previous_results' not in st.session_state:
            st.session_state.previous_results = None

        if submit:
            st.session_state.submit = True
            st.session_state.lr = lr
            st.session_state.epoch = epoch
            if 'results' in st.session_state:
                st.session_state.previous_results = st.session_state.results
                del st.session_state.results

if st.session_state.submit:
    lg = LogReg(learning_rate=st.session_state.lr)
    st.session_state.lg = lg
    lg.fit(st.session_state.X_train, st.session_state.y_train, epochs=st.session_state.epoch)
    st.session_state.results = {
        'accuracy': lg.accuracy(st.session_state.X_test, st.session_state.y_test),
        'coefficients': lg.coef_,
        'score': lg.score(st.session_state.X_test, st.session_state.y_test),
        'plotly_fig': lg.plotly(st.session_state.X_train, st.session_state.y_train,
                                st.session_state.X_test, st.session_state.y_test),
        'matplot_fig': lg.matplot(st.session_state.X_train, st.session_state.y_train,
                                  st.session_state.X_test, st.session_state.y_test)
    }


def calculate_change(current, previous):
    if previous is None:
        return None
    return ((current - previous) / abs(previous)) * 100


if 'results' in st.session_state:
    st.subheader("Результаты обучения модели")

    st.write("### Коэффициенты:")
    st.write(f"**W0:** {st.session_state.results['coefficients'][0]:.4f}")
    st.write(f"**W1:** {st.session_state.results['coefficients'][1]:.4f}")
    st.write(f"**W2:** {st.session_state.results['coefficients'][2]:.4f}")

    st.write("### Оценка модели:")
    score_change = calculate_change(st.session_state.results['score'],
                                    st.session_state.previous_results[
                                        'score'] if st.session_state.previous_results else None)
    accuracy_change = calculate_change(st.session_state.results['accuracy'],
                                       st.session_state.previous_results[
                                           'accuracy'] if st.session_state.previous_results else None)
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Score", value=f"{st.session_state.results['score']:.4f}",
                  delta=f"{score_change:.2f}%" if score_change is not None else None)
    with col2:
        st.metric(label="Accuracy", value=f"{st.session_state.results['accuracy']:.4f}",
                  delta=f"{accuracy_change:.2f}%" if accuracy_change is not None else None)

    st.subheader('Узнай свои шансы на кредит!')
    with st.form(key='reg_form'):
        cols_1 = st.columns(3)
        with cols_1[0]:
            sq = st.slider(label='Кредитный рейтинг', min_value=0., max_value=10., step=0.01)
        with cols_1[1]:
            dist = st.number_input(label='Доход т.р.', min_value=10, max_value=200)
        with cols_1[2]:
            submit_1 = st.form_submit_button(label="Узнать!")
    if submit_1:
        user_input = st.session_state.ss.transform(np.array([[sq, dist]]))
        with cols_1[0]:
            if st.session_state.lg.predict(user_input)[0] == 1:
                st.write(f'Кредит одобрен')
            else:
                st.write(f'Никакого тебе кредита!')
        with cols_1[1]:
            st.write(f'Вероятность {st.session_state.lg.predict_proba(user_input)[0] * 100:.1f} %')

    with st.form(key='plot_form'):
        cols = st.columns(2)
        with cols[0]:
            plot_choice = st.checkbox("Интерактивный")
        with cols[1]:
            plot = st.form_submit_button(label="Дай график")

    if plot:
        if plot_choice:
            st.plotly_chart(st.session_state.results['plotly_fig'])
        else:
            st.pyplot(st.session_state.results['matplot_fig'])
