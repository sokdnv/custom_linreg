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

    def fit(self, X, y, epochs=500):
        self.coef_ = np.random.uniform(low=-1, high=1, size=X.shape[1] + 1)

        for epoch in range(1, epochs + 1):
            self.coef_ -= self.learning_rate * self.grad(X, y)
        formatted_weights = ' '.join([f'{coef:.4f}' for coef in self.coef_])

        print(f'Веса модели {formatted_weights}. Потеря {self.score(X, y)}')

    def matplot(self, X_train, y_train, X_test, y_test):
        if X_train.shape[1] != 2:
            return 'Ошибка размерности данных!'

        X = np.vstack((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        fig = plt.figure(figsize=(20, 12))
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette={0: 'blue', 1: 'red'}, legend='full')

        x_values = [np.min(X[:, 0] - 1), np.max(X[:, 0] + 1)]
        y_values = -(self.coef_[1] * np.array(x_values) + self.coef_[0]) / self.coef_[2]
        plt.plot(x_values, y_values, label='Прямая лог. регрессии')

        plt.xlabel('Фича 1')
        plt.ylabel('Фича 2')
        plt.legend()
        return fig

    def plotly(self, X_train, y_train, X_test, y_test):
        if X_train.shape[1] != 2:
            return 'Ошибка размерности данных!'

        X = np.vstack((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        fig = make_subplots(rows=1, cols=1)

        fig.add_trace(go.Scatter(
            x=X[y == 0][:, 0], y=X[y == 0][:, 1],
            mode='markers', marker=dict(color='blue'), name='Class 0'
        ))

        fig.add_trace(go.Scatter(
            x=X[y == 1][:, 0], y=X[y == 1][:, 1],
            mode='markers', marker=dict(color='red'), name='Class 1'
        ))

        x_values = [np.min(X[:, 0] - 1), np.max(X[:, 0] + 1)]
        y_values = -(self.coef_[1] * np.array(x_values) + self.coef_[0]) / self.coef_[2]
        fig.add_trace(go.Scatter(
            x=x_values, y=y_values,
            mode='lines', name='Прямая лог. регрессии'
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

        if submit:
            st.session_state.submit = True
            st.session_state.lr = lr
            st.session_state.epoch = epoch

if st.session_state.submit and 'results' not in st.session_state:
    lg = LogReg(learning_rate=st.session_state.lr)
    st.session_state.lg = lg
    lg.fit(st.session_state.X_train, st.session_state.y_train, epochs=st.session_state.epoch)
    st.session_state.results = {
        'coefficients': lg.coef_,
        'score': lg.score(st.session_state.X_test, st.session_state.y_test),
        'plotly_fig': lg.plotly(st.session_state.X_train, st.session_state.y_train,
                                st.session_state.X_test, st.session_state.y_test),
        'matplot_fig': lg.matplot(st.session_state.X_train, st.session_state.y_train,
                                  st.session_state.X_test, st.session_state.y_test)
    }

if 'results' in st.session_state:
    st.markdown(f'**Коэффициенты:** W0: {st.session_state.results["coefficients"][0]:.4f}, '
                f'W1: {st.session_state.results["coefficients"][1]:.4f}, '
                f'W2: {st.session_state.results["coefficients"][2]:.4f}')
    st.markdown(f'**Score:** {st.session_state.results["score"]:.4f}')

    st.subheader('Узнай свои шансы на кредит!')
    with st.form(key='reg_form'):
        cols_1 = st.columns(3)
        with cols_1[0]:
            sq = st.slider(label='Кредитный рейтинг', min_value=0, max_value=10)
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
            st.write(f'Вероятность одобрения {st.session_state.lg.predict_proba(user_input)[0]:.4f}')


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
