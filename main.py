import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np

class LinReg:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.coef_ = None
        self.intercept_ = np.random.random()

    def predict(self, X):
        y_pred = X @ self.coef_[1:] + self.coef_[0]
        return y_pred

    def derivative_w(self, X, y):
        y_pred = self.predict(X)
        return -2 * (X.T @ (y - y_pred)) / len(y)

    def derivative_w0(self, X, y):
        y_pred = self.predict(X)
        return -2 * (y - y_pred).mean()

    def grad(self, X, y):
        dw0 = self.derivative_w0(X, y)
        dw = self.derivative_w(X, y)
        return np.concatenate(([dw0], dw))

    def score(self, X, y):
        return ((y - self.predict(X)) ** 2).mean()

    def fit(self, X, y, epochs=30):
        self.coef_ = np.zeros(X.shape[1] + 1)
        self.coef_[0] = self.intercept_

        for epoch in range(1, epochs + 1):
            self.coef_ -= self.learning_rate * self.grad(X, y)

    def matplot(self, X_train, y_train, X_test, y_test):
        if X_train.shape[1] != 2:
            return 'Ошибка размерности данных! Никакого тебе 3D графика.'

        sns.set(style="whitegrid")
        fig = plt.figure(figsize=(15, 20))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X_train[:, 0], X_train[:, 1], y_train, color='red', label='Train')
        ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='blue', label='Test')

        x1_range = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 10)
        x2_range = np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 10)
        x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)

        y_pred_mesh = self.coef_[0] + self.coef_[1] * x1_mesh + self.coef_[2] * x2_mesh

        ax.plot_surface(x1_mesh, x2_mesh, y_pred_mesh, color='green', alpha=0.3)
        ax.view_init(elev=30, azim=30)

        ax.set(xlabel='Фича 1', ylabel='Фича 2', zlabel='y')
        ax.legend()
        return fig

    def plotly(self, X_train, y_train, X_test, y_test):
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'surface'}]])

        train_scatter = go.Scatter3d(
            x=X_train[:, 0], y=X_train[:, 1], z=y_train,
            mode='markers', marker=dict(color='red', size=5), name='Train'
        )

        test_scatter = go.Scatter3d(
            x=X_test[:, 0], y=X_test[:, 1], z=y_test,
            mode='markers', marker=dict(color='blue', size=5), name='Test'
        )

        x1_range = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 10)
        x2_range = np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 10)
        x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)

        y_pred_mesh = self.coef_[0] + self.coef_[1] * x1_mesh + self.coef_[2] * x2_mesh

        prediction_surface = go.Surface(
            x=x1_mesh, y=x2_mesh, z=y_pred_mesh,
            colorscale='Viridis', opacity=0.7, name='Prediction Plane'
        )

        fig.add_trace(train_scatter)
        fig.add_trace(test_scatter)
        fig.add_trace(prediction_surface)

        fig.update_layout(scene=dict(
            xaxis_title='Фича 1',
            yaxis_title='Фича 2',
            zaxis_title='y'
        ), width=1000, height=800)

        return fig


if 'flag' not in st.session_state:
    st.session_state.flag = False
if 'submit' not in st.session_state:
    st.session_state.submit = False

st.title('Корявая линейная регрессия')
st.caption('но я старался')
st.divider()

button = st.sidebar.button('Погонять на подготовленных данных')

if button:
    train = pd.read_csv('data/train_flats.csv')
    test = pd.read_csv('data/test_flats.csv')

    ss = StandardScaler()
    train_s = ss.fit_transform(train)
    test_s = ss.transform(test)

    st.session_state.y_train = train_s[:, -1]
    st.session_state.X_train = train_s[:, :-1]

    st.session_state.y_test = test_s[:, -1]
    st.session_state.X_test = test_s[:, :-1]

    st.session_state.flag = True

if st.session_state.flag:
    with st.sidebar.form(key='compression_form'):
        st.header('Настройки')
        lr = st.radio('Скорость обучения', [0.1, 0.01, 0.001])
        epoch = st.slider('Количество эпох', 5, 100)
        submit = st.form_submit_button(label="Обучаем")

        if submit:
            st.session_state.submit = True
            st.session_state.lr = lr
            st.session_state.epoch = epoch

if st.session_state.submit and 'results' not in st.session_state:
    ln = LinReg(learning_rate=st.session_state.lr)
    ln.fit(st.session_state.X_train, st.session_state.y_train, epochs=st.session_state.epoch)
    st.session_state.results = {
        'coefficients': ln.coef_,
        'score': ln.score(st.session_state.X_test, st.session_state.y_test),
        'plotly_fig': ln.plotly(st.session_state.X_train, st.session_state.y_train,
                                st.session_state.X_test, st.session_state.y_test),
        'matplot_fig': ln.matplot(st.session_state.X_train, st.session_state.y_train,
                                  st.session_state.X_test, st.session_state.y_test)
    }

if 'results' in st.session_state:
    st.markdown(f'**Коэффициенты:** W0: {st.session_state.results["coefficients"][0]:.4f}, '
                f'W1: {st.session_state.results["coefficients"][1]:.4f}, '
                f'W2: {st.session_state.results["coefficients"][2]:.4f}')
    st.markdown(f'**Score:** {st.session_state.results["score"]:.4f}')

    with st.form(key='plot_form'):
        cols = st.columns(2)
        with cols[0]:
            plot_choice = st.toggle("Интерактивный")
        with cols[1]:
            plot = st.form_submit_button(label="Дай график")

    if plot:
        if plot_choice:
            st.plotly_chart(st.session_state.results['plotly_fig'])
        else:
            st.pyplot(st.session_state.results['matplot_fig'])
