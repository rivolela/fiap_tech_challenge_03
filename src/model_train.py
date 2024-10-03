# model_train.py
import streamlit as st
import pandas as pd
import requests
from io import StringIO
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Função para baixar e limpar os dados
def load_and_clean_data(file_path):
    # Baixar o arquivo CSV
    response = requests.get(file_path, verify=True)
    if response.status_code == 200:
        df = pd.read_csv(StringIO(response.text), delimiter=',', quotechar='"', engine='python', on_bad_lines='skip')
        df.columns = df.columns.str.strip()  # Limpar os nomes das colunas
        return df
    else:
        print(f"Erro ao baixar o arquivo: {response.status_code}")
        return None

# Função para tratar dados nulos
def clean_nulls(df):
    df_clean = df.dropna()
    print("Rows removed due to null values:", df.shape[0] - df_clean.shape[0])
    return df_clean

# Função para tratar colunas numéricas
def convert_numeric_columns(df, colunas_numericas):
    for coluna in colunas_numericas:
        if coluna in df.columns:
            if df[coluna].dtype == 'object':
                df[coluna] = df[coluna].str.replace(',', '.').astype(float)
            else:
                df[coluna] = df[coluna].astype(float)
    return df


def tratar_outliers_iqr(df, colunas):
    for coluna in colunas:
        # Convertendo a coluna para float para evitar problemas de tipos
        df[coluna] = df[coluna].astype(float)
        
        Q1 = df[coluna].quantile(0.25)
        Q3 = df[coluna].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        # Aplicando os limites para remover outliers
        df.loc[df[coluna] < limite_inferior, coluna] = limite_inferior
        df.loc[df[coluna] > limite_superior, coluna] = limite_superior
        
        print(f"Coluna: {coluna}, Limite Superior Calculado: {limite_superior}")

    return df


# Função para normalização e padronização
def normalize_data(df, colunas_a_normalizar):
    scaler = StandardScaler()
    df.loc[:, colunas_a_normalizar] = scaler.fit_transform(df[colunas_a_normalizar])
    return df

# Função para codificação de dados categóricos
def encode_categorical_data(df):
    label_encoder = LabelEncoder()
    df['Periodo_encoded'] = label_encoder.fit_transform(df['Periodo'])

    if 'Bairro' in df.columns:
        df = pd.get_dummies(df, columns=['Bairro'], prefix='Bairro')
    return df

# Função para engenharia de features
def feature_engineering(df):
    df['media_valor_m2'] = (df['valor_m2_novo'] + df['valor_m2_existente']) / 2
    for lag in range(1, 8):
        df[f'media_valor_m2_lag_{lag}'] = df['media_valor_m2'].shift(lag)
    df.dropna(inplace=True)
    return df

# Função para treinar o modelo
def train_model(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor())
    ])
    
    param_grid = {
        'model__learning_rate': [0.07],
        'model__max_depth': [15],
        'model__min_samples_leaf': [3],
        'model__min_samples_split': [10],
        'model__n_estimators': [250]
    }

    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    return grid_search

# Função para fazer previsões
def make_predictions(grid_search, X_test):
    return grid_search.predict(X_test)


# Função para plotar os resultados
import matplotlib.pyplot as plt


# Função para plotar os resultados
def plot_results(real_values, predicted_values, future_predictions):
    # Cria uma nova figura
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plota os valores reais
    ax.plot(real_values, label='Valor Real', marker='o', linestyle='-', color='blue')
    # Plota as previsões
    ax.plot(predicted_values, label='Previsão', marker='x', linestyle='--', color='orange')
    # Plota a média dos valores reais
    ax.axhline(y=np.mean(real_values), color='r', linestyle='--', label='Média Real')
    # Plota as previsões futuras
    ax.plot(np.arange(len(real_values), len(real_values) + len(future_predictions)), future_predictions, label='Previsões Futuras', marker='s', linestyle='--', color='green')
    
    # Adiciona legendas e grid
    ax.legend()
    ax.grid()
    ax.set_title('Previsão de Valor Médio do M² - Testes e Previsões Futuras')
    ax.set_xlabel('Observações no DF de teste')
    ax.set_ylabel('Valor Médio do M²')
    
    # Exibir o gráfico no Streamlit
    st.pyplot(fig)  # Passa a figura criada para o Streamlit



