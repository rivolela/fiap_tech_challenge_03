import streamlit as st
from model_train import (
    load_and_clean_data, clean_nulls, convert_numeric_columns,
    tratar_outliers_iqr, normalize_data, encode_categorical_data,
    feature_engineering, train_model, make_predictions, plot_results
)
import numpy as np
from sklearn.metrics import mean_squared_error

def main():
    st.title("Análise de Previsão do Valor Médio do M²")

    # URL do CSV diretamente do GitHub
    file_path = "https://raw.githubusercontent.com/rivolela/fiap_tech_challenge_03/ae57bb6cd423c7b38a3208c9b6ea112d168d1344/csv/data_source_tech_challenge_03.csv"

    # Carregar e limpar os dados
    st.write("Carregando dados...")
    df = load_and_clean_data(file_path)
    if df is None:
        st.error("Erro ao carregar dados.")
        return

    df_clean = clean_nulls(df)

    # Definindo colunas numéricas
    colunas_numericas = [
        'valor_m2_novo', 'valor_m2_existente', 'taxa_inflacao_nacional',
        'taxa_juros_emprestimo_nacional', 'indice_preco_habitacao_alojamento_novo',
        'indice_preco_habitacao_alojamento_existente', 'taxa_desemprego_16_a_74_anos'
    ]

    df_clean = convert_numeric_columns(df_clean, colunas_numericas)
    df_tratado = tratar_outliers_iqr(df_clean, colunas_numericas)

    # Normalização
    colunas_a_normalizar = ['taxa_inflacao_nacional', 'taxa_juros_emprestimo_nacional', 'taxa_desemprego_16_a_74_anos']
    df_normalizado = normalize_data(df_tratado, colunas_a_normalizar)

    # Codificação de dados categóricos
    df_normalizado = encode_categorical_data(df_normalizado)

    # Engenharia de features
    df_normalizado = feature_engineering(df_normalizado)

    # Divisão dos dados
    train_size = int(len(df_normalizado) * 0.8)
    train, test = df_normalizado.iloc[:train_size], df_normalizado.iloc[train_size:]

    # Definindo variáveis dependentes e independentes
    X_train = train[['taxa_inflacao_nacional', 'taxa_juros_emprestimo_nacional',
                      'indice_preco_habitacao_alojamento_novo', 
                      'indice_preco_habitacao_alojamento_existente', 
                      'taxa_desemprego_16_a_74_anos'] +
                     [f'media_valor_m2_lag_{lag}' for lag in range(1, 8)]]
    y_train = train['media_valor_m2']

    X_test = test[['taxa_inflacao_nacional', 'taxa_juros_emprestimo_nacional',
                    'indice_preco_habitacao_alojamento_novo', 
                    'indice_preco_habitacao_alojamento_existente', 
                    'taxa_desemprego_16_a_74_anos'] +
                   [f'media_valor_m2_lag_{lag}' for lag in range(1, 8)]]
    y_test = test['media_valor_m2']

    # Treinar o modelo
    st.write("Treinando o modelo...")
    grid_search = train_model(X_train, y_train)

    # Fazer previsões
    st.write("Fazendo previsões...")
    y_pred = make_predictions(grid_search, X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write("Melhores Hiperparâmetros:", grid_search.best_params_)

    # Previsões Futuras
    n_futuros = 10
    future_predictions = []
    current_data = X_test.iloc[-1].to_numpy().reshape(1, -1)

    lag_columns = [f'media_valor_m2_lag_{i}' for i in range(1, 8)]
    for _ in range(n_futuros):
        future_pred = make_predictions(grid_search, current_data)
        future_predictions.append(future_pred[0])
        current_data = np.roll(current_data, -1)
        current_data[-1, -1] = future_pred

    # Plotar resultados
    st.write("Plotando resultados...")
    plot_results(y_test.values, y_pred, future_predictions)

if __name__ == "__main__":
    main()
