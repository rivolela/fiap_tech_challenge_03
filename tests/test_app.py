# tests/test_app.py
import pytest
import pandas as pd
from unittest.mock import patch
from src.model_train import load_and_clean_data, clean_nulls, convert_numeric_columns, feature_engineering

# Simulando o conteúdo de um CSV para os testes
@pytest.fixture
def mock_data():
    return "valor_m2_novo,valor_m2_existente\n1000,1500\n2000,2500\n3000,3500"

@pytest.fixture
def mock_requests_get(mocker, mock_data):
    """Mock da função requests.get para retornar dados simulados."""
    mock = mocker.patch('app.requests.get')
    mock.return_value.status_code = 200
    mock.return_value.text = mock_data
    return mock

def test_load_and_clean_data(mock_requests_get):
    file_path = "https://fakeurl.com/data.csv"
    df = load_and_clean_data(file_path)
    
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 3  # Deve conter 3 linhas de dados

def test_clean_nulls():
    df = pd.DataFrame({
        'a': [1, 2, None],
        'b': [4, None, 6]
    })
    df_clean = clean_nulls(df)
    assert df_clean.shape[0] == 1  # Deve remover duas linhas com nulos

def test_convert_numeric_columns():
    df = pd.DataFrame({'a': ['1,2', '3,4', '5,6']})
    df_converted = convert_numeric_columns(df, ['a'])
    assert df_converted['a'].dtype == float  # Deve ser convertido para float

def test_feature_engineering():
    df = pd.DataFrame({
        'valor_m2_novo': [1000, 2000, 3000],
        'valor_m2_existente': [1500, 2500, 3500]
    })
    
    df_engineered = feature_engineering(df)
    assert 'media_valor_m2' in df_engineered.columns  # Verifica se a coluna foi criada
    assert df_engineered['media_valor_m2'].isnull().sum() == 0  # Verifica se há valores nulos na nova coluna
