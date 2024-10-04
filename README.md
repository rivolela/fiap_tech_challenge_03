# Tech Challenge 03 | FIAP 2024

## 1 - Objetivo

Este projeto faz parte de um desafio académico no FIAP College, onde vou desenvolver uma aplicação para investidores imobiliários na região do Porto. A aplicação irá prever os preços médios dos aluguéis para os próximos trimestres, utilizando dados de várias fontes oficiais portuguesas. Um modelo de aprendizado de máquina será treinado com os dados coletados, e as previsões serão entregues por meio de um aplicativo da Streamlit para uso dos investidores.

## 2 - Problema

Um grupo de investidores imobiliários precisa saber a previsão de preços de m2 de venda dos ativos (imóveis) para os trimesstre na região do Porto em Portugal.
Eles buscam maximizar o retorno das operações (compra/venda) sobre estes ativos (imóveis) por meio de um sistema preditivo.
A solução deve usar dados públicos e oficiais de instituições portuguesas e apresentar a média de previsão de venda do m2 por meio de uma linha temporal.
Caso não seja oneroso, criar filtros por região e tipo de imóvel.

## 3 - Notebook

Ainda em desenvolvimento:
https://colab.research.google.com/drive/1ddEYw6tDER6-j9I09H7P9UM9RFex6z4O?usp=sharing


## 4 - Install

```console
pip install -r requirements.txt
```

## 5 - Instanciar a Aplicação 

```console
streamlit run src/app.py
```

## 6 - Testes

```console
pytest
```