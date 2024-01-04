# Indicador de performance do estudante para provas de matemática
 - Projeto de ponta a ponta,desde a leitura dos dados até o ambiente de produção
 - Objetivo do projeto é criar um ambiente simulando um ambiente profissional, finalizando com o modelo em produção
 - O projeto possui logs, mensagens de erro que dizem em qual script e em qual linha o erro apareceu, arquivo de setup e requisitos
 - Os notebooks contendo a EDA e o treinamento dos modelos foram pegos do github de referência para este projeto, pois o foco é na criação do ambiente e desenvolvimento da aplicação para produção
 - Pipeline para preprocessamento dos dados realizado, com a criação do objeto de preprocessamento me formato .pkl para ser usado no pipeline de predição
 - Código para treino dos modelos foi modularizado e colocado em formato de pipeline, incluindo logs e erros 
 - Pipelines para treino e seleção do melhor modelo encontrado, incluindo o ajuste de hiperparâmetros utilizando o GridSearchCV
 - Pipeline para predição do modelo, a partir do melhor modelo escolhido e salvo em .pkl
 - Produção utilizando Flask junto com arquivos html para um front-end servindo para captura dos dados do usuário

# Recursos utilizados
 - **Python**: 3.11
 - **Bibliotecas**: pandas, numpy, seaborn, matplotlib, sklearn, catboost, xgboost, dill, Flask
 - **Para utilizar o arquivo de requisitos**: `pip install -r requirements.txt`
 - **Guia do projeto no youtube**: https://www.youtube.com/watch?v=1m3CPP-93RI

# Aplicação Flask para colocar o modelo em produção
A aplicação Flask foi construida utilizando arquivos html para servir de front-end e realizar a captura dos dados do usuário. Com isso, tem-se o botão para realizar a predição e o resultadoaparece na tela. O modelo utilizado é o modelo encontrado durante o treinamento e que obteve os melhores resultados e foi salvo no formato .pkl. A aplicação final pode ser visualizada nas imagens abaixo.
![imagem16](https://github.com/vitorccmanso/mlproject/assets/129124026/481afc01-0e51-433a-a124-1740577693bc)
![imagem17](https://github.com/vitorccmanso/mlproject/assets/129124026/6897cb10-12e4-4689-89cd-8be142159f62)
