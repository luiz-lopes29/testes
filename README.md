# testes

ESTE TÉCNICO DATA SCIENCE

a - Como foi a definição da sua estratégia de modelagem?

Entender como cada uma das bases se conectava e de que maneira poderia gerar os preços da estadia que foi o problema escolhido.

b- Como foi definida a função de custo utilizada?

Com base nos período entre datas da listing, foi gerado o numero de dias. A partir daí foi criada variável chamada bookin30/60/90/365, com base availability, para que entre os dias fosse adotado % de ocupação, gerando os dias ocupados de cada estadia. 
Também foi considerado o custo unitário da tabela calendar e aplicado o preço do período. Após isso agrupado pel média a estadia de cada host id, chegando a variável resposta preço médio da estadia stay_price.

c. Qual foi o critério utilizado na seleção do modelo final?
Selecionei as variáveis explicativas com maior correlação para base do modelo. Após essa pré=seleção, foi utilizado feature selection, pelo método do F-test, onde foram selecionadas 10 melhores features.
Para o aplicação do modelo de regressão de machine learning foram escolhidos três métodos conhecidos para problema de regressão linear com dados supervisionados: Gradient Boosting, Linear Regression e Random forest. 


d. Qual foi o critério utilizado para validação do modelo?

Utilizar a separação da base de treino e teste e aplicar depois métricas de acurácia como as R2 e RMSE, realizar foi selecionado o modelo com menor erro, que foi o Gradient Boosting na base de treino. 
Cabe destacar, embora o resultado na base de teste, não tenha sido bom, é possível que ajustes futuros no balanceamento da base, bem como realizar o tunning de hiperparâmetros consiga-se alcançar resultados melhores.


Por que escolheu utilizar este método?

Gradient Boosting é um método que dá bom resultados para amostras pequenas, como a utilizada para o modelo.

e. Quais evidências você possui de que seu modelo é
suficientemente bom?

As evidências de ser um bom modelo é o resultado alcançado de 74% R2 de métrica de acurácia, sem a utilização de tunning de hiperparâmetros. 

