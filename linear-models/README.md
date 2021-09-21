# Modelos Lineares
O modelos lineares são bastante utilizados na estatística e, por isso, também são muito importantes no aprendizado de máquina. O termo *Modelo Linear* é uma generalização que descreve uma série de modelos cuja saída é uma combinação linear de variáveis. Como exemplo dessa categoria, explicaremos o algoritmo de **Regressão Linear** nessa sessão.


## Conceito e Fundamentação
O Objetivo da Regressão Linear é definir uma reta (ou plano) que defina o padrão dos dados com a menor diferença possível entre o a saída calculada e o valor real. (Isso ficará mais claro no gráfico a seguir que ilustra esse padrão). 

<div>
<img src="https://s.dicionariofinanceiro.com/imagens/normdist-regression.jpg" width="500">
</div>

No caso do exemplo acima, temos uma Regressão Linear Simples, em que somente uma variável é analisada. A função que representa a saída é dada pela expressão a seguir:

<div>
<img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{150}&space;\bg_white&space;\[f(X) = w_{0} + w_{1} * x_{1}\]" > 
<div/>

Onde ***w<sub>0</sub>*** representa o ponto inicial da reta e ***w<sub>1</sub>*** representa a inclinação da reta. Esses são parâmetros que o algoritmo utiliza para definir a reta. Dessa forma,  ***x<sub>1</sub>*** representa o atributo de entrada que foi dada ao modelo. E com esses valores ele consegue fazer as previsões.

O exemplo abaixo ilustra o resultado de uma Regressão Linear Múltipla, em que mais de uma variável independente é usada para fazer a predição.

<img src="https://ichi.pro/assets/images/max/724/0*qq0yaecNRQiugnif.png" width="500">



## Classes de Problemas com melhores resultados
Esse tipo de algoritmo é aplicado quando há uma boa correlação linear (positiva ou negativa) entre os dados, ou seja, quando o relacionamento ou associação entre os dados pode ser definido com uma reta.

A correlação entre resultados é a medida estatística utilizada para calcular a associação entre os pontos.

Correlação Linear de Pearson: mede a correlação linear entre a nuvem de pontos. O resultado varia entre -1 e 1, sendo:

- -1 : Correlação linear perfeita negativa
-  1 : Correlação linear perfeita positiva
-  0 : Não tem correlação linear

<img src="https://bookdown.org/cienciadedadosnaep/ciencia_de_dados/Fig_correlacao.png" width="500">

## Os Resíduos

Observando os gráficos acima, é possível perceber que há uma diferença entre os os valores reais (***Y***) e os valores calculados pelo modelo calculado (<img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;\bg_white&space;\widehat{Y}"/> ) A diferença entre esses valores é chamada *resíduo* e representa o erro do modelo para cada valor calculado. Quanto maior o valor do resíduo, maior é o erro e, consequentemente, menor o ajuste do modelo. Geralmente, isso ocorre quando os dados de entrada possuem baixa correlação.

Algumas formas de validação dos modelos são:
- MAE (Erro médio absoluto): média dos resíduos de todos os pontos;
- MSE (Média dos erros ao quadrado): Média da soma do quadrado dos resíduos;
- QSR (Soma dos quadrados dos resíduos): Soma do quadrado de todos os resíduos para cada ponto;

## Exemplo de uma aplicação em Python

```python:
# Importações
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Dados de entrada
x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)

# Transformações
x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)

# Treinamento do modelo
model = LinearRegression().fit(x_, y)

# Resultados
r_sq = model.score(x_, y)
intercept, coefficients = model.intercept_, model.coef_

# Teste com novos dados
y_pred = model.predict(x_)

