# Estado da arte: Machine Learning
Principais algoritmos de Machine Learning. Uma explicação detalhada de cada um deles, como como funcionam, para quais problemas cada um deles é indicado.


### **O que é aprendizado de máquina**
Algoritmos de *machine learning* permitem que computadores "aprendam" padrões a partir de dados. E somente pela análise desses dados. O aprendizado de páquina é extremamente baseado em estatística, que é extremamente importante em diversas áreas da ciência, finanças e indústria. 

Através da análise de dados históricos, é possível prever o comportamentos e resolver problemas complexos, tais como: prever o preço de ações daqui uma semana, identificar letras e números manuscritos ou mesmo identificar pessoas com maior risco de desenvolver doenças.

Em um cenário típico, o algoritmo de aprendizado retorna uma saída quantitativa ou qualitativa que queremos prever baseado em diversas características. Temos também uma base de dados de treinamento, que possui as características de entrada e saídas correspondentes. É com base nesses dados que o algoritmo será treinado para aprender o problema e realizar as predições.


### **O que é um modelo**
Um modelo de *machine learning* é um algoritmo parametrizado (treinado) para reconhecer padrões de uma base de dados (dataset). Um modelo também pode ser treinado a partir da combinação de diferentes algoritmos de forma a conseguir um resultado mais preciso.

Por exemplo, podemos criar um modelo para reconhecer caracteres manuscritos. Para isso, passamos uma base de imagens de caracteres manuscritos e informamos qual caractere aquela imagem representa. Dessa forma o modelo será treinado para reconhecer o padrão para cada caractere e, ao receber uma nova imagem, irá utilizar o padrão aprendido para prever o caractere correspondente.

### **Modelos de classificação e regressão**
Se analizarmos diferentes exemplos de modelos de predição, podemos perceber que a saída desses modelos pode ser bem diferente. Um dos exemplos mais famosos é o modelo [flor de Iris (ou Iris de Fisher)](https://www.kaggle.com/arshid/iris-flower-dataset), em que o algoritmo determina a espécie das flores utilizando características físicas delas para isso. Nesse modelo, a saída o algoritmo é *qualitativa* (a espécie da Iris) e sempre será um dos valores do grupo *G = {setosa, virginca, versicolor}*. 

Nesse exemplo, não há uma categoria maior que outra, ou seja, elas não são ordenáveis. Porém, se analizarmos o [exemplo de predição de glicose](https://www.kaggle.com/houcembenmansour/predict-diabetes-based-on-diagnostic-measures), a saída é uma medida *quantitativa*, em que os valores podem ser ordenados.

Mesmo com essas diferenças, em ambos os exemplos, faz sentido utilizarmos as entradas para prever as saídas. No primeiro exemplo, a partir das características físicas das flores (entradas) é determinada a espécie (saída). No segundo exemplo, a pertir de dados como colesterol, pressão arterial, idade, etc., a taxa de glicose é determinada. 

Por convenção, os algoritmos de predição são agrupados de acordo com o tipo de saída: algoritmos de *regressão* têm saídas *quantitativas* enquanto algoritmos de *classificação* têm saídas *qualitativas*. 

Também é possível agrupar as variáveis de entrada (*inputs*) como quantitativas e qualitativas. Alguns algotimos tendem a ter melhores resultados em problemas com inputs quantitativos enquanto outros para inputs qualitativos.

Variáveis de entrada geralmente são representadas pelo símbolo ***X***. Se ***X*** é um vetor, os componentes são referenciados por ***X<sub>j</sub>***. Variáveis de saída quantitativas são representadas por ***Y*** e qualitativas por ***G***. Letras maiúsculas são utilizadas para denominar as variáveis de forma geral, enquanto as amostras em letras minúsculas (*x<sub>n<sub>* representa o n-ésimo valor de ***X***). Dado uma entrada X, podemos referenciar a saída calculada pelo modelo por <img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;\bg_white&space;\widehat{Y}"/> se for um valor quantitativo ou por <img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;\bg_white&space;\widehat{G}"/>  caso seja qualitativo.



# Algoritmos de aprendizado supervisionado

## Principais algoritmos
- [Modelos Lineares](linear-models)
- [K-Nearest Neighbors](k-nearest-neighbors)
- [Naive Bayes Classifiers](NaiveBayesClassifiers)
- [Decision Trees e Ensembles of Decisions Trees](Decision-Trees-e-Ensembles-of-Decisions-Trees)


# Algoritmos de aprendizado não supervisionado

### Principais algoritmos
- [K-Means Clustering](K-MeansClustering)
- [Agglomerative Clustering](AgglomerativeClustering)

# Deep Learning
### Principais algoritmos
- [Redes Neurais Profundas](Redes-Neurais-Profundas)
- [Processamento de Linguagem Natural](processamento-de-linguagem-natural)


### Este repositório é mantido por:

[João Pedro](https://www.linkedin.com/in/joaopedrogfmattos/)

[João Victor](https://www.linkedin.com/in/jvictor-antonucci/)

[Phelipe Ávila](https://www.linkedin.com/in/phelipeavila/)
