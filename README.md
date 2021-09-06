# Estado da arte: Machine Learning
Principais algoritmos de Machine Learning. Uma explicação detalhada de cada um deles, como como funcionam, para quais problemas cada um deles é indicado.


### **o que é aprendizado de máquina**
### **o que é um modelo**
### **modelos de classificação e regressão**
Se analizarmos diferentes exemplos de modelos de predição, podemos perceber que a saída desses modelos pode ser bem diferente. Um dos exemplos mais famosos é o modelo [flor de Iris (ou Iris de Fisher)](https://www.kaggle.com/arshid/iris-flower-dataset), em que o algoritmo determina a espécie das flores utilizando características físicas delas para isso. Nesse modelo, a saída o algoritmo é *qualitativa* (a espécie da Iris) e sempre será um dos valores do grupo *G = {setosa, virginca, versicolor}*. 

Nesse exemplo, não há uma categoria maior que outra, ou seja, elas não são ordenáveis. Porém, se analizarmos o [exemplo de predição de glicose](https://www.kaggle.com/houcembenmansour/predict-diabetes-based-on-diagnostic-measures), a saída é uma medida *quantitativa*, em que os valores podem ser ordenados.

Mesmo com essas diferenças, em ambos os exemplos, faz sentido utilizarmos as entradas para prever as saídas. No primeiro exemplo, a partir das características físicas das flores (entradas) é determinada a espécie (saída). No segundo exemplo, a pertir de dados como colesterol, pressão arterial, idade, etc., a taxa de glicose é determinada. 

Por convenção, os algoritmos de predição são agrupados de acordo com o tipo de saída: algoritmos de *regressão* têm saídas *quantitativas* enquanto algoritmos de *classificação* têm saídas *qualitativas*. 

Também é possível agrupar as variáveis de entrada (*inputs*) como quantitativas e qualitativas. Alguns algotimos tendem a ter melhores resultados em problemas com inputs quantitativos enquanto outros para inputs qualitativos.

Variáveis de entrada geralmente são representadas pelo símbolo ***X***. Se ***X*** é um vetor, os componentes são referenciados por ***X<sub>j</sub>***. Variáveis de saída quantitativas são representadas por ***Y*** e qualitativas por ***G***. Letras maiúsculas são utilizadas para denominar as variáveis de forma geral, enquanto as amostras em letras minúsculas (*x<sub>n<sub>* representa o n-ésimo valor de ***X***). Dado uma entrada X, podemos referenciar a saída calculada pelo modelo por <img src="https://render.githubusercontent.com/render/math?math=\hat{Y}"> se for um valor quantitativo ou por <img src="https://render.githubusercontent.com/render/math?math=\hat{G}"> caso seja qualitativo.



# Algoritmos de aprendizado supervisionado

## Principais algoritmos
- [Modelos Lineares](Linear-Models)
- [K-Nearest Neighbors](k-nearest-neighbors)
- [Naive Bayes Classifiers](NaiveBayesClassifiers)
- [Decision Trees e Ensembles of Decisions Trees](https://www.google.com/)
- [Support Vector Machines](https://www.google.com/)
- [Neural Networks](https://www.google.com/)

# Algoritmos de aprendizado não supervisionado

## Definição
## Principais algoritmos
- [PCA e AutoEncoder (Redução de Dimensionalidade)](https://adotg.github.io/knn-what-how-why/)
- [K-Means Clustering](https://www.google.com/)
- [Agglomerative Clustering](https://www.google.com/)

# Deep Learning
## Definição
## Principais algoritmos
- [Redes Neurais Profundas](https://www.google.com/)
- [Processamento de Linguagem Natural](https://www.google.com/)


## Este repositório é mantido por:

[João Pedro](https://medium.com/)

[João Victor](https://medium.com/)

[Phelipe Ávila](https://www.linkedin.com/in/phelipeavila/)

