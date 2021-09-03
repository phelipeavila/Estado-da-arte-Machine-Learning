# Naive Bayes Classifiers

Adivindo do Teorema de Bayes, o algoritmo Naive Bayes tem uma forte correlação com a área de estatística. Para que possamos ter uma compreensão completa desse algoritmo, vamos recapitular o teorema estatístico. Nele estamos tratando de probabilidade condicional. Vamos a uma situação: 
    
    * Se o tempo está frio, a chance de pescar um peixe grande é 20%. 
    * Se faz calor, a probabilidade de pescar um peixe grande é 50%. 
    * Sabemos que amanhã tem 30% de chance de fazer calor
    * Supondo que amanhã eu pegue um peixe grande, qual a probabilidade de ter feito sol?

Exemplo de aplicação:

* Filtro de Spam - identifica palavras chaves
* Mineração de emoções - Computação afetiva - identifica palavras chaves - Uma empresa mede o grau de satisfação com a empresa baseado nas palavras associadas a ela nas redes sociais.
* Clusterização de documento texto - jornais, materias, arquivos - identifica palavras nos documentos e os separa por tipo: politica, economia, esportes...


### Duas formas de Naive Bayes
*   Gaussian Naive Bayes
*   Multinomial Naive Bayes

### TODO
Falar sobre bias e pq da palavra naive
    Como a ordem de palavras nao altera os resultados e pode causar problemas
    Custo beneficio de ignorar essa ordem -> mesmo assim mostra bons resultados
Falar sobre Variance

### Funcionamento

Cria uma tabela de probabilidade de algo ocorrer, baseado nas informações que temos no nosso dataset. 
Tenho um dataset falando na inadiplencia de pessoas e tenho dados como salario, balanço patrimonial, bens pessoais para garantia e as informações sobre qual foi o risco associado previamente a cada pessoa.
Ao fazer uma tabela de probabilidade, iremos a probabilidade de uma pessoa que tem salario acima da média ter um risco alto, médio e baixo. E assim por diante para cada uma das informações que temos no nosso dataset. 

Nesse algoritmo o treinamento ou fase de aprendizagem é a parte do código que é retirado do dataset que temos a probabilidade de cada ocorrencia baseado em cada informação que temos. 



## Conceito (O que é? Pra que serve? )
## Classes de Problemas com melhores resultados
## Definição Teórica e Modelagem Matemática
## Vantagens e Desvantagens (limitações)
## Exemplo de uma aplicação em Python
