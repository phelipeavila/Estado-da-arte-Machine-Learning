# Processamento de Linguagem Natural


## Funcionamento

O processamento de linguagem natural incorpora técnicas diversas para interpretar a linguagem humana, desde métodos estatísticos e de machine learning a abordagens algorítmicas e baseadas em regras. É preciso de uma boa variedade de abordagens, porque dados baseados em texto ou voz divergem muito, assim como suas aplicações práticas.

## Conceito (O que é? Pra que serve? )

Processamento de linguagem natural (PLN) é uma vertente da inteligência artificial que ajuda computadores a entender, interpretar e manipular a linguagem humana. O PLN resulta de diversas disciplinas, incluindo ciência da computação e linguística computacional, que buscam preencher a lacuna entre a comunicação humana e o entendimento dos computadores.

O objetivo do PLN é fornecer aos computadores a capacidade de entender e compor textos. “Entender” um texto significa reconhecer o contexto, fazer análise sintática, semântica, léxica e morfológica, criar resumos, extrair informação, interpretar os sentidos, analisar sentimentos e até aprender conceitos com os textos processados.

Exemplo de aplicação:

* Nuvem de palavras;
* Traduções de textos, sites;
* Interpretação de sentimentos de postagens, textos;

## Classes de Problemas com melhores resultados

A linguagem humana é surpreendentemente complexa e diversa. Nós nos expressamos de infinitas maneiras, tanto verbalmente quanto por escrito. Não apenas existem centenas de idiomas e dialetos, como há também um conjunto único de regras gramaticais e de sintaxe, expressões e gírias dentro de cada um deles. Quando escrevemos, costumamos cometer erros ou abreviar palavras, ou omitimos pontuações; quando falamos, carregamos sotaques regionais, tendemos a murmurar e emprestamos termos de outros idiomas.

*   O PLN é importante porque ajuda a resolver a ambiguidade na linguagem e adiciona uma estrutura numérica útil aos dados para muitas aplicações downstream, como reconhecimento de fala ou análise de texto;

## Definição Teórica e Modelagem Matemática

Em termos gerais, as tarefas do PLN segmentam a linguagem em partes menores e essenciais, tenta entender as relações entre elas e explora como esses pedaços funcionam juntos para criar significado.

O funcionamento do computador de Alan Turing pode ser comparado a uma teoria matemática, facilmente aplicada à linguística, que surgiu também durante os anos 40, a Teoria da Informação. Ela tenta explicar o funcionamento do processo de comunicação/ transferência de sinais, que começa com a intenção e codificação do falante, ou código criptografado; em seguida, a mensagem é emitida/ recebida; ela passa pelo processo de decodificação (pela máquina ou cérebro do receptor); e, por fim, há a interpretação da mensagem captada, o que permite uma reação ou resposta do ouvinte.

Agora, se até para nós fica difícil de identificar e discernir ambiguidades, como uma máquina, desprovida de cognição, pode fazê-lo? Para os humanos, passamos pelo que é chamado de Teoria do Labirinto, onde a mente é o labirinto, e cabe a nós escolher o melhor caminho (processos mentais) para realizar tarefas. Ela é baseada nos princípios de Primazia da Sintaxe e Parser (análise sintática). O primeiro defende que mecanismos da linguagem que nós utilizamos são operações sintáticas básicas, como “merge” – isso mesmo, “merge” também é um termo computacional muito conhecido, que se trata de uma concatenação não aleatória. Parser, por sua vez, é capaz de resolver nosso problema, já que trata da ordem prioritária escolhida pelo cérebro no caso de ambiguidades, ou seja, qual caminho do labirinto deve ser tomado. Entre duas estruturas distintas, aquela com maior simplicidade sempre será escolhida, ou seja, a que possui um léxico que é acessado com maior frequência terá preferência. Programas como Siri, Alexa e Google funcionam da mesma maneira! Eles possuem dicionários, que podem ajudar no caso de ambiguidades lexicais, relacionando a palavra ambígua com outras palavras da frase e acionando o significado mais provável, algoritmos de árvores de decisão, que trabalham através de modelos estatísticos para escolher a alternativa mais plausível, e aprendizagem automática, que possibilita que o algoritmo melhore ao passo que te “conhece” melhor.

## Vantagens e Desvantagens (limitações)

    As vantagens desse algoritmo é o aprendizado constante e a grande quantidade de dados que podem ser interpretados de uma vez em pouco tempo, sem ter pausas.
    E uma das desvantagens é a falta de otimização da interpretação de formas mais complexas de comunicação, como gírias, ambiguidades, ironias etc.

## Exemplo de uma aplicação em Python

Aplicação em que dada uma avaliação de um filme, dizer se ela é positiva ou negativa:

```python:
# Remoção de acentos

import unidecode

def utf8_to_ascii(text):
    return unidecode.unidecode(text)

# Remoção de tags HTML (<div>, <p>, <h1>, <br>)

import re

def delete_html_nodes(text):
    regex = re.compile("<.+>")
    
    return re.sub(regex, "", text)

# Tokenização "I thought this was" -> ["I", "thought", "this", "was"]

import spacy

def tokenize(corpus, deacc=True, trim_html=True, header="review"):
    nlp = spacy.load("en_core_web_md")
    
    tokens = []
    for index, row in corpus.iterrows():
        document = row[header]
        # remove accents
        if deacc:
            document = utf8_to_ascii(document)
        
        # remove HTML tags and its content
        if trim_html:
            document = delete_html_nodes(document)
        
        spacy_doc = nlp(document)
        
        tokens.append([token for token in spacy_doc])
            
    return tokens

# Remoção de stop words (a, an, as, and, at, both, by, for, to)

def remove_stop_words(corpus):
    _tokens = []
    index = -1
    for document in corpus:
        _tokens.append([])
        index += 1
        
        for token in document:
            if not token.is_stop:
                _tokens[index].append(token)
            
    return _tokens

# Lematização

def lemmatize(corpus, remove_punct=True, remove_digits=True):
    lemmatized = []
    index = -1
    for document in corpus:
        lemmatized.append([])
        index += 1
        
        for token in document:
            # punctuation removal
            if remove_punct and token.is_punct:
                continue
                
            # digits removal
            if remove_digits and token.is_digit:
                continue

            lemmatized[index].append(token.lemma_)
            
        lemmatized[index] = " ".join(lemmatized[index])
        
        
    return lemmatized

tokens = tokenize(
        dataset,
        deacc=True,
        trim_html=True)

no_stop_words = remove_stop_words(tokens)

preprocessed_corpus = lemmatize(
        no_stop_words,
        remove_punct=True,
        remove_digits=True)

dataset.iloc[2, 0]

preprocessed_corpus[2]

labels = dataset.iloc[:, 1].map({"negative": 0, "positive": 1})

# Extração de Características

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(preprocessed_corpus)

features_dataset = tfidf_vectorizer.transform(preprocessed_corpus)

features_dataset.shape

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features_dataset, labels, shuffle=False, random_state=42)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)

# Criação do modelo

rom sklearn.svm import SVC

svm_model = SVC(probability=True)

svm_model.fit(X_train, y_train)

predictions = svm_model.predict(X_test)

predictions
```