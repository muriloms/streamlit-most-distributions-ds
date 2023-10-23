

import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def plot_distribution(dist_name, params):
    fig, ax = plt.subplots()
    x = np.linspace(-10, 10, 1000)
    if dist_name == "Normal Distribution":
        mean, std = params
        y = stats.norm.pdf(x, mean, std)
    elif dist_name == "Bernoulli Distribution":
        p = params[0]
        y = stats.bernoulli.pmf(x, p)
    elif dist_name == "Binomial Distribution":
        n, p = params
        x = np.arange(0, n+1)
        y = stats.binom.pmf(x, n, p)
    elif dist_name == "Poisson Distribution":
        lam = params[0]
        x = np.arange(0, 20)
        y = stats.poisson.pmf(x, lam)
    elif dist_name == "Exponential Distribution":
        lam = params[0]
        y = stats.expon.pdf(x, scale=1/lam)
    elif dist_name == "Gamma Distribution":
        a, b = params
        y = stats.gamma.pdf(x, a, scale=1/b)
    elif dist_name == "Beta Distribution":
        a, b = params
        x = np.linspace(0, 1, 1000)
        y = stats.beta.pdf(x, a, b)
    elif dist_name == "Uniform Distribution":
        a, b = params
        y = stats.uniform.pdf(x, a, b-a)
    elif dist_name == "Log-Normal Distribution":
        mean, std = params
        y = stats.lognorm.pdf(x, std, scale=np.exp(mean))
    elif dist_name == "Student t-distribution":
        df = params[0]
        y = stats.t.pdf(x, df)


    plt.plot(x, y)
    plt.title(dist_name)
    plt.grid(True)
    st.pyplot(fig)

    if dist_name == "Normal Distribution":
        st.latex(r'f(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}')
        st.write("""
        A distribuição mais amplamente utilizada em ciência de dados.\n
        Caracterizada por uma curva simétrica em forma de sino.\n
        É parametrizada por dois parâmetros—média e desvio padrão.\n
        Exemplo: Altura de indivíduos.
        """)
    elif dist_name == "Bernoulli Distribution":
        st.latex(r'P(X=k) = p^k (1-p)^{1-k} \quad \text{for } k \in \{0,1\}')
        st.write("""
        Uma distribuição de probabilidade discreta que modela o resultado de um evento binário.\n
        É parametrizada por um parâmetro—a probabilidade de sucesso.\n
        Exemplo: Modelagem do resultado de um lançamento de moeda.\n
        """)
    elif dist_name == "Binomial Distribution":
        st.latex(r'P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}')
        st.write("""
        É a distribuição de Bernoulli repetida várias vezes.\n
        Uma distribuição de probabilidade discreta que representa o número de sucessos em um número fixo de tentativas independentes de Bernoulli.\n
        É parametrizada por dois parâmetros—o número de tentativas e a probabilidade de sucesso.\n
        """)
    elif dist_name == "Poisson Distribution":
        st.latex(r'P(X=k) = \frac{e^{-\lambda} \lambda^k}{k!}')
        st.write("""
        Uma distribuição de probabilidade discreta que modela o número de eventos que ocorrem em um intervalo fixo de tempo ou espaço.\n
        É parametrizada por um parâmetro—lambda, a taxa de ocorrência.\n
        Exemplo: Analisando o número de gols que um time marcará durante um período específico.\n
        """)
    elif dist_name == "Exponential Distribution":
        st.latex(r'f(x|\lambda) = \lambda e^{-\lambda x}')
        st.write("""
        Uma distribuição de probabilidade contínua que modela o tempo entre eventos ocorrendo em um processo de Poisson.\n
        É parametrizada por um parâmetro—lambda, a taxa média de eventos.\n
        Exemplo: Analisando o tempo entre gols marcados por um time.\n
        """)
    elif dist_name == "Gamma Distribution":
        st.latex(r'f(x|\alpha,\beta) = \frac{\beta^\alpha x^{\alpha-1} e^{-\beta x}}{\Gamma(\alpha)}')
        st.write("""
        É uma variação da distribuição exponencial.\n
        Uma distribuição de probabilidade contínua que modela o tempo de espera para um número especificado de eventos em um processo de Poisson.\n
        É parametrizada por dois parâmetros—alpha (forma) e beta (taxa).\n
        Exemplo: Analisando o tempo que levaria para um time marcar, digamos, três gols.\n
        """)
    elif dist_name == "Beta Distribution":
        st.latex(r'f(x|\alpha,\beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}')
        st.write("""
        É usada para modelar probabilidades, portanto, é limitada entre [0,1].\n
        Difere da Binomial no aspecto de que na Binomial, a probabilidade é um parâmetro.\n
        Mas na Beta, a probabilidade é uma variável aleatória.\n
        """)
    elif dist_name == "Uniform Distribution":
        st.latex(r'f(x|a,b) = \frac{1}{b-a}')
        st.write("""
        Todos os resultados dentro de um determinado intervalo são igualmente prováveis.\n
        Pode ser contínua ou discreta.\n
        É parametrizada por dois parâmetros: a (valor mínimo) e b (valor máximo).\n
        Exemplo: Simulando a rolagem de um dado justo de seis faces, onde cada resultado (1, 2, 3, 4, 5, 6) tem uma probabilidade igual.\n
        """)
    elif dist_name == "Log-Normal Distribution":
        st.latex(r'f(x|\mu,\sigma) = \frac{1}{x\sigma \sqrt{2\pi}} e^{-\frac{(\ln x - \mu)^2}{2\sigma^2}}')
        st.write("""
        Uma distribuição de probabilidade contínua onde o logaritmo da variável segue uma distribuição normal.\n
        É parametrizada por dois parâmetros—média e desvio padrão.\n
        Exemplo: Normalmente, nos retornos de ações, o logaritmo natural segue uma distribuição normal.\n
        """)
    elif dist_name == "Student t-distribution":
        st.latex(r'f(x|\nu) = \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\nu \pi} \Gamma(\frac{\nu}{2})} \left(1+\frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}')
        st.write("""
        Semelhante à distribuição normal, mas com caudas mais longas.\n
        É usada no t-SNE para modelar similaridades de pares em baixa dimensão.\n
        """)
    

# Definir uma variável para verificar se a distribuição foi gerada
distribuicao_gerada = False

# Título
if not distribuicao_gerada:
    st.title("Aplicativo de Distribuições Estatísticas")
    st.write("""
    Bem-vindo ao aplicativo de distribuições estatísticas! 
    Este aplicativo oferece uma maneira interativa de aprender sobre diversas distribuições estatísticas utilizadas em ciência de dados. 
    Selecione uma distribuição da lista à esquerda, ajuste os parâmetros conforme necessário e clique em "Gerar" para visualizar a distribuição.
    """)
else:
    st.title(dist_name)

distribution = st.sidebar.selectbox("Escolha a distribuição:", [
    "Normal Distribution", 
    "Bernoulli Distribution", 
    "Binomial Distribution", 
    "Poisson Distribution", 
    "Exponential Distribution", 
    "Gamma Distribution", 
    "Beta Distribution", 
    "Uniform Distribution", 
    "Log-Normal Distribution", 
    "Student t-distribution"
])

if distribution == "Normal Distribution":
    mean = st.sidebar.number_input("Mean", value=0.0)
    std = st.sidebar.number_input("Standard Deviation", value=1.0, min_value=0.01)
    params = [mean, std]
elif distribution == "Bernoulli Distribution":
    p = st.sidebar.slider("Probability of success", 0.0, 1.0, 0.5)
    params = [p]
elif distribution == "Binomial Distribution":
    n = st.sidebar.number_input("Number of trials", value=10, min_value=1, format="%i")
    p = st.sidebar.slider("Probability of success", 0.0, 1.0, 0.5)
    params = [n, p]
elif distribution == "Poisson Distribution":
    lam = st.sidebar.number_input("Rate of occurrence (lambda)", value=5.0, min_value=0.1)
    params = [lam]
elif distribution == "Exponential Distribution":
    lam = st.sidebar.number_input("Average rate of events (lambda)", value=1.0, min_value=0.01)
    params = [lam]
elif distribution == "Gamma Distribution":
    a = st.sidebar.number_input("Shape (alpha)", value=2.0, min_value=0.1)
    b = st.sidebar.number_input("Rate (beta)", value=1.0, min_value=0.01)
    params = [a, b]
elif distribution == "Beta Distribution":
    a = st.sidebar.number_input("Alpha", value=2.0, min_value=0.1)
    b = st.sidebar.number_input("Beta", value=2.0, min_value=0.1)
    params = [a, b]
elif distribution == "Uniform Distribution":
    a = st.sidebar.number_input("Minimum value (a)", value=0.0)
    b = st.sidebar.number_input("Maximum value (b)", value=1.0)
    params = [a, b]
elif distribution == "Log-Normal Distribution":
    mean = st.sidebar.number_input("Mean", value=0.0)
    std = st.sidebar.number_input("Standard Deviation", value=1.0, min_value=0.01)
    params = [mean, std]
elif distribution == "Student t-distribution":
    df = st.sidebar.number_input("Degrees of freedom", value=10.0, min_value=1.0)
    params = [df]

if st.sidebar.button("Generate"):
    distribuicao_gerada = True
    plot_distribution(distribution, params)


