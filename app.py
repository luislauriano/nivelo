import streamlit as st
import pandas as pd
import pickle
import gzip
import plotly.express as px
from xgboost import XGBClassifier

# Carregamento do modelo
with gzip.open('model.pkl.gz', 'rb') as f:
    model = pickle.load(f)

# Função para preparar os dados de entrada
def prepare_input(bncc_componente, acertos, possui_diagnostico_de_transtorno, turma, ano):
    # Ajustando os dados de entrada com base nas variáveis da sua tabela
    input_data = pd.DataFrame({
        'bncc_componente': [bncc_componente],
        
        'faltas': [faltas],
        'possui_diagnostico_de_transtorno': [possui_diagnostico_de_transtorno],
        'turma': [turma]
        
    })
    
    # Convertendo variáveis categóricas em variáveis dummy
    input_data = pd.get_dummies(input_data, columns=['bncc_componente', 'turma'])
    
    # Garantindo que as colunas do input_data correspondam ao modelo treinado
    model_columns = model.feature_names_in_
    input_data = input_data.reindex(columns=model_columns, fill_value=0)
    
    return input_data

# Carregar os dados do CSV (ajustar conforme seus dados reais)
df = pd.read_csv('Dados_tratados.csv')

# Calculando algumas métricas para exibição
media_acertos = df['total_acertos'].mean()

maior_acertos = df['total_acertos'].max()


with st.container():
    col4, col5 = st.columns(2)

    with col4:
        st.markdown(
            f"""
            <style>
            .metric-container {{
                display: flex;
                align-items: center;
                font-size: 13px;
                color: #034381
            }}
            .metric-label {{
                margin-right: 10px;
            }}
            .metric-value {{
                font-size: 20px; 
                font-weight: medium;
                padding: 5px 10px;
                border-radius: 10px;
                background-color: #034381;
                color: #FFDC02;
                font-family: 'Poppins', sans-serif;
            }}
            </style>
            <div class="metric-container">
                <div class="metric-label">Média de Acertos:</div>
                <div class="metric-value">{media_acertos:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col5:
        st.markdown(
            f"""
            <style>
            .metric-container-dois {{
                display: flex;
                align-items: center;
                font-size: 13px;
                color: #034381
            }}
            .metric-label-dois {{
                margin-right: 10px;
            }}
            .metric-value-dois {{
                font-size: 20px; 
                font-weight: medium;
                padding: 5px 10px;
                border-radius: 10px;
                background-color: #FFDC02;
                color: #034381;
                font-family: 'Poppins', sans-serif;
            }}
            </style>
            <div class="metric-container-dois">
                <div class="metric-label-dois">Maior Acerto:</div>
                <div class="metric-value-dois">{maior_acertos:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Exibir gráfico total de acertos por alunos
        grouped_data = df.groupby(['id_aluno'])['total_acertos'].sum()[:100]
        st.markdown(
            """
            <style>
            .custom-subheader {{
                margin-top: 20px;
                color: #034381;
                font-size: 18px;
                font-weight: medium;
            }}
            </style>
            <div class="custom-subheader">
                Acertos por Alunos
            </div>
            """,
            unsafe_allow_html=True
        )
        st.bar_chart(grouped_data)

    with col2:
        # Média de dificuldade acertos dos alunos
        grouped_data_dois = df.groupby('id_aluno')[['dificuldade_media_acertos']].sum()[:100]

        st.markdown(
            """
            <style>
            .custom-subheader-dois {{
                margin-top: 20px;
                color: #034381;
                font-size: 18px;
                font-weight: medium;
            }}
            </style>
            <div class="custom-subheader-dois">
                Média de dificuldade acertos dos alunos
            </div>
            """,
            unsafe_allow_html=True
        )
        st.area_chart(grouped_data_dois)

    col6, col7 = st.columns(2)
    
    with col6:
        # Taxa de acerto BNCC
        grouped_data_seis = df.groupby('id_aluno')[['taxa_acerto_EF06MA', 'taxa_acerto_EF07MA']].sum()[:10]
        st.markdown(
            """
            <style>
            .custom-subheader-seis {{
                margin-top: 20px;
                color: #034381;
                font-size: 18px;
                font-weight: medium;
            }}
            </style>
            <div class="custom-subheader-seis">
                Taxa de acerto BNCC
            </div>
            """,
            unsafe_allow_html=True
        )
        st.bar_chart(grouped_data_seis)

    with col7:
        # Gráfico de tempo de estudo médio por turma
        grouped_data_sete =  df.groupby('dificuldade')[['acertou_EF06MA',
       'acertou_EF07MA', 'acertou_EF08MA', 'acertou_EF09MA', 'acertou_EM13MA']].sum()[:10]
        st.markdown(
            """
            <style>
            .custom-subheader-sete {{
                margin-top: 20px;
                color: #034381;
                font-size: 18px;
                font-weight: medium;
            }}
            </style>
            <div class="custom-subheader-sete">
                Dificuldade com base nas BNCCs
            </div>
            """,
            unsafe_allow_html=True
        )
        st.bar_chart(grouped_data_sete)

st.markdown(
    """
    <style>
    .custom-subheader-tres {
        font-size: 18px; /* Ajuste o tamanho da fonte conforme necessário */
        font-weight: bold; /* Define o peso da fonte */
        text-align: center; /* Centraliza o texto */
        margin-bottom: 10px; /* Espaço abaixo do título */
        font-family: 'Poppins', sans-serif; /* Define a fonte como Poppins */
        color: #034381;
    }
    .custom-text-tres {
        font-size: 14px; /* Ajuste o tamanho da fonte conforme necessário */
        text-align: center; /* Centraliza o texto */
        line-height: 1.5; /* Ajusta o espaçamento entre linhas */
        margin-bottom: 40px; /* Espaço abaixo do texto */
        font-weight: mixed;
        
    }
    </style>
    <div class="custom-subheader-tres">
        Insira as informações solicitadas abaixo para a previsão do modelo:
    </div>
    <div class="custom-text-tres">
        Aqui você consegue <strong>prever</strong> a chance de um aluno possuir dificuldade ou não com base <strong>na prova de nivelamento aplicada</strong>,
        podendo assim <strong>entender melhor o perfil dos alunos<br></strong> Resultado igual a 1 para quando o aluno pode possuir alguma dificuldade.
    </div>
    """,
    unsafe_allow_html=True
)

with st.container():
    
    col1, col2 = st.columns(2)
    with col1:
        bncc_componente = st.selectbox('Selecione o código BNCC', ['EF06MA', 'EF07MA','EF08MA','EF09MA', 'EM13MA'])
    with col2:
        idade = st.slider('Idade do aluno', 3, 25)
    
    col4, col5 = st.columns(2)
    with col4:
        faltas = st.number_input('Quantidade de faltas', 0, 200, 25)
    with col5:
        possui_diagnostico_de_transtorno = st.selectbox('Possui diagnostico de algum transtorno', ['Sim', 'Não'])
    
    turma = st.selectbox('Selecione a Turma', ['EM','FM'])
    

# Preparar os dados de entrada
input_data = prepare_input(bncc_componente, idade, faltas, possui_diagnostico_de_transtorno, turma)

try:
    if not input_data.empty:
        predicao = model.predict(input_data)
        st.write(f"Previsão se o aluno irá possui alguma dificuldade: {int(predicao[0])}")
    else:
        st.write("Não foi possível fazer a previsão devido a problemas com os dados de entrada.")
except Exception as e:
    st.write(f"Erro ao fazer a previsão: {e}")

st.markdown(
    """
    <style>
    .custom-subheader-quatro {
        margin-top: 20px;
        margin-bottom: 20px;
        text-align: center;
        font-size: 20px; /* Ajuste o tamanho da fonte conforme necessário */
        font-weight: bold; /* Define o peso da fonte */
        color: #034381; /* Define a cor azul */

    }
    </style>
    <div class="custom-subheader-quatro">
        Previsão de Dificuldade dos Alunos - Base de Dados
    </div>
    """,
    unsafe_allow_html=True
)

st.dataframe(df.head())
