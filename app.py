
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# 1. Configuração da página
# --------------------------------------------------
st.set_page_config(
    page_title="Previsão de Evasão Acadêmica",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# 2. Carregamento do modelo treinado no Colab
# --------------------------------------------------
CAMINHO_MODELO = Path(__file__).parent / "modelo_evasao.joblib"

st.title("🎓 Previsão de Evasão Acadêmica")
st.caption(
    "Projeto acadêmico | Machine Learning aplicado à educação"
)

try:
    pacote = joblib.load(CAMINHO_MODELO)
    modelo = pacote["modelo"]
    colunas_entrada = pacote["colunas_entrada"]
    limiar = float(pacote["limiar"])
    metricas = pacote["metricas_teste"]
except FileNotFoundError:
    st.error(
        "Arquivo modelo_evasao.joblib não encontrado. "
        "Adicione o arquivo à raiz do repositório no GitHub."
    )
    st.stop()
except Exception as erro:
    st.error(f"Não foi possível carregar o modelo: {erro}")
    st.stop()

st.info(
    f"Modelo: Regressão Logística | "
    f"Limiar de classificação: {limiar:.2f}"
)

# --------------------------------------------------
# 3. Navegação
# --------------------------------------------------
aba_visao, aba_modelos, aba_previsao, aba_metodologia = st.tabs(
    [
        "Visão geral",
        "Comparação dos modelos",
        "Simular previsões",
        "Metodologia"
    ]
)

# --------------------------------------------------
# 4. Visão geral
# --------------------------------------------------
with aba_visao:
    st.header("Resultados da avaliação final")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Acurácia", f"{metricas['Accuracy']:.2%}")
    c2.metric("Precisão", f"{metricas['Precision']:.2%}")
    c3.metric("Recall", f"{metricas['Recall']:.2%}")
    c4.metric("AUC", f"{metricas['AUC']:.2%}")

    st.subheader("O que esses indicadores significam?")

    st.markdown(
        """
        - **Acurácia:** proporção total de previsões corretas.
        - **Precisão:** entre os estudantes classificados como risco de evasão,
          quantos realmente pertencem à classe de evasão no conjunto avaliado.
        - **Recall:** entre os casos reais de evasão no conjunto avaliado,
          quantos o modelo conseguiu identificar.
        - **AUC:** capacidade de distinguir as duas classes considerando
          diferentes limiares de classificação.
        """
    )

    st.subheader("Matriz de confusão — limiar 0,30")

    matriz = pd.DataFrame(
        [[516, 85], [48, 236]],
        index=["Real: não evasão", "Real: evasão"],
        columns=["Previsto: não evasão", "Previsto: evasão"]
    )

    fig_matriz = px.imshow(
        matriz,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(
            x="Classe prevista",
            y="Classe real",
            color="Quantidade"
        )
    )
    st.plotly_chart(fig_matriz, use_container_width=True)

    st.caption(
        "Valores obtidos na avaliação final do conjunto de teste. "
        "A classe de evasão foi definida como 1."
    )

    st.subheader("Por que usar o limiar 0,30?")

    st.write(
        "O limiar de 0,30 foi escolhido na validação para priorizar "
        "a identificação de mais casos de evasão. No teste final, "
        "o recall foi de 83,10%, contra 70,42% com o limiar de 0,50. "
        "Essa escolha também aumenta os falsos positivos."
    )

# --------------------------------------------------
# 5. Comparação dos quatro modelos
# --------------------------------------------------
with aba_modelos:
    st.header("Comparação dos modelos avaliados")

    dados_modelos = pd.DataFrame({
        "Modelo": [
            "Regressão Logística",
            "Árvore de Decisão",
            "Random Forest",
            "Gradient Boosting"
        ],
        "Acurácia": [0.8667, 0.7808, 0.8599, 0.8576],
        "Precisão": [0.8547, 0.6573, 0.8604, 0.8319],
        "Recall": [0.7042, 0.6620, 0.6725, 0.6972],
        "F1-Score": [0.7722, 0.6596, 0.7549, 0.7586],
        "AUC": [0.9126, 0.7495, 0.8995, 0.9066]
    })

    st.dataframe(
        dados_modelos.style.format({
            "Acurácia": "{:.2%}",
            "Precisão": "{:.2%}",
            "Recall": "{:.2%}",
            "F1-Score": "{:.2%}",
            "AUC": "{:.2%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    metrica_selecionada = st.selectbox(
        "Escolha uma métrica para comparar:",
        ["Acurácia", "Precisão", "Recall", "F1-Score", "AUC"]
    )

    fig_modelos = px.bar(
        dados_modelos,
        x="Modelo",
        y=metrica_selecionada,
        color="Modelo",
        text=dados_modelos[metrica_selecionada].map(
            lambda valor: f"{valor:.1%}"
        ),
        title=f"Comparação por {metrica_selecionada}"
    )
    fig_modelos.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig_modelos, use_container_width=True)

    st.caption(
        "As métricas desta comparação foram calculadas na avaliação "
        "inicial dos modelos com limiar 0,50. A avaliação final da "
        "Regressão Logística com limiar 0,30 é apresentada na visão geral."
    )

# --------------------------------------------------
# 6. Simulação de previsões
# --------------------------------------------------
with aba_previsao:
    st.header("Simular previsões de evasão")

    st.write(
        "Envie um arquivo CSV com as variáveis utilizadas no treinamento. "
        "O sistema calculará a probabilidade estimada de evasão e aplicará "
        "o limiar de classificação de 0,30."
    )

    st.warning(
        "Utilize dados anonimizados. A previsão é uma estimativa do modelo "
        "e não deve ser usada isoladamente para tomar decisões sobre estudantes."
    )

    arquivo = st.file_uploader(
        "Selecione o arquivo CSV",
        type=["csv"]
    )

    if arquivo is not None:
        try:
            dados = pd.read_csv(arquivo, sep=None, engine="python")

            faltantes = [
                coluna for coluna in colunas_entrada
                if coluna not in dados.columns
            ]

            if faltantes:
                st.error(
                    "O arquivo não contém todas as variáveis necessárias "
                    "para a previsão."
                )
                st.write("Colunas ausentes:", faltantes)
                st.info(
                    "Use o arquivo de exemplo exportado do Colab "
                    "como referência para o formato esperado."
                )
            elif dados.empty:
                st.error("O arquivo não possui linhas para prever.")
            else:
                # Mantém somente as variáveis esperadas e a ordem original.
                X_novos = dados[colunas_entrada].copy()

                if X_novos.isnull().any().any():
                    st.error(
                        "Existem valores vazios nas variáveis de entrada. "
                        "Corrija-os antes de gerar as previsões."
                    )
                else:
                    probabilidades = modelo.predict_proba(X_novos)[:, 1]
                    previsoes = (probabilidades >= limiar).astype(int)

                    resultado = dados.copy()
                    resultado["Probabilidade_evasao"] = probabilidades
                    resultado["Previsao"] = previsoes
                    resultado["Classificacao"] = [
                        "Risco identificado" if classe == 1
                        else "Risco não identificado pelo modelo"
                        for classe in previsoes
                    ]

                    st.success(
                        f"Previsões concluídas para {len(resultado)} registros."
                    )

                    total_risco = int((previsoes == 1).sum())
                    total_sem_risco = int((previsoes == 0).sum())

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Registros analisados", len(resultado))
                    c2.metric("Risco identificado", total_risco)
                    c3.metric("Risco não identificado", total_sem_risco)

                    st.subheader("Resultados das previsões")

                    st.dataframe(
                        resultado,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        "Baixar resultados em CSV",
                        data=resultado.to_csv(
                            index=False
                        ).encode("utf-8-sig"),
                        file_name="resultados_previsao_evasao.csv",
                        mime="text/csv"
                    )

                    fig_risco = px.histogram(
                        resultado,
                        x="Probabilidade_evasao",
                        nbins=20,
                        title="Distribuição das probabilidades estimadas",
                        labels={
                            "Probabilidade_evasao": "Probabilidade estimada",
                            "count": "Quantidade de registros"
                        }
                    )
                    fig_risco.add_vline(
                        x=limiar,
                        line_dash="dash",
                        annotation_text=f"Limiar {limiar:.2f}"
                    )
                    st.plotly_chart(
                        fig_risco,
                        use_container_width=True
                    )

        except Exception as erro:
            st.error(
                "Não foi possível processar o CSV. "
                "Confira o formato e os tipos das colunas."
            )
            st.caption(f"Detalhe técnico: {erro}")

# --------------------------------------------------
# 7. Metodologia e limitações
# --------------------------------------------------
with aba_metodologia:
    st.header("Como o projeto foi desenvolvido")

    st.markdown(
        """
        **Base de dados:** Predict Students' Dropout and Academic Success,
        disponibilizada no UCI Machine Learning Repository.

        **Preparação:** transformação do alvo em classificação binária,
        exclusão das variáveis do segundo semestre e preparação das
        variáveis numéricas e categóricas.

        **Divisão dos dados:** 80% para treinamento e 20% para teste,
        com estratificação das classes.

        **Modelos comparados:** Regressão Logística, Árvore de Decisão,
        Random Forest e Gradient Boosting.

        **Modelo utilizado nas simulações:** Regressão Logística,
        com limiar de classificação 0,30.
        """
    )

    st.subheader("Limitações")

    st.markdown(
        """
        - O modelo foi avaliado em uma única base pública.
        - O resultado depende das características dos dados disponíveis.
        - O aumento do recall vem acompanhado de mais falsos positivos.
        - A importância das variáveis não comprova causalidade.
        - Uma aplicação real exigiria avaliação humana, proteção dos dados
          e análise de possíveis vieses.
        """
    )

    st.caption(
        "Este dashboard é um protótipo acadêmico para fins demonstrativos."
    )

# --------------------------------------------------
# 8. Rodapé
# --------------------------------------------------
st.divider()
st.caption(
    "Projeto acadêmico — Sistema de previsão de evasão acadêmica "
    "utilizando Machine Learning."
)
