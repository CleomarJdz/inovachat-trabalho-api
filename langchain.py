# langchain.py
# Módulo para integração com LangChain e OpenAI

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Configuração do modelo
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_template(
    "Responda a seguinte pergunta: {pergunta}"
)

chain = prompt | llm

def perguntar_ia(texto):
    """
    Função para fazer perguntas à IA usando LangChain com OpenAI.
    """
    resposta = chain.invoke({
        "pergunta": texto
    })
    return resposta.content