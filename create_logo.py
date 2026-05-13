
# ===== LANGCHAIN IMPORTS =====
# Alteração automática realizada para integração com LangChain

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Configuração do modelo
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_template(
    "Responda a seguinte pergunta: {pergunta}"
)

chain = LLMChain(
    llm=llm,
    prompt=prompt
)

# ===== FIM LANGCHAIN =====

from PIL import Image, ImageDraw, ImageFont
import os

# Create favicon (32x32)
favicon_size = (32, 32)
favicon = Image.new('RGBA', favicon_size, (0, 123, 255, 255))  # Blue background
draw = ImageDraw.Draw(favicon)

# Try to use a font, fallback to default
try:
    font = ImageFont.truetype("arial.ttf", 12)
except:
    font = ImageFont.load_default()

# Draw "IC" for INOVACHAT
draw.text((8, 8), "IC", fill="white", font=font)

favicon.save('static/favicon.ico')

# Create logo for app (200x60)
logo_size = (200, 60)
logo = Image.new('RGBA', logo_size, (0, 123, 255, 255))
draw_logo = ImageDraw.Draw(logo)

try:
    font_logo = ImageFont.truetype("arial.ttf", 24)
except:
    font_logo = ImageFont.load_default()

# Draw "INOVACHAT 2.0"
draw_logo.text((10, 15), "INOVACHAT 2.0", fill="white", font=font_logo)

logo.save('static/logo.png')

print("Logo and favicon created!")

# ===== EXEMPLO DE USO LANGCHAIN =====
# Alteração automática adicionada

def perguntar_ia(texto):
    resposta = chain.invoke({
        "pergunta": texto
    })

    return resposta

# ===== FIM EXEMPLO =====
