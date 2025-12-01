from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatMaritalk
from my_models import GEMINI_FLASH, MARITACA_SABIA
from my_keys import GEMINI_API_KEY, MARITACA_API_KEY
from my_helper import encode_image

llm = ChatGoogleGenerativeAI(
    api_key=GEMINI_API_KEY,
    model=GEMINI_FLASH
)

resposta = llm.invoke("Quais canais de YouTube você recomenda para que eu possa saber mais a respeito de smartphones?")
print("Gemini: ", resposta.content)


llm = ChatMaritalk(
    api_key=MARITACA_API_KEY,
    model=MARITACA_SABIA
)

resposta = llm.invoke("Quais canais de YouTube você recomenda para que eu possa saber mais a respeito de smartphones?")
print("Maritaca: ", resposta.content)

# A análise de imagem foi removida pois o modelo da Maritaca selecionado não suporta imagens
