import os
import openai
from dotenv import load_dotenv

load_dotenv()

maritaca_api_key = os.getenv("MARITACA_API_KEY")

client = openai.OpenAI(
    api_key=maritaca_api_key,
    base_url="https://chat.maritaca.ai/api"
)

pergunta = "Qual é o significado da expressão 'chutar o balde' no Brasil?"
response = client.chat.completions.create(
    model="sabia-3",
    messages=[
        { "role": "user", "content": pergunta }
    ],
    max_tokens=8000
)

resposta = response.choices[0].message.content
print(f"Pergunta: {pergunta}\nResposta: {resposta}")
