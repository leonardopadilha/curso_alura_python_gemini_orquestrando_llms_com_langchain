from langchain_google_genai import ChatGoogleGenerativeAI
from my_models import GEMINI_FLASH, MARITACA_SABIA
from my_keys import GEMINI_API_KEY, MARITACA_API_KEY
from langchain_core.globals import set_debug

set_debug(False)

from langchain import hub # Para buscar templates de agents e ajudam a instruir o sistema sobre como tratar os dados de entrada
from langchain.agents import create_react_agent # Irá ajudar a criar o agente inteligente que nesse caso, consegue decidir qual é a ferramenta mais adequada para cada tarefa.
from langchain.agents import Tool
from ferramenta_analisadora_imagem import FerramentaAnalisadoraImagem
from ferramenta_explicadora import FerramentaExplicadora

class AgenteOrquestrador:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            api_key=GEMINI_API_KEY,
            model=GEMINI_FLASH
        )

        ferramenta_analisadora_imagem = FerramentaAnalisadoraImagem() # Irá analisar a imagem e retornar o resultado
        ferramenta_explicadora = FerramentaExplicadora() # Irá explicar o tema da pergunta do usuário

        self.tools = [
            Tool(
                name = ferramenta_analisadora_imagem.name,
                func = ferramenta_analisadora_imagem.run,
                description = ferramenta_analisadora_imagem.description,
                return_direct = ferramenta_analisadora_imagem.return_direct
            ),
            Tool(
                name = ferramenta_explicadora.name,
                func = ferramenta_explicadora.run,
                description = ferramenta_explicadora.description,
                return_direct = ferramenta_explicadora.return_direct
            )
        ]

        prompt = hub.pull("hwchase17/react")
        self.agente = create_react_agent(self.llm, self.tools, prompt)