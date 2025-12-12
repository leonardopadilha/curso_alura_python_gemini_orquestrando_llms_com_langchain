from langchain.agents import AgentExecutor
from orquestrador import AgenteOrquestrador

def main():
    agente = AgenteOrquestrador()
    orquestrador = AgentExecutor(
        agent=agente.agente,
        tools=agente.tools,
        verbose=True
    )

    pergunta = "Me explique sobre a função map() em Python"
    resposta = orquestrador.invoke({"input": pergunta})
    print(resposta)

if __name__ == "__main__":
    main()