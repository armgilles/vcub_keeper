import os

import polars as pl
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_mistralai.chat_models import ChatMistralAI

from vcub_keeper.config import CONFIG_LLM
from vcub_keeper.llm.crewai.tool_python import get_distance

load_dotenv()


MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


#######################
#### PROMPT ####
#######################

template_llm = """Réponds aux questions suivantes du mieux que tu peux. Tu as accès aux
outils suivants en sachant que les dataframes sont déjà charger :

{tools}

Utilise le format suivant:

Question: la question à laquelle tu dois répondre
Réflexion: tu dois toujours réfléchir à ce que tu vas faire
Action: l'action à entreprendre, doit être l'une de [{tool_names}]
Entrée d'Action: les paramètres de l'action
Observation: le résultat de l'action
... (ce cycle Réflexion/Action/Entrée d'Action/Observation peut se répéter 3 fois, mais si tu
obtiens le résultat, tu dois donner la réponse finale)
Réflexion: Je connais maintenant la réponse finale
Réponse Finale: la réponse finale à la question initiale

Commençons!

Question: {input}
Réflexion:{agent_scratchpad}"""

template_llm = """Tu es un assistant spécialisé dans l'analyse des données des stations VCub de Bordeaux.
        
Le dataframe contient des informations sur les stations avec les colonnes suivantes:
- station_id: l'id de la station
- date: la date 
- available_stands: le nombre de place disponible
- available_bikes: le nombre de vélos disponibles
- status: le statut de la station (1: tout vas bien / 0: Maintenance ou problème)
- lat: la lattitude de la staion
- lon: la longitude de la station
- station_name: le nom de la station

Ta réponse doit être structurée selon ce format:
- Si tu dois effectuer une action, indique "response_type: action" et fournis:
  * tool: le nom de l'outil à utiliser (python_repl_ast ou calculate_distance)
  * tool_input: les paramètres de l'outil
  * thought: ton raisonnement pour cette action

- Si tu connais la réponse finale, écris uniquement :
Final Answer: [ta réponse]
Ne génère pas d'autres réflexions ou actions après une réponse finale.
"""


# """
# Après avoir trouvé la réponse, termine simplement avec:
# Final Answer: [votre réponse concise]
# """

prefix_agent = """Tu es un assistant spécialisé dans l'analyse des données des stations VCub de Bordeaux.

Données disponibles :
Tu as accès à un dataframe contenant les colonnes suivantes :
- station_id : l'ID de la station
- date : la date
- available_stands : le nombre de places disponibles
- available_bikes : le nombre de vélos disponibles
- status : le statut de la station (1 : OK, 0 : en maintenance ou problème)
- lat : la latitude de la station
- lon : la longitude de la station
- station_name : le nom de la station

Les données ne sont pas triéer ou filtrées. Tu dois utiliser les bonnes pratiques de Pandas!
"""

prompt = prefix_agent + template_llm


prompt_gestion_erreurs = """Vérifiez votre sortie et assurez-vous
            qu'elle est conforme ! Ne sortez pas une Action et une réponse
            finale en même temps. Lorsque vous avez une Final Answer, vous
            devez vous arrêter et ne pas continuer à réfléchir."""


def create_chat(model: str, temperature: float = 0.1) -> ChatMistralAI:
    """



    Parameters
    ----------
    model : str
        _description_
    temperature : float, optional
        _description_, by default 0.0

    Returns
    -------
    ChatMistralAI
        _description_

    Examples
    --------
    chat_llm = create_chat(model="mistral-small-latest", temperature=0.1)
    """
    # To avoid rate limit errors (429 - Requests rate limit exceeded)
    rate_limiter = InMemoryRateLimiter(requests_per_second=3, check_every_n_seconds=0.3, max_bucket_size=4)

    # To have access to older message
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chat_llm = ChatMistralAI(
        model=model,
        temperature=temperature,
        openai_api_key=MISTRAL_API_KEY,
        rate_limiter=rate_limiter,
        memory=memory,
        # model_kwargs={
        #     "top_p": 0.92,
        #     "repetition_penalty": 1.1,
        #     "max_tokens": 1024,  # Limit token generation
        # },
    )

    return chat_llm


def create_agent(chat: ChatMistralAI, last_info_station: pl.DataFrame, **kwargs) -> AgentExecutor:
    """


    Parameters
    ----------
    chat : ChatMistralAI
        _description_

    Returns
    -------
    create_pandas_dataframe_agent
        _description_
    """

    memory = getattr(chat, "memory", None)
    if not memory:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="output")

    # Paramètres par défaut
    default_params = {
        "agent_type": AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        "return_intermediate_steps": CONFIG_LLM["vcub_agent"]["return_intermediate_steps"],
        "number_of_head_rows": last_info_station.shape[0],
        "prefix": CONFIG_LLM["vcub_agent_prompt"]["prefix_agent"] + CONFIG_LLM["vcub_agent_prompt"]["template_llm"],
        "extra_tools": build_tools(),
        "max_iterations": CONFIG_LLM["vcub_agent"]["max_iterations"],
        "allow_dangerous_code": CONFIG_LLM["vcub_agent"]["allow_dangerous_code"],
        "verbose": CONFIG_LLM["vcub_agent"]["verbose"],
        "early_stopping_method": CONFIG_LLM["vcub_agent"]["early_stopping_method"],
        "agent_executor_kwargs": {
            "handle_parsing_errors": CONFIG_LLM["vcub_agent_prompt"]["prompt_gestion_erreurs"],
            "memory": memory,
        },
    }

    agent = create_pandas_dataframe_agent(
        llm=chat,
        df=last_info_station.to_pandas(),
        **default_params,
    )

    return agent


def build_tools() -> list[Tool]:
    """


    Returns
    -------
    _type_
        _description_
    """

    tools = [
        Tool(
            name="calculate_distance",
            func=get_distance,
            description=CONFIG_LLM["calculate_distance_prompt"]["prompt_descrption"],
        ),
    ]

    return tools
