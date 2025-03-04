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

    # # To have access to older message
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="output")

    chat_llm = ChatMistralAI(
        model=model,
        temperature=temperature,
        openai_api_key=MISTRAL_API_KEY,
        rate_limiter=rate_limiter,
        # memory=memory,
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
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="output", input_key="input"
        )

    # Check if chat.memory exists - if using the mistral API it might not store memory directly
    chat_memory = getattr(chat, "memory", None)
    if chat_memory and hasattr(chat_memory, "chat_memory") and hasattr(chat_memory.chat_memory, "messages"):
        # Transfer messages from chat memory to agent memory if possible
        memory.chat_memory.messages = chat_memory.chat_memory.messages

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
