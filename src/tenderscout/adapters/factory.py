from langchain_openai import ChatOpenAI

from tenderscout.adapters.fake_llm import FakeLLM
from tenderscout.adapters.openai_llm import OpenAILLM
from tenderscout.config import Settings
from tenderscout.ports.llm import LLMPort


def build_llm(settings: Settings) -> LLMPort:
    if settings.llm_provider == "fake":
        return FakeLLM()
    if not settings.openai_api_key.get_secret_value():
        raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
    model = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
        timeout=60,
        max_retries=2,
    )
    return OpenAILLM(model)
