from typing import Any, Optional, Sequence, Union

from langchain.schema import SystemMessage, HumanMessage
from langchain_community.chat_models import ChatOpenAI
from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseLanguageModel, LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableLambda, Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output
from langchain_experimental.graph_transformers import LLMGraphTransformer
from pydantic import BaseModel, Field


class CustomLLM(BaseLanguageModel, BaseModel):
    llm: ChatOpenAI = Field(...)  # LLM als Pydantic-Feld
    system_prompt: str = Field(...)  # Systemnachricht als Pydantic-Feld

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Output:
        pass

    def generate_prompt(self, prompts: list[PromptValue], stop: Optional[list[str]] = None, callbacks: Callbacks = None,
                        **kwargs: Any) -> LLMResult:
        pass

    async def agenerate_prompt(self, prompts: list[PromptValue], stop: Optional[list[str]] = None,
                               callbacks: Callbacks = None, **kwargs: Any) -> LLMResult:
        pass

    def with_structured_output(self, schema: Union[dict, type], **kwargs: Any) -> Runnable[
        LanguageModelInput, Union[dict, BaseModel]]:
        pass

    def predict(self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any) -> str:
        pass

    def predict_messages(self, messages: list[BaseMessage], *, stop: Optional[Sequence[str]] = None,
                         **kwargs: Any) -> BaseMessage:
        pass

    async def apredict(self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any) -> str:
        pass

    async def apredict_messages(self, messages: list[BaseMessage], *, stop: Optional[Sequence[str]] = None,
                                **kwargs: Any) -> BaseMessage:
        pass


# LLM-Wrapper mit festem System-Prompt

# LLM initialisieren
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    max_tokens=200
)

custom_llm = CustomLLM(llm=llm, system_prompt="Der All-Name ist Lukas.")
# Wrapper-Funktion für den System-Prompt

