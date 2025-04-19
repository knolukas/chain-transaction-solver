from abc import ABC
from typing import Optional, Any, Sequence, Union

from langchain_core.callbacks import Callbacks
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.base import BaseLanguageModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableConfig, Runnable
from langchain_core.runnables.utils import Input, Output
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import Field, ConfigDict, BaseModel

from abc import ABC
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.vectorstores import VectorStoreRetriever


class RAGModel(BaseLanguageModel, Runnable, ABC):
    retriever: VectorStoreRetriever = Field(...)
    llm: BaseLanguageModel = Field(...)

    def _call(self, prompt, stop=None):
        docs = self.retriever.get_relevant_documents(prompt)
        context = "\n".join([doc.page_content for doc in docs])
        full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:"
        return self.llm.predict(full_prompt)

    def invoke(self, prompt: str, config: Optional[RunnableConfig] = None, **kwargs) -> str:
        """Ruft das Modell mit dem Prompt auf und nutzt RAG."""
        docs = self.retriever.get_relevant_documents(prompt)
        context = "\n".join([doc.page_content for doc in docs]) if docs else "Kein Kontext verfügbar."
        full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:"

        return self.llm.invoke(full_prompt, **kwargs)

    def __call__(self, prompt: str) -> str:
        """Macht das Modell direkt aufrufbar."""
        return self.invoke(prompt)

    def get_num_tokens(self, text: str) -> int:
        return len(text.split())

    def generate_prompt(self, prompts: list[PromptValue], stop: Optional[list[str]] = None, callbacks: Callbacks = None,
                        **kwargs: Any) -> LLMResult:
        pass

    async def agenerate_prompt(self, prompts: list[PromptValue], stop: Optional[list[str]] = None,
                               callbacks: Callbacks = None, **kwargs: Any) -> LLMResult:
        pass


    def predict(self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any) -> str:
        """Muss implementiert sein, sonst ist `None` der Rückgabewert."""
        return self._call(text, stop=stop)

    def with_structured_output(self, schema: Union[dict, type], **kwargs: Any) -> Runnable:
        """Erzeugt einen Dummy-Runnable, um Fehler zu vermeiden."""
        return self

    def predict_messages(self, messages: list[BaseMessage], *, stop: Optional[Sequence[str]] = None,
                         **kwargs: Any) -> BaseMessage:
        pass

    async def apredict(self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any) -> str:
        pass

    async def apredict_messages(self, messages: list[BaseMessage], *, stop: Optional[Sequence[str]] = None,
                                **kwargs: Any) -> BaseMessage:
        pass
