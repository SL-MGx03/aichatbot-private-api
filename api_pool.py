import itertools
import os
from langchain_groq import ChatGroq

def load_groq_keys() -> list[str]:
    raw_keys = os.getenv("GROQ_API_KEYS", "")
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

    if not keys:
        single_key = os.getenv("GROQ_API_KEY")
        if single_key:
            keys.append(single_key.strip())

    if not keys:
        raise ValueError("No Groq API keys found in environment variables.")

    return keys

_keys = load_groq_keys()
_key_cycle = itertools.cycle(_keys)

def get_next_groq_key() -> str:
    return next(_key_cycle)

class DynamicChatGroq:
    """Proxy class that initializes ChatGroq with a fresh key on every invocation."""

    def __init__(self, model: str = "llama-3.3-70b-versatile", temperature: float = 0, **kwargs):
        self.model = model
        self.temperature = temperature
        self.kwargs = kwargs

    def _get_client(self) -> ChatGroq:
        return ChatGroq(
            model=self.model,
            temperature=self.temperature,
            groq_api_key=get_next_groq_key(),
            **self.kwargs
        )

    def invoke(self, *args, **kwargs):
        return self._get_client().invoke(*args, **kwargs)

    def with_structured_output(self, schema, **kwargs):
        def _structured_invoke(*args, **call_kwargs):
            client = self._get_client()
            extractor = client.with_structured_output(schema, **kwargs)
            return extractor.invoke(*args, **call_kwargs)

        class StructuredProxy:
            def invoke(self, *args, **call_kwargs):
                return _structured_invoke(*args, **call_kwargs)

        return StructuredProxy()
