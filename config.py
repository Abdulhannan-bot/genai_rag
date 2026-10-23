import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER","grok").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "huggingface_local").lower()
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-small-en-v1.5")


def _require(env_var: str) -> str:
    value = os.getenv(env_var)
    if not value:
        raise RuntimeError(
            f"Missing {env_var}. Copy .env.example to .env and fill it in, "
            f"or export {env_var} in your shell."
        )
    return value

def get_llm(temperature: float = 0.0):
    from langchain_openai import ChatOpenAI

    if LLM_PROVIDER == "grok":
        return ChatOpenAI(
            model=LLM_MODEL,
            api_key=_require("GROK_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            temperature=temperature,
        )

    elif LLM_PROVIDER == "openrouter":
        return ChatOpenAI(
            model=LLM_MODEL,
            api_key=_require("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
        )

    elif LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=_require("GOOGLE_API_KEY"),
            temperature=temperature,
        )

    else:
        raise ValueError(f"Unknown LLM_PROVIDER={LLM_PROVIDER!r} (use groq | openrouter | gemini)")

def get_embeddings():
    if EMBEDDINGS_PROVIDER == "huggingface_local":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

    elif EMBEDDINGS_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=_require("GOOGLE_API_KEY"),
        )

    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER={EMBEDDINGS_PROVIDER!r} (use huggingface_local | gemini)")