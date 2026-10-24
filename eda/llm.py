"""Cliente mínimo para dos proveedores de LLM: Groq (nube) y Ollama (local).

Ambos reciben la misma lista de mensajes estilo chat y devuelven texto.
"""

from __future__ import annotations

import os
import re

import httpx

PROVIDERS = ("groq", "ollama")

GROQ_DEFAULT = "openai/gpt-oss-120b"
GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

# Modelos pequeños que caben en ~8 GB de RAM y funcionan sin GPU.
OLLAMA_DEFAULT = "llama3.2:3b"
OLLAMA_SUGGESTED = ["llama3.2:3b", "qwen2.5:3b", "gemma3:4b", "llama3.2:1b"]
OLLAMA_URL = "http://localhost:11434"


def default_provider() -> str:
    return os.getenv("LLM_PROVIDER", "groq").lower()


def default_model(provider: str) -> str:
    if provider == "ollama":
        return os.getenv("OLLAMA_MODEL", OLLAMA_DEFAULT)
    return os.getenv("GROQ_MODEL", GROQ_DEFAULT)


def ollama_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", OLLAMA_URL).rstrip("/")


def ollama_models(base_url: str | None = None) -> list[str]:
    """Modelos descargados en Ollama; lista vacía si el servidor no responde."""
    try:
        r = httpx.get(f"{base_url or ollama_url()}/api/tags", timeout=3)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))
    except (httpx.HTTPError, ValueError):
        return []


def is_available(provider: str, api_key: str | None = None) -> bool:
    if provider == "groq":
        return bool(api_key or os.getenv("GROQ_API_KEY"))
    if provider == "ollama":
        return bool(ollama_models())
    return False


def _clean(text: str) -> str:
    # Algunos modelos (Qwen, DeepSeek) incluyen su razonamiento entre etiquetas <think>.
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    if not text:
        raise RuntimeError("El modelo devolvió una respuesta vacía.")
    return text


def chat(
    messages: list[dict],
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 6000,
) -> str:
    provider = (provider or default_provider()).lower()
    model = model or default_model(provider)

    if provider == "groq":
        from groq import Groq  # import diferido: el resto del paquete funciona sin groq

        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Falta GROQ_API_KEY.")
        response = Groq(api_key=api_key).chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            # Holgado: los modelos de razonamiento gastan tokens antes de responder.
            max_tokens=max_tokens,
        )
        return _clean(response.choices[0].message.content)

    if provider == "ollama":
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            # El contexto por defecto de Ollama (2-4k tokens) se queda corto para el perfil.
            "options": {"temperature": temperature, "num_ctx": 8192, "num_predict": min(max_tokens, 3000)},
        }
        try:
            # En CPU un modelo de 3B puede tardar varios minutos en redactar el informe.
            r = httpx.post(f"{ollama_url()}/api/chat", json=payload, timeout=900)
        except httpx.ConnectError as exc:
            raise RuntimeError("Ollama no está en ejecución (abre la app Ollama o ejecuta 'ollama serve').") from exc
        if r.status_code == 404:
            raise RuntimeError(f"El modelo '{model}' no está descargado. Ejecuta: ollama pull {model}")
        r.raise_for_status()
        return _clean(r.json()["message"]["content"])

    raise ValueError(f"Proveedor desconocido: {provider}. Usa uno de {PROVIDERS}.")
