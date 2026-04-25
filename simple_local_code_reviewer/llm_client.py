"""
llm_client.py
──────────────
Downloads and runs the Qwen2.5-Coder-3B-Instruct GGUF model locally via
llama_cpp. No external API calls are made during inference.
"""

import logging
from functools import cached_property

log = logging.getLogger(__name__)

# Model config (can be changed to larger models for better performance)
MODEL_CONFIGS = {
    "3b": {
        "repo_id": "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
        "filename": "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        "size_gb": 1.9,
        "context": 32768,
    },
    "7b": {
        "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "filename": "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "size_gb": 4.2,
        "context": 32768,
    },
    "14b": {
        "repo_id": "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
        "filename": "qwen2.5-coder-14b-instruct-q4_k_m.gguf",
        "size_gb": 8.0,
        "context": 32768,
    },
    "32b": {
        "repo_id": "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
        "filename": "qwen2.5-coder-32b-instruct-q4_k_m.gguf",
        "size_gb": 18.0,
        "context": 32768,
    },
}

# Default model (can be overridden)
DEFAULT_MODEL_SIZE = "7b"  # Changed from 3b to 7b for better code analysis


def get_model_config(model_size: str = DEFAULT_MODEL_SIZE):
    """Get model configuration for the specified size."""
    if model_size not in MODEL_CONFIGS:
        available = list(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model size '{model_size}'. Available: {available}")
    return MODEL_CONFIGS[model_size]

# Current model config
config = get_model_config(DEFAULT_MODEL_SIZE)
REPO_ID = config["repo_id"]
FILENAME = config["filename"]
CTX_SIZE = config["context"]

# Generation defaults
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.1   # low temp → deterministic, precise code edits
DEFAULT_TOP_P = 0.95
CTX_SIZE = 32768 # 16384  # 8192 <- default             # Qwen2.5-3B context window; full capacity = 32768


class LLMClient:
    """
    Wraps llama_cpp.Llama loaded from HuggingFace Hub.

    The model is downloaded once to the HuggingFace cache (~/.cache/huggingface)
    and reused on subsequent runs.
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        n_ctx: int | None = None,
        n_gpu_layers: int = -1,   # -1 = offload all layers to GPU if available
        verbose: bool = False,
    ):
        self.model_size = model_size
        config = get_model_config(model_size)
        self.n_ctx = n_ctx or config["context"]
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        # Trigger eager load so errors surface early
        _ = self.model

    @cached_property
    def model(self):
        """Lazy-load the GGUF model (downloaded from HuggingFace Hub if needed)."""
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is not installed.\n"
                "Install it with:  pip install llama-cpp-python\n"
                "For GPU support:  CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
            ) from e

        config = get_model_config(self.model_size)
        repo_id = config["repo_id"]
        filename = config["filename"]

        log.info(f"Loading model: {repo_id} / {filename}")
        log.info("  (Downloading from HuggingFace Hub on first run — may take a few minutes)")

        llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose,
            # local_dir="./models" # Optional: specify a custom local directory for caching models instead of ~/.cache/huggingface
        )
        log.info(f"Model {repo_id} / {filename} loaded successfully.")
        return llm

    # ── Core inference ───────────────────────────────────────────────────────

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        """
        Send a chat-formatted prompt to the model and return the assistant reply.

        Uses the ChatML format that Qwen2.5-Instruct models expect.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["<|im_end|>", "<|endoftext|>"],
        )

        return response["choices"][0]["message"]["content"].strip()
