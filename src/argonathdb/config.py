"""RAGLite config."""
import contextlib
import os
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Literal

from platformdirs import user_data_dir
from sqlalchemy.engine import URL


from argonathdb.lazy_llama import llama_supports_gpu_offload


with contextlib.redirect_stdout(StringIO()):
    from rerankers.models.flashrank_ranker import FlashRankRanker
    from rerankers.models.ranker import BaseRanker


cache_path = Path(user_data_dir("raglite", ensure_exists=True))


@dataclass(frozen=True)

class RagLiteConfig:
    # Database config.
    db_url: str | URL = f"sqlite:///{(cache_path / 'raglite.db').as_posix()}"
    llm : str = field(
        default_factory= lambda: (
              "llama-cpp-python/lm-kit/bge-m3-gguf/*F16.gguf@1024"
            if llama_supports_gpu_offload() or (os.cpu_count() or 1) >= 4  # noqa: PLR2004
            else "llama-cpp-python/lm-kit/bge-m3-gguf/*Q4_K_M.gguf@1024"
        )
    
    )
    llm_max_tries : int = 4
     # Embedder config used for indexing.
    embedder : str = field(
        default_factory=lambda:(
            "llama-cpp-python/lm-kit/bge-m3-gguf/*F16.gguf@1024"
            if llama_supports_gpu_offload() or (os.cpu_count() or 1) >= 4  # noqa: PLR2004
            else "llama-cpp-python/lm-kit/bge-m3-gguf/*Q4_K_M.gguf@1024"
        )
    )
    embedder_normalize : bool = True
    embedder_sentence_window_size : int = 3
    # Chunk config used to partition documents into chunks.
    chunk_max_size : int = 1440
    #Vector search config
    vector_search_index_metric = Literal["cosine","dot","l1","l2"] = "cosine"
    vector_search_query_adapter : bool = True

    #Reranking all the configs
    reranker : BaseRanker | tuple[tuple[str,BaseRanker],...] | None = field(
        default_factory=lambda : (
            ("en", FlashRankRanker("ms-marco-MiniLM-L-12-v2", verbose=0, cache_dir=cache_path)),
            ("other", FlashRankRanker("ms-marco-MultiBERT-L-12", verbose=0, cache_dir=cache_path)),
        ),
        compare=False,
    )

    def __post__init(self) -> None:
         # Late chunking with llama-cpp-python does not apply sentence windowing.
         if self.embedder.startswith("llama-cpp-python"):
             object.__setattr__(self,"embedder_sentence_window_size", 1)