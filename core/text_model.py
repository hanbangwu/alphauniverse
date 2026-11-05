from __future__ import annotations

from functools import cache
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from .config import QUERY_PREFIX, WEMM, WEMM_DIM
from .device import device

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


BATCH = 1024

LOADING = Lock()


@cache
def text_model() -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    kernels = (
        {
            "dtype": torch.bfloat16,
            "use_kernels": True,
            "attn_implementation": "kernels-community/flash-attn2",
        }
        if device().type == "cuda"
        else {}
    )
    model = SentenceTransformer(
        WEMM,
        device=device(),
        trust_remote_code=True,
        truncate_dim=WEMM_DIM,
        model_kwargs=kernels,
    )
    trained = Tokenizer.from_file(hf_hub_download(WEMM, "tokenizer.json"))
    model[0].processor.tokenizer.backend_tokenizer.normalizer = trained.normalizer
    return model


def embed_queries(texts: list[str]) -> np.ndarray:
    with LOADING:
        model = text_model()
    return model.encode_query(
        [QUERY_PREFIX.format(text) for text in texts], normalize_embeddings=True
    )
