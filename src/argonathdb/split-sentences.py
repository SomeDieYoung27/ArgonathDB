import warnings
from collections.abc import Callable
from functools import cache
from typing import Any

import numpy as np
from markdown_it import MarkdownIt
from scipy.optimize import OptimizeWarning, linprog
from wtpsplit_lite import SaT
from argonathdb.typing import FloatVector

@cache

def _load_sat() -> tuple[SaT,dict[str,any]]:
    """Load a Segment any Text (SaT) model."""
    sat = SAT("sat-31-sm")
    sat_kwargs = {"stride": 128, "block_size": 256, "weighting": "hat"}
    return sat,sat_kwargs


def markdown_sentence_boundaries(doc : str) -> FloatVector:
     """Determine known sentence boundaries from a Markdown document."""

     def get_markdown_heading_indexes(doc : str) -> list[tuple[int,int]]:
          md = MarkdownIt()
          tokens = md.parse(doc)
          headings = []
          lines = doc.splitlines(keepends=True)
          char_idx = [0]
          for line in lines:
               char_idx.append(char_idx[-1] + len(line))
          for token in tokens:
               if token.type == "heading_open":
                     start_line, end_line = token.map
                     heading_start = char_idx[start_line]
                     heading_end = char_idx[end_line]
                     headings.append((heading_start, heading_end))


          return headings 
     

     headings = get_markdown_heading_indexes(doc)
     boundary_probas = np.full(len(doc),np.nan)
     for heading_start,heading_end in headings:
          if 0 <= heading_start-1 < len(boundary_probas):
               boundary_probas[heading_start-1] = 1

          boundary_probas[heading_start : heading_end-1] = 0
          if 0 <= heading_end -1 < len(boundary_probas):
               boundary_probas[heading_end - 1] = 1

     return boundary_probas

def _split_sentences(
          doc : str,
          probas : FloatVector,
          *, min_len: int, max_len: int | None = None
) -> list[str]:
      # Solve an optimisation problem to find the best sentence boundaries given the predicted
    # boundary probabilities. The objective is to select boundaries that maximise the sum of the
    # boundary probabilities above a given threshold, subject to the resulting sentences not being
    # shorter or longer than the given minimum or maximum length, respectively.
    sentence_threshold = 0.25 
    c = probas - sentence_threshold
    N = len(probas)
    M = N - min_len + 1
    diagonals = [np.ones(M, dtype=np.float32) for _ in range(min_len)]
    offsets = list(range(min_len))
    A_min = sparse.diags(diagonals,offsets,shape=(M, N), format="csr")
    b_min = np.ones(M, dtype=np.float32)
    bounds = [(0,1)] * N
    bounds[:min_len-1] = [(0, 0)] * (min_len - 1)
    bounds[-min_len:] = [(0,0)] * min_len
    if max_len is not None and (M := N-max_len+1) > 0:
        diagonals = [np.ones(M, dtype=np.float32) for _ in range(max_len)]
        offsets = list(range(max_len))
        A_max = sparse.diags(diagonals, offsets, shape=(M, N), format="csr")
        b_max = np.ones(M, dtype=np.float32)
        A_ub = sparse.vstack([A_min, -A_max], format="csr")  # noqa: N806
        b_ub = np.hstack([b_min, -b_max])

    else:
        A_ub = A_min
        b_ub = b_min

    x0 = (probas >= sentence_threshold).astype(np.float32)
    with warnings.catch_warnings():
         warnings.filterwarnings("ignore", category=OptimizeWarning)
         res = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, x0=x0, integrality=[1] * N)

    if not res.success:
         error_message = "Optimization of sentence partitions failed."
         raise ValueError(error_message)

def split_sentences(
    doc: str,
    min_len: int = 4,
    max_len: int | None = None,
    boundary_probas: FloatVector | Callable[[str], FloatVector] = markdown_sentence_boundaries,
) -> list[str]:
     
     if len(doc) <= min_len:
          return [doc]
     
     sat,sat_kwargs = _load_sat()
     predicted_probas = sat.predict_proba(doc,**sat_kwargs)
     # Override the predicted boundary probabilities with the known boundary probabilities.
     known_probas = boundary_probas(doc) if callable(boundary_probas) else boundary_probas
     probas = predicted_probas.copy()
     probas[np.isfinite(known_probas)] = known_probas[np.isfinite(known_probas)]

     # Propagate the boundary probabilities so that whitespace is always trailing and never leading.
     is_space = np.array([char.isspace() for char in doc],dtype=np.bool_)
     start = np.where(np.insert(~is_space[:-1] & is_space[1:],len(is_space),-1,False))[0]
     end = np.where(np.insert(~is_space[1:] & is_space[:-1], 0, False))[0]
     start = start[start < np.max(end,initial=-1)]
     end = end[end > np.min(start, initial=len(is_space))]

     for i,j in zip(start,end,strict=True):
          min_proba,max_proba = np.min(probas[i:j],np.max(probas[i:j]))
          probas[i : j - 1] = min_proba  # From the non-whitespace to the penultimate whitespace char.
          probas[j - 1] = max_proba  # The last whitespace char.
          
     sentences = _split_sentences(doc,probas,min_len=min_len,max_len=None)

     if max_len is not None:
          sentences = [subsentence for sentence in sentences for subsentence in ([sentence] if len(sentence) <= max_len else _split_sentences( sentence,
                    probas[doc.index(sentence) : doc.index(sentence) + len(sentence)],
                    min_len=min_len,
                    max_len=max_len,))]
          
     return sentences
