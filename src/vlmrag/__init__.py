"""VLM-assisted document extraction for retrieval-augmented generation.

Sub-packages:

* ``vlmrag.vlm`` - request types, backends, client (limiter, retry, cache, usage), JSON recovery
* ``vlmrag.render`` - PDF rendering (pypdfium2), image tiling, LibreOffice conversion
* ``vlmrag.augment`` - optional retrieval-summary prefix
* ``vlmrag.techniques`` - the extraction techniques (``drawing``, ``slides``, ``docx``)
* ``vlmrag.synthetic`` - deterministic synthetic fixtures with ground truth

Heavy dependencies (openai, pypdfium2) are imported lazily by the modules that
need them, so importing the package itself is cheap.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Settings  # noqa: E402
from .schema import Chunk, doc_id_for, read_jsonl, write_jsonl  # noqa: E402

__all__ = ["Chunk", "Settings", "__version__", "doc_id_for", "read_jsonl", "write_jsonl"]
