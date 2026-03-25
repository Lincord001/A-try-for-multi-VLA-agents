"""RAG and semantic-topology pipeline modules."""


def main() -> None:
    # Lazy import avoids runpy warning when invoking `python -m rag_pipeline.rag_main`.
    from .rag_main import main as _entry_main

    _entry_main()


__all__ = ["main"]
