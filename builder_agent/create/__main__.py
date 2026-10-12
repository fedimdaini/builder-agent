"""python -m builder_agent.create <repo> [--dry-run] [--model qwen2.5-coder:7b] [--slots FILE] [--expected FILE]
    [--no-write-back]

Runs the whole Builder on <repo> and, only if every sandbox stage passes, commits the generated files on a
new builder/<date> branch (the repo is left on its original branch). --dry-run: everything but the write.
--slots: use a slot answer file instead of the LLM. --expected: a known prediction for the smoke test.
--no-write-back: don't add this run's checked fixes to memory/learned/."""

import argparse
import json
import sys
from pathlib import Path

from ..llm import OllamaClient
from . import MEMORY_SET, RETRIEVER, create


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m builder_agent.create", description=__doc__)
    p.add_argument("repo")
    p.add_argument("--dry-run", action="store_true", help="show what would be written; never touch the repo")
    p.add_argument("--model", default="qwen2.5-coder:7b", help="Ollama model for slot filling and diagnosis")
    p.add_argument("--slots", help="slot answer file instead of the LLM (recorded as such in the commit)")
    p.add_argument("--expected", help="expected prediction JSON for the smoke test")
    p.add_argument("--embed-model", default="nomic-embed-text", help="Ollama embedding model for the retriever")
    p.add_argument("--no-write-back", action="store_true", help="don't write this run's checked fixes to memory")
    args = p.parse_args()

    from ..llm.ollama import OllamaEmbedder
    from ..memory import FixMemory, retriever
    from ..memory.learned import memory_docs, snapshot_sha256, write_back
    docs = memory_docs(MEMORY_SET)
    retrieve = retriever(FixMemory.build(OllamaEmbedder(args.embed_model), docs=docs), RETRIEVER)

    def back(result):
        result.memory_set, result.memory_sha256 = MEMORY_SET, snapshot_sha256(docs)
        if not args.no_write_back:
            write_back(result, None, evaluation=False, memory_set=MEMORY_SET, memory_sha256=result.memory_sha256)

    expected = json.loads(Path(args.expected).read_text(encoding="utf-8")) if args.expected else None
    res = create(args.repo, OllamaClient(args.model), slots_file=args.slots, expected=expected, dry_run=args.dry_run,
                 retriever=retrieve, retriever_name=RETRIEVER, write_back=back)
    sys.stdout.reconfigure(encoding="utf-8")
    print(res.text())
    sys.exit(0 if res.status in ("written", "dry-run") else 1)


if __name__ == "__main__":
    main()
