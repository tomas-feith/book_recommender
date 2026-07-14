"""Fine-tune the content encoder to be *collaborative-aware* (cold-start).

The item-item CF core (EASE-R) is silent on a book with no ratings -- an all-zero
row -- so brand-new / unrated books are ranked purely by content. This distills
EASE's co-read structure INTO the content embeddings so that even an unseen book,
from its text alone, lands near the books it would be co-read with.

Method: build (anchor, positive) pairs from each book's strongest EASE neighbors,
then fine-tune bge-small with an in-batch contrastive (InfoNCE) objective. A
leakage-free cold-start eval (books held out of both the CF matrix AND the
training pairs) confirmed this generalizes to unseen books (~+12% content
Recall@10), which is capability EASE structurally cannot provide.

Output: a SentenceTransformer directory at ``data/coread-encoder`` that
``scripts/build_embeddings.py`` picks up automatically. Regenerable; gitignored.

Run (needs the real dataset built, in the torch env):
    uv run --no-sync python scripts/finetune_coread.py --steps 60
    uv run --no-sync python scripts/build_embeddings.py   # re-embed with it
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse

from eval.data import book_to_text, load_books

BASE_MODEL = "BAAI/bge-small-en-v1.5"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "coread-encoder"
CKPT = DATA / "coread-encoder.ckpt.pt"


def build_pairs(books: list[dict], neighbors: int) -> list[tuple[int, int]]:
    """(anchor, positive) index pairs from each book's top EASE neighbors."""
    cf = np.load(DATA / "real_cf.npz", allow_pickle=True)
    cfpos = {str(b): i for i, b in enumerate(cf["ids"].tolist())}
    perm = np.array([cfpos[b["id"]] for b in books])
    sim = sparse.csr_matrix(
        (cf["sim_data"], cf["sim_indices"], cf["sim_indptr"]), shape=tuple(cf["sim_shape"])
    )[perm][:, perm].tocsr()
    pairs: list[tuple[int, int]] = []
    for i in range(len(books)):
        row = sim.getrow(i)
        for j in row.indices[np.argsort(-row.data)][:neighbors]:
            pairs.append((i, int(j)))
    return pairs


def encode(model, tok, texts: list[str], bs: int, grad: bool) -> torch.Tensor:
    outs = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for s in range(0, len(texts), bs):
            enc = tok(
                texts[s : s + bs],
                padding=True,
                truncation=True,
                max_length=160,
                return_tensors="pt",
            )
            outs.append(F.normalize(model(**enc).last_hidden_state[:, 0], dim=1))  # CLS pooling
    return torch.cat(outs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune bge-small on EASE co-read pairs.")
    ap.add_argument("--steps", type=int, default=60, help="Optimizer steps (converges ~40-60).")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--neighbors", type=int, default=4, help="EASE neighbors per book -> pairs.")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from transformers import AutoModel, AutoTokenizer  # heavy; import lazily

    books = load_books(DATA / "real_books.json")
    texts = [book_to_text(b) for b in books]
    pairs = build_pairs(books, args.neighbors)
    random.Random(0).shuffle(pairs)
    print(f"{len(books)} books | {len(pairs)} co-read pairs | {args.steps} steps", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModel.from_pretrained(BASE_MODEL)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start = 0
    if CKPT.exists():
        ck = torch.load(CKPT)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        print(f"resumed from step {start}", flush=True)

    model.train()
    bs, t0 = args.batch_size, time.perf_counter()
    for step in range(start, args.steps):
        off = (step * bs) % max(len(pairs) - bs, 1)
        batch = pairs[off : off + bs]
        a = encode(model, tok, [texts[i] for i, _ in batch], bs, grad=True)
        p = encode(model, tok, [texts[j] for _, j in batch], bs, grad=True)
        loss = F.cross_entropy((a @ p.T) * 20.0, torch.arange(len(batch)))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            print(
                f"  step {step}/{args.steps} loss={loss.item():.3f} "
                f"({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
        if step % 10 == 0 and step > start:
            tmp = CKPT.with_suffix(".tmp.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, tmp)
            tmp.replace(CKPT)

    model.eval()
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(BASE_MODEL)
    st[0].auto_model.load_state_dict(model.state_dict())  # graft the tuned weights
    st.save(str(OUT))
    CKPT.unlink(missing_ok=True)
    print(f"Saved co-read encoder -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
