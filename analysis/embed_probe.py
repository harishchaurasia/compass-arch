"""Does a REAL semantic embedding beat the lexical proxy on the discrimination gap?

FINDINGS.md section 11 showed a TF-IDF affinity between a high-risk action and the
user's request adds real held-out signal on retail (0.40 -> 0.54 AUC) but is too
crude to beat the shipped rule on airline, where long requests share generic
vocabulary with any reservation. The open question was whether that ceiling is the
*idea* (semantic match doesn't carry enough signal) or the *proxy* (TF-IDF is too
shallow). This script answers it by swapping TF-IDF for genuine sentence embeddings
(nomic-embed-text via a local Ollama server) behind the exact same feature interface
and re-running the identical held-out CV harness.

Everything except the cosine backend is shared with semantic_probe / learned_probe,
so any difference in AUC is attributable to the embedding, not the pipeline.

Requires: a local Ollama server with `nomic-embed-text` pulled
  (curl http://localhost:11434/api/tags  should list it; else `ollama pull nomic-embed-text`).
Embeddings are cached to analysis/.embed_cache.json (gitignored), so the first run
does the model calls and every rerun is instant and offline.

Usage:  uv run python analysis/embed_probe.py
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from discrimination import DOMAINS, auc  # noqa: E402
from learned_probe import AFFINITY_FEATS, BASE_FEATS, evaluate  # noqa: E402
from semantic_probe import SemProbe, collect  # noqa: E402

OLLAMA = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"
MAX_CHARS = 2000  # nomic context is ~2k tokens; observations can be long JSON blobs
CACHE = Path(__file__).resolve().parent / ".embed_cache.json"


class Embedder:
    """cosine(a, b) over sentence embeddings, with a persistent on-disk cache
    keyed by text so a rerun makes zero model calls. Same interface as
    semantic_probe.Tfidf (a single .cosine method), so it drops straight into
    collect(make_sim=...)."""

    def __init__(self, model: str = MODEL):
        self.model = model
        self._cache: dict[str, list[float]] = {}
        self._dirty = False
        if CACHE.exists():
            self._cache = json.loads(CACHE.read_text(encoding="utf-8"))

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.model}\0{text}".encode()).hexdigest()

    def _embed(self, text: str) -> list[float]:
        text = text[:MAX_CHARS]
        key = self._key(text)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        body = json.dumps({"model": self.model, "prompt": text}).encode()
        req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            vec = json.loads(resp.read())["embedding"]
        self._cache[key] = vec
        self._dirty = True
        return vec

    def cosine(self, a: str, b: str) -> float:
        va, vb = self._embed(a), self._embed(b)
        dot = sum(x * y for x, y in zip(va, vb))
        na = math.sqrt(sum(x * x for x in va))
        nb = math.sqrt(sum(y * y for y in vb))
        return dot / (na * nb) if na and nb else 0.0

    def save(self) -> None:
        if self._dirty:
            CACHE.write_text(json.dumps(self._cache), encoding="utf-8")


def _fmt(x: float) -> str:
    return "  nan" if x != x else f"{x:.3f}"


def _univariate(probes: list[SemProbe], name: str) -> None:
    """Per-model AUC of each embedding affinity feature vs clean-not-compound,
    directly comparable to semantic_probe's TF-IDF table."""
    doms = sorted({p.domain for p in probes})
    print(f"\n{name} affinity, univariate AUC (clean > compound), by domain:")
    print(f"  {'domain':8s} {'n':>4s} {'#comp':>5s}  {'shipped':>7s}  "
          f"{'target':>7s}  {'action':>7s}  {'minjust':>7s}")
    for d in doms:
        r = [p for p in probes if p.domain == d]
        clean = [1 - p.compound for p in r]
        cols = {
            "shipped": [p.mean_success_prob for p in r],
            "target": [p.target_affinity for p in r],
            "action": [p.action_affinity for p in r],
            "minjust": [p.min_arg_justif for p in r],
        }
        vals = "  ".join(f"{_fmt(auc(cols[k], clean)):>7s}"
                         for k in ("shipped", "target", "action", "minjust"))
        print(f"  {d:8s} {len(r):>4d} {sum(p.compound for p in r):>5d}  {vals}")


def main() -> int:
    try:
        embedder = Embedder()
        # Fail fast with a clear message if the server / model is unavailable.
        embedder.cosine("ping", "pong")
    except (urllib.error.URLError, KeyError, ConnectionError) as e:
        print(f"Could not reach Ollama embeddings ({MODEL}) at {OLLAMA}: {e}\n"
              f"Start the server and `ollama pull {MODEL}`, then rerun.", file=sys.stderr)
        return 1

    embed_probes: list[SemProbe] = []
    tfidf_probes: list[SemProbe] = []
    for dom in DOMAINS:
        embed_probes += collect(dom, make_sim=lambda _corpus: embedder)
        tfidf_probes += collect(dom)  # default TF-IDF backend
    embedder.save()

    print(f"High-risk Compass trials (pooled): {len(embed_probes)}  "
          f"(compound {sum(p.compound for p in embed_probes)})")
    _univariate(embed_probes, "EMBEDDING")

    print("\nHeld-out AUC (mean over seeds, 5-fold CV) - full probe, TF-IDF vs embedding:")
    print(f"  {'backend':12s} {'pooled':>14s}  {'retail':>7s}  {'airline':>7s}")
    for name, probes in (("TF-IDF", tfidf_probes), ("Embedding", embed_probes)):
        r = evaluate(probes, BASE_FEATS + AFFINITY_FEATS)
        pd = r["per_domain"]
        print(f"  {name:12s} {r['pooled_mean']:.3f} +/-{r['pooled_std']:.3f}   "
              f"{pd.get('retail', float('nan')):>7.3f}  {pd.get('airline', float('nan')):>7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
