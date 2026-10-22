"""
Streamlit frontend for the MS MARCO passage-retrieval project
(TF-IDF, BM25, DPR, Sentence-Transformer, RRF hybrids).

Place this file in the repo root (next to app.py / run.py) and run:

    uv sync
    uv run pip install streamlit pandas        # or: uv add streamlit
    uv run streamlit run streamlit_app.py --server.fileWatcherType none

Prerequisite: `run.py` must already have built the indexes under artifacts/
(BM25 is required: it also stores the passage text).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _js_default(o):
    """JSON fallback for numpy scalars etc."""
    return o.item() if hasattr(o, "item") else str(o)


def _install_java_worker_guards() -> None:
    """Prevent the JVM worker from loading PyTorch/FAISS/sklearn/OpenMP native dylibs."""
    import importlib.abc
    import importlib.machinery
    import types

    class _StubModule(types.ModuleType):
        def __init__(self, name: str):
            super().__init__(name)
            self.__path__ = []

        def __getattr__(self, attr: str):
            if attr == "__version__":
                return "0.0.0"
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            return _StubModule(f"{self.__name__}.{attr}")

        def __call__(self, *args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return _StubModule(self.__name__)

        def __getitem__(self, key):
            return self

        def __or__(self, other):
            return self

        def __ror__(self, other):
            return self

        def __bool__(self) -> bool:
            return False

        def __iter__(self):
            return iter(())

        def __mro_entries__(self, bases):
            return (object,)

    class _BlockNativeOpenMP(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        BLOCKED_TOP = {
            "torch",
            "faiss",
            "sklearn",
            "scipy",
            "transformers",
            "sentence_transformers",
            "onnxruntime",
        }
        BLOCKED_PREFIXES = (
            "pyserini.encode",
            "pyserini.search.faiss",
            "pyserini.search.lucene._impact_searcher",
            "pyserini.search.lucene._hnsw_searcher",
        )

        def find_spec(self, fullname: str, path=None, target=None):
            if (
                fullname.split(".")[0] in self.BLOCKED_TOP
                or fullname.startswith(self.BLOCKED_PREFIXES)
            ):
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
            return None

        def create_module(self, spec):
            return _StubModule(spec.name)

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, _BlockNativeOpenMP())


def _worker_main(argv: list[str]) -> None:
    """Search worker, run in its own process and MAIN thread (like app.py / run.py).

    Three roles, each in a separate process so that no process ever holds more than one
    family of native OpenMP libraries (mixing them crashes on macOS / Apple Silicon):
      java  -> BM25 and passage text (uses the JVM; PyTorch/FAISS/sklearn imports are blocked)
      tfidf -> TF-IDF (scikit-learn)
      dense -> Sentence-Transformer / DPR (PyTorch + FAISS, never starts the JVM)
    Protocol: JSON lines on a private pipe (the original stdout); fd 1 is redirected to stderr."""
    import argparse
    from dataclasses import replace

    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)

    def send(obj) -> None:
        proto.write(json.dumps(obj, default=_js_default) + "\n")
        proto.flush()

    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--role", choices=["java", "tfidf", "dense"], default="java")
    ap.add_argument("--artifacts-dir", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--faiss-device", default="cpu")
    ap.add_argument("--jvm-heap-gb", type=int, default=0)
    a = ap.parse_args(argv)

    try:  # diagnostics: what did we inherit from the parent?
        import signal

        blocked = sorted(signal.Signals(x).name for x in signal.pthread_sigmask(signal.SIG_BLOCK, set()))
        ignored = [signal.Signals(n).name for n in signal.valid_signals()
                   if n != signal.SIGKILL and n != signal.SIGSTOP and signal.getsignal(n) == signal.SIG_IGN]
        print(f"[worker:{a.role}] blocked signals={blocked} ignored={ignored} "
              f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
              f"KMP_DUPLICATE_LIB_OK={os.environ.get('KMP_DUPLICATE_LIB_OK')}",
              file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001
        pass

    try:
        if a.role == "java":
            _install_java_worker_guards()

        from msmarco_ir.config import Config
        from msmarco_ir.utils import configure_jvm, setup_logging

        worker_device = "cpu" if a.role == "java" else a.device
        cfg = replace(Config(), artifacts_dir=Path(a.artifacts_dir), device=worker_device,
                      faiss_device=a.faiss_device, jvm_heap_gb=a.jvm_heap_gb or None)
        setup_logging()
        if a.role == "java":
            configure_jvm(cfg.jvm_heap_gb)

        from msmarco_ir.runs import PAD_PID
        from msmarco_ir.service import SearchService

        service = SearchService(cfg)
        send({"ready": True, "systems": service.available_systems(),
              "hybrids": {k: list(v) for k, v in cfg.hybrids.items()},
              "rrf_k": cfg.rrf_k, "retrieve_k": cfg.retrieve_k})
    except Exception as e:  # noqa: BLE001
        send({"ready": False, "error": f"{type(e).__name__}: {e}"})
        return

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req["op"] == "run":  # ranked (pid, score) list for one base system
                run = service._run(req["system"], req["query"], int(req["depth"]))
                items = [[int(p), float(sc)] for p, sc in zip(run.pids[0], run.scores[0]) if p != PAD_PID]
                send({"items": items})
            elif req["op"] == "text":  # passage text for pids (java worker only)
                send({"texts": [service.passage_text(int(p)) for p in req["pids"]]})
            else:
                send({"error": f"unknown op {req['op']}"})
        except Exception as e:  # noqa: BLE001
            send({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__" and "--worker" in sys.argv:
    _worker_main(sys.argv[1:])
    sys.exit(0)

import html
import re
import subprocess
import threading
import time

import pandas as pd
import streamlit as st

st.set_page_config(page_title="MS MARCO Passage Search", layout="wide")

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "is", "are", "was",
    "were", "be", "by", "with", "what", "how", "who", "when", "where", "why", "which", "do",
    "does", "did", "it", "this", "that", "as", "from",
}

EXAMPLE_QUERIES = [
    "what is the capital of france",
    "symptoms of vitamin d deficiency",
    "how does a neural network learn",
    "average salary of a software engineer",
    "who invented the telephone",
    "how long to boil an egg",
]

CSS = """
<style>
.hit {border:1px solid rgba(128,128,128,.35); border-radius:12px; padding:14px 18px;
      margin-bottom:12px; background:rgba(128,128,128,.06);}
.hit-head {display:flex; gap:10px; align-items:center; margin-bottom:8px; flex-wrap:wrap;}
.badge {font-size:.78rem; padding:2px 10px; border-radius:999px; font-weight:600;}
.b-rank {background:#4f46e5; color:#fff;}
.b-score {background:rgba(16,185,129,.18); color:#059669;}
.b-pid {background:rgba(128,128,128,.2);}
.hit-text {font-size:.97rem; line-height:1.55;}
.hit-text mark {background:#fde68a; color:#111; padding:0 2px; border-radius:3px;}
.scorebar {height:4px; border-radius:2px; background:rgba(128,128,128,.2); margin-top:10px;}
.scorebar > div {height:4px; border-radius:2px; background:linear-gradient(90deg,#6366f1,#10b981);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Backend loading
# --------------------------------------------------------------------------- #
_LAUNCHER = (
    "import os, signal, sys;"
    "signal.pthread_sigmask(signal.SIG_SETMASK, []);"
    "[signal.signal(s, signal.SIG_DFL) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)];"
    "os.execv(sys.argv[1], sys.argv[1:])"
)


class WorkerCrashed(RuntimeError):
    """The worker process died (native crash); the request can be retried on a fresh worker."""


class Worker:
    """One search worker process, spoken to over JSON lines."""

    def __init__(self, role: str, base_args: list[str]):
        self.role = role
        cmd = [sys.executable, "-c", _LAUNCHER, sys.executable, str(Path(__file__).resolve()),
               "--worker", "--role", role, *base_args]
        env = dict(os.environ)
        env["TOKENIZERS_PARALLELISM"] = "false"
        for var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[var] = "1"
        if role == "dense":
            # PyTorch + FAISS: only single-threaded OpenMP was verified stable. KMP_DUPLICATE_LIB_OK
            # made Sentence-Transformer crash in testing, so make sure it is NOT set here.
            env.pop("KMP_DUPLICATE_LIB_OK", None)
            env.pop("KMP_INIT_AT_FORK", None)
        else:
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            env["KMP_INIT_AT_FORK"] = "FALSE"

        self.log_path = ROOT / f".worker_{role}.log"
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                                     text=True, bufsize=1, cwd=str(ROOT), env=env, start_new_session=True)
        self.lock = threading.Lock()
        first = self.proc.stdout.readline()
        if not first:
            code = self.proc.wait()
            raise RuntimeError(f"The {role} worker exited during startup (exit code {code}).\n{self._log_tail()}")
        self.info = json.loads(first)
        if not self.info.get("ready"):
            raise RuntimeError(self.info.get("error", f"{role} worker failed to start"))

    def _log_tail(self, n: int = 12) -> str:
        try:
            lines = [l for l in self.log_path.read_text(errors="replace").splitlines() if l.strip()]
            return f"Last lines of {self.log_path.name}:\n" + "\n".join(lines[-n:])
        except Exception:  # noqa: BLE001
            return ""

    def alive(self) -> bool:
        return self.proc.poll() is None

    def call(self, req: dict) -> dict:
        with self.lock:
            if not self.alive():
                raise WorkerCrashed(f"The {self.role} worker is not running.")
            try:
                self.proc.stdin.write(json.dumps(req) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                raise WorkerCrashed(f"The {self.role} worker closed its pipe.\n{self._log_tail()}")
            line = self.proc.stdout.readline()
            if not line:
                code = self.proc.wait()
                raise WorkerCrashed(f"The {self.role} worker crashed (exit code {code}).\n{self._log_tail()}")
            msg = json.loads(line)
            if "error" in msg:
                raise RuntimeError(msg["error"])
            return msg


class Backend:
    """Routes queries to the right worker, caches runs, retries after native crashes,
    and fuses hybrid (RRF) runs."""

    ROLE_OF = {"bm25": "java", "tfidf": "tfidf"}  # every other base system is dense

    def __init__(self, artifacts_dir: str, device: str, faiss_device: str, jvm_heap_gb: int):
        self.base_args = ["--artifacts-dir", artifacts_dir, "--device", device,
                          "--faiss-device", faiss_device, "--jvm-heap-gb", str(jvm_heap_gb)]
        self._locks = {r: threading.Lock() for r in ("java", "tfidf", "dense")}
        java = Worker("java", self.base_args)
        self.workers: dict[str, Worker | None] = {"java": java, "tfidf": None, "dense": None}
        info = java.info
        self.systems: dict[str, str] = info["systems"]
        self.hybrids: dict[str, list[str]] = info["hybrids"]
        self.rrf_k: int = info["rrf_k"]
        self.retrieve_k: int = info["retrieve_k"]
        self._run_cache: dict[tuple[str, str, int], list[list]] = {}
        self._text_cache: dict[int, str] = {}

    def alive(self) -> bool:
        w = self.workers["java"]
        return w is not None and w.alive()

    def _role(self, system: str) -> str:
        return self.ROLE_OF.get(system, "dense")

    def _worker(self, role: str) -> Worker:
        with self._locks[role]:  # started lazily; restarted automatically if it died
            w = self.workers.get(role)
            if w is None or not w.alive():
                w = Worker(role, self.base_args)
                self.workers[role] = w
            return w

    def _call(self, system: str, req: dict, tries: int = 3) -> dict:
        """Native OpenMP crashes on macOS are intermittent: restart the worker and retry."""
        last: Exception | None = None
        for _ in range(tries):
            try:
                return self._worker(self._role(system)).call(req)
            except WorkerCrashed as e:
                last = e
        raise RuntimeError(f"Worker crashed {tries} times in a row.\n{last}")

    def _run(self, system: str, query: str, depth: int) -> list[list]:
        for (cached_sys, cached_q, cached_depth), items in list(self._run_cache.items()):
            if cached_sys == system and cached_q == query and cached_depth >= depth:
                return items[:depth]

        items = self._call(system, {"op": "run", "system": system, "query": query, "depth": depth})["items"]
        if len(self._run_cache) > 128:
            self._run_cache.clear()
        self._run_cache[(system, query, depth)] = items
        return items

    def _fetch_texts(self, pids: list[int]) -> list[str]:
        missing = [p for p in pids if p not in self._text_cache]
        if missing:
            fetched = self._call("bm25", {"op": "text", "pids": missing})["texts"]
            for p, txt in zip(missing, fetched):
                self._text_cache[p] = txt
        return [self._text_cache[p] for p in pids]

    def search(self, system: str, query: str, k: int) -> list[dict]:
        if system in self.hybrids:  # Reciprocal Rank Fusion: sum of 1 / (rrf_k + rank)
            depth = min(self.retrieve_k, max(k * 10, 100))
            parts = self.hybrids[system]
            java_parts = [p for p in parts if self._role(p) == "java"]
            other_parts = [p for p in parts if self._role(p) != "java"]

            part_results: dict[str, list[list]] = {}
            if java_parts and other_parts:
                err: list[Exception] = []

                def _run_java():  # BM25 runs in its own process, in parallel with the dense/TF-IDF part
                    try:
                        for p in java_parts:
                            part_results[p] = self._run(p, query, depth)
                    except Exception as e:  # noqa: BLE001
                        err.append(e)

                t = threading.Thread(target=_run_java)
                t.start()
                try:
                    for p in other_parts:
                        part_results[p] = self._run(p, query, depth)
                finally:
                    t.join()
                if err:
                    raise err[0]
            else:
                for p in parts:
                    part_results[p] = self._run(p, query, depth)

            fused: dict[int, float] = {}
            for part in parts:
                for rank, (pid, _score) in enumerate(part_results[part], start=1):
                    fused[pid] = fused.get(pid, 0.0) + 1.0 / (self.rrf_k + rank)
            ranked = sorted(fused.items(), key=lambda t: -t[1])[:k]
        else:
            ranked = [tuple(x) for x in self._run(system, query, k)[:k]]

        if not ranked:
            return []
        texts = self._fetch_texts([int(pid) for pid, _ in ranked])
        return [{"rank": i + 1, "pid": int(pid), "score": float(sc), "text": txt}
                for i, ((pid, sc), txt) in enumerate(zip(ranked, texts))]


@st.cache_resource(show_spinner="Starting search backend and loading indexes (first run only)...")
def _start_backend(artifacts_dir: str, device: str, faiss_device: str, jvm_heap_gb: int) -> Backend:
    return Backend(artifacts_dir, device, faiss_device, jvm_heap_gb)


def get_backend(*args) -> Backend:
    b = _start_backend(*args)
    if not b.alive():
        _start_backend.clear()
        b = _start_backend(*args)
    return b


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def highlight(text: str, query: str) -> str:
    """HTML-escape passage text and wrap query terms in <mark>."""
    safe = html.escape(text)
    terms = {t for t in re.findall(r"\w+", query.lower()) if t not in STOPWORDS and len(t) > 1}
    if not terms:
        return safe
    pattern = re.compile(r"\b(" + "|".join(sorted(map(re.escape, terms), key=len, reverse=True)) + r")\b",
                         re.IGNORECASE)
    return pattern.sub(r"<mark>\1</mark>", safe)


def render_hits(hits: list[dict], query: str, show_bars: bool = True) -> None:
    if not hits:
        st.info("No results returned.")
        return
    scores = [h["score"] for h in hits]
    lo, hi = min(scores), max(scores)
    span = (hi - lo) or 1.0
    for h in hits:
        pct = 100 * (h["score"] - lo) / span if len(hits) > 1 else 100
        pct = max(pct, 4)
        bar = f'<div class="scorebar"><div style="width:{pct:.0f}%"></div></div>' if show_bars else ""
        st.markdown(
            f"""
            <div class="hit">
              <div class="hit-head">
                <span class="badge b-rank">#{h['rank']}</span>
                <span class="badge b-score">score {h['score']:.4f}</span>
                <span class="badge b-pid">pid {h['pid']}</span>
              </div>
              <div class="hit-text">{highlight(str(h['text']), query)}</div>
              {bar}
            </div>
            """,
            unsafe_allow_html=True,
        )


def hits_to_df(hits: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"rank": h["rank"], "passage id": h["pid"], "score": round(float(h["score"]), 4), "passage": h["text"]}
         for h in hits]
    )


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def metric_df_from_json(data) -> pd.DataFrame | None:
    """Best-effort conversion of an unknown summary.json shape into a table."""
    if data is None:
        return None
    try:
        if isinstance(data, dict) and data and all(isinstance(v, dict) for v in data.values()):
            df = pd.DataFrame(data).T
            df.index.name = "system"
            return df.reset_index()
        if isinstance(data, list) and data and all(isinstance(v, dict) for v in data):
            return pd.json_normalize(data)
        if isinstance(data, dict):
            return pd.json_normalize(data)
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("MS MARCO IR")
    st.caption("TF-IDF · BM25 · DPR · Sentence-Transformer · RRF")
    st.divider()
    st.subheader("Backend settings")

    # Auto-detect folders like artifacts/, artifacts-small/, artifacts-2500 containing an indexes/ dir
    _found = sorted(p for p in ROOT.glob("artifacts*") if (p / "indexes").is_dir())
    # Prioritize small / 2500 folders automatically if present
    _preferred = [p for p in _found if "small" in p.name.lower() or "2500" in p.name.lower()]
    _default = str(_preferred[0]) if _preferred else (str(_found[0]) if _found else str(ROOT / "artifacts"))

    if len(_found) > 1:
        _default = st.selectbox(
            "Detected artifact folders",
            [str(p) for p in _found],
            index=_found.index(Path(_default)) if Path(_default) in _found else 0
        )
    artifacts_dir = st.text_input("Artifacts directory", value=_default)
    if not (Path(artifacts_dir) / "indexes").is_dir():
        st.warning(f"No `indexes/` folder inside `{artifacts_dir}`. Check the path.")
    device = st.selectbox("Model device", ["auto", "cpu", "cuda", "mps"], index=0)
    faiss_device = st.selectbox("FAISS device", ["cpu", "gpu"], index=0)
    jvm_heap = st.number_input("JVM heap (GB, 0 = default)", min_value=0, max_value=256, value=0, step=1)
    st.caption("Changing these reloads the indexes in a fresh worker process.")
    if st.button("Restart search backend"):
        _start_backend.clear()
        st.rerun()

# Load backend
service = None
systems: dict[str, str] = {}
load_error = None
try:
    service = get_backend(artifacts_dir, device, faiss_device, int(jvm_heap))
    systems = service.systems
except Exception as e:  # noqa: BLE001
    load_error = e

art = Path(artifacts_dir)

st.title("MS MARCO Passage Retrieval (2,500 Subset)")
st.markdown("Search the indexed MS MARCO corpus with several retrieval systems, compare them side by side, "
            "and inspect the offline evaluation results.")

if load_error is not None:
    st.error(f"Could not load the search backend: `{load_error}`")
    st.markdown(
        "Make sure Java 21 is on PATH, `uv sync` was run, and the indexes exist. "
        "The **Evaluation results** tab still works if `artifacts/eval` exists."
    )
elif "bm25" not in systems:
    st.warning("BM25 index not found. Build it first; it also stores passage text.")

label_to_key = {label: key for key, label in systems.items()}
labels = list(label_to_key)

tab_search, tab_compare, tab_eval, tab_about = st.tabs(
    ["Search", "Compare systems", "Evaluation results", "About"]
)

# --------------------------------------------------------------------------- #
# Tab 1: Search
# --------------------------------------------------------------------------- #
with tab_search:
    if not systems:
        st.info("Backend not available, so searching is disabled.")
    else:
        st.caption("Try an example:")
        cols = st.columns(len(EXAMPLE_QUERIES))
        for i, ex in enumerate(EXAMPLE_QUERIES):
            if cols[i].button(ex, key=f"ex_{i}", width="stretch"):
                st.session_state["q_search"] = ex

        with st.form("search_form"):
            c1, c2, c3 = st.columns([4, 2, 1])
            query = c1.text_input("Query", key="q_search", placeholder="what is the capital of france")
            default_idx = 1 if len(labels) > 1 else 0
            sys_label = c2.selectbox("Retrieval system", labels, index=default_idx)
            k = c3.slider("Top-k", 1, 10, 10)
            submitted = st.form_submit_button("Search", type="primary")

        if submitted:
            q = (query or "").strip()
            if not q:
                st.warning("Please enter a query.")
            else:
                t0 = time.perf_counter()
                with st.spinner(f"Searching with {sys_label}..."):
                    hits = service.search(label_to_key[sys_label], q, int(k))
                st.session_state["search_out"] = {
                    "q": q, "label": sys_label, "hits": hits, "ms": (time.perf_counter() - t0) * 1000,
                }

        out = st.session_state.get("search_out")
        if out:
            m1, m2, m3 = st.columns(3)
            m1.metric("System", out["label"])
            m2.metric("Results", len(out["hits"]))
            m3.metric("Latency", f"{out['ms']:.0f} ms")
            st.markdown(f"**Results for:** _{out['q']}_")
            view = st.radio("View", ["Cards", "Table"], horizontal=True, label_visibility="collapsed")
            if view == "Cards":
                render_hits(out["hits"], out["q"])
            else:
                st.dataframe(hits_to_df(out["hits"]), width="stretch", hide_index=True,
                             column_config={"passage": st.column_config.TextColumn(width="large")})
            st.download_button("Download results (CSV)", hits_to_df(out["hits"]).to_csv(index=False),
                               file_name="search_results.csv", mime="text/csv")

# --------------------------------------------------------------------------- #
# Tab 2: Compare systems
# --------------------------------------------------------------------------- #
with tab_compare:
    if not systems:
        st.info("Backend not available, so comparison is disabled.")
    else:
        with st.form("compare_form"):
            c1, c2 = st.columns([4, 1])
            cq = c1.text_input("Query", placeholder="who invented the telephone", key="q_compare")
            ck = c2.slider("Top-k", 1, 10, 5, key="k_compare")
            chosen = st.multiselect("Systems to compare", labels, default=labels)
            go = st.form_submit_button("Compare", type="primary")

        if go:
            q = (cq or "").strip()
            if not q:
                st.warning("Please enter a query.")
            elif not chosen:
                st.warning("Pick at least one system.")
            else:
                results, timings = {}, {}
                prog = st.progress(0.0, text="Running systems...")
                for i, lab in enumerate(chosen):
                    t0 = time.perf_counter()
                    results[lab] = service.search(label_to_key[lab], q, int(ck))
                    timings[lab] = (time.perf_counter() - t0) * 1000
                    prog.progress((i + 1) / len(chosen), text=f"Finished {lab}")
                prog.empty()
                st.session_state["cmp_out"] = {"q": q, "res": results, "ms": timings, "k": int(ck)}

        cmp_out = st.session_state.get("cmp_out")
        if cmp_out:
            q, res, ms = cmp_out["q"], cmp_out["res"], cmp_out["ms"]
            st.markdown(f"**Query:** _{q}_")

            st.subheader("Side-by-side results")
            cols = st.columns(len(res))
            for col, (lab, hits) in zip(cols, res.items()):
                with col:
                    st.markdown(f"##### {lab}")
                    st.caption(f"{ms[lab]:.0f} ms")
                    for h in hits:
                        with st.container(border=True):
                            st.markdown(f"**#{h['rank']}** · score `{h['score']:.4f}` · pid `{h['pid']}`")
                            st.markdown(highlight(str(h["text"]), q), unsafe_allow_html=True)

            st.subheader("Rank of each passage per system")
            st.caption("Rows are passages that appeared in at least one system's top-k; blank = not retrieved.")
            texts, rank_rows = {}, {}
            for lab, hits in res.items():
                for h in hits:
                    texts[h["pid"]] = h["text"]
                    rank_rows.setdefault(h["pid"], {})[lab] = h["rank"]
            rank_df = pd.DataFrame.from_dict(rank_rows, orient="index").reindex(columns=list(res))
            rank_df.insert(0, "systems agreeing", rank_df.notna().sum(axis=1))
            rank_df.insert(1, "passage", [str(texts[p])[:160] + ("..." if len(str(texts[p])) > 160 else "")
                                          for p in rank_df.index])
            rank_df.index.name = "pid"
            rank_df = rank_df.sort_values("systems agreeing", ascending=False)
            st.dataframe(rank_df, width="stretch",
                         column_config={"passage": st.column_config.TextColumn(width="large")})

            if len(res) > 1:
                st.subheader("Top-k overlap between systems (Jaccard)")
                names = list(res)
                sets = {n: {h["pid"] for h in res[n]} for n in names}
                mat = pd.DataFrame(
                    [[len(sets[a] & sets[b]) / max(len(sets[a] | sets[b]), 1) for b in names] for a in names],
                    index=names, columns=names,
                )
                st.dataframe(mat.style.format("{:.2f}"), width="stretch")

# --------------------------------------------------------------------------- #
# Tab 3: Evaluation results
# --------------------------------------------------------------------------- #
with tab_eval:
    eval_dir = art / "eval"
    if not eval_dir.exists():
        st.info(f"No evaluation output found at `{eval_dir}`. Run evaluation script to produce it.")
    else:
        table = None
        csv_path = eval_dir / "comparison_table.csv"
        if csv_path.exists():
            try:
                table = pd.read_csv(csv_path)
            except Exception:
                table = None
        if table is None:
            table = metric_df_from_json(read_json(eval_dir / "summary.json"))

        st.subheader("System comparison")
        if table is not None and not table.empty:
            num_cols = table.select_dtypes("number").columns.tolist()
            st.dataframe(
                table.style.highlight_max(subset=num_cols, color="#bbf7d0") if num_cols else table,
                width="stretch", hide_index=True,
            )
            st.caption("Best value per metric is highlighted.")
            if num_cols:
                label_col = table.columns[0] if table.columns[0] not in num_cols else None
                chart_df = table.set_index(label_col) if label_col else table
                picked = st.multiselect("Metrics to chart", num_cols, default=num_cols[: min(4, len(num_cols))])
                if picked:
                    st.bar_chart(chart_df[picked])
            st.download_button("Download table (CSV)", table.to_csv(index=False),
                               file_name="comparison_table.csv", mime="text/csv")
        else:
            md = eval_dir / "comparison_table.md"
            if md.exists():
                st.markdown(md.read_text())
            else:
                st.info("No comparison table or summary.json found.")

        fig = art / "figures" / "precision_recall.png"
        if fig.exists():
            st.subheader("Precision-recall")
            st.image(str(fig), width="stretch")

        per_sys = sorted(p for p in eval_dir.glob("*.json") if p.name != "summary.json")
        if per_sys:
            st.subheader("Per-system details")
            for p in per_sys:
                with st.expander(p.stem):
                    data = read_json(p)
                    if data is None:
                        st.write("Could not parse this file.")
                    else:
                        st.json(data, expanded=1)

    meta = read_json(art / "prepared" / "prepare_meta.json")
    if meta is not None:
        st.subheader("Data preparation stats")
        st.caption("Duplicate / consistency statistics from corpus preparation.")
        st.json(meta, expanded=1)

    log = art / "logs" / "run.log"
    if log.exists():
        with st.expander("Last 100 lines of run.log"):
            st.code("\n".join(log.read_text(errors="replace").splitlines()[-100:]), language="text")

# --------------------------------------------------------------------------- #
# Tab 4: About
# --------------------------------------------------------------------------- #
with tab_about:
    st.markdown(
        """
### Systems
| System | Idea |
|---|---|
| **TF-IDF** | Sparse lexical retrieval weighted by term frequency × inverse document frequency. |
| **BM25** | Probabilistic lexical ranking with term-frequency saturation and length normalisation (Pyserini/Anserini). |
| **DPR** | Dense Passage Retrieval: bi-encoder embeddings searched with FAISS. |
| **Sentence-Transformer** | Dense bi-encoder embeddings, also searched with FAISS. |
| **RRF** | Reciprocal Rank Fusion: combines ranked lists from several systems using `Σ 1 / (k + rank)`. |

### How to use this UI
- **Search**: one query, one system, ranked passages with query terms highlighted.
- **Compare systems**: the same query on many systems, side by side, with rank agreement and overlap.
- **Evaluation results**: metrics, the precision-recall plot and stats generated by the pipeline.
        """
    )
    if systems:
        st.markdown("**Systems available in this session:** " + ", ".join(f"`{l}`" for l in labels))