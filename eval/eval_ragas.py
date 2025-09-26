# src/eval_ragas.py
from __future__ import annotations
import os, json, time, argparse, statistics, pathlib as _pl
from typing import List, Dict, Any
import uuid
from datetime import datetime

import pandas as pd
from datasets import Dataset
from ragas import RunConfig, evaluate
from ragas.metrics import answer_relevancy
from ragas.metrics import FaithfulnesswithHHEM

# --- bootstrap ---
import sys
ROOT = _pl.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ===== sua stack =====
from src.pipeline import get_app, get_retriever  # get_retriever mantido por compatibilidade

# ===============================
# Utilitários básicos
# ===============================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    import io
    with io.open(path, "r", encoding="utf-8-sig") as f:
        text = f.read().strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Raiz JSON deve ser lista.")
        return data
    rows: List[Dict[str, Any]] = []
    for ln, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("//"):
            continue
        try:
            rows.append(json.loads(s))
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON inválido na linha {ln}: {s[:120]} ...") from e
    return rows

def _extract_scores(result_obj):
    import numpy as np
    def _to_float_mean(x):
        if x is None: return None
        if isinstance(x, (int, float)): return float(x)
        if hasattr(x, "item"):
            try: return float(x.item())
            except Exception: pass
        if isinstance(x, (list, tuple)):
            vals = []
            for i in x:
                try: vals.append(float(i if isinstance(i,(int,float)) else str(i)))
                except Exception: pass
            return float(np.mean(vals)) if vals else None
        try: return float(str(x))
        except Exception: return None
    try:
        df_scores = result_obj.to_pandas()
        if isinstance(df_scores, pd.DataFrame):
            metric_col = "metric" if "metric" in df_scores.columns else None
            value_col  = "score"  if "score"  in df_scores.columns else ("value" if "value" in df_scores.columns else None)
            if metric_col and value_col:
                out = {}
                for _, row in df_scores.iterrows():
                    name = str(row[metric_col]); val = _to_float_mean(row[value_col])
                    if val is not None: out[name] = val
                if out: return out
    except Exception: pass
    d = getattr(result_obj, "_scores_dict", None)
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            name = getattr(k, "name", str(k))
            val = _to_float_mean(v)
            if val is not None: out[name] = val
        if out: return out
    try:
        out = {}
        for m in result_obj:
            name = m.get("metric"); score = _to_float_mean(m.get("score"))
            if name and score is not None: out[str(name)] = score
        if out: return out
    except Exception: pass
    d2 = getattr(result_obj, "scores", None)
    if isinstance(d2, dict):
        out = {}
        for k, v in d2.items():
            val = _to_float_mean(v)
            if val is not None: out[str(k)] = val
        if out: return out
    return {}

# ===============================
# Truncadores “token-aware” por PALAVRAS (evita 512 tokens)
# ===============================
def _clamp_words(text: str, max_words: int) -> str:
    if not text or max_words <= 0:
        return text or ""
    w = text.split()
    if len(w) <= max_words:
        return text
    return " ".join(w[:max_words])

def _truncate_contexts_for_hhem(contexts: List[str]) -> List[str]:
    """
    Trunca contextos por PALAVRAS para evitar estourar modelos NLI (típico 512 tokens).
    ENV:
      - HHEM_CTX_K (default=2)
      - HHEM_CTX_WORDS (default=120)
    """
    max_k     = int(os.getenv("HHEM_CTX_K", "2"))
    max_words = int(os.getenv("HHEM_CTX_WORDS", "120"))
    out = []
    for c in contexts[:max_k]:
        out.append(_clamp_words(c or "", max_words))
    return out

def _truncate_answer_for_hhem(answer: str) -> str:
    """
    Trunca a RESPOSTA final por PALAVRAS para evitar estourar a janela em NLI.
    ENV:
      - HHEM_ANS_WORDS (default=220)
    """
    max_words = int(os.getenv("HHEM_ANS_WORDS", "220"))
    return _clamp_words(answer or "", max_words)

# ===============================
# LLM logger (intercepta prompts enviados ao juiz)
# ===============================
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import BaseMessage

class LoggingChatOllama(ChatOllama):
    def __init__(self, *args, log_dir: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_dir = log_dir
        self._log_idx = 0
        if self._log_dir:
            _pl.Path(self._log_dir).mkdir(parents=True, exist_ok=True)

    def _dump_msgs(self, messages):
        if not self._log_dir:
            return
        self._log_idx += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = _pl.Path(self._log_dir) / f"llm_call_{ts}_{self._log_idx:04d}.md"
        def _fmt(m):
            if isinstance(m, BaseMessage):
                role = m.type or m.__class__.__name__
                content = (m.content or "").strip()
                return f"### {role.upper()}\n\n{content}\n"
            return f"### MESSAGE\n\n{str(m)}\n"
        try:
            with open(fname, "w", encoding="utf-8") as f:
                f.write("# LLM Judge Call\n\n")
                if isinstance(messages, dict):
                    for k, v in messages.items():
                        f.write(f"## {k}\n\n{str(v)}\n\n")
                elif isinstance(messages, (list, tuple)):
                    for m in messages:
                        f.write(_fmt(m) + "\n")
                else:
                    f.write(_fmt(messages))
        except Exception:
            # logging não deve quebrar a avaliação
            pass

    def invoke(self, input, **kwargs):
        try:
            self._dump_msgs(input)
        except Exception:
            pass
        return super().invoke(input, **kwargs)

    def generate(self, messages, **kwargs):
        try:
            self._dump_msgs(messages)
        except Exception:
            pass
        return super().generate(messages, **kwargs)

def _dump_item_inputs(dump_dir: str, item_id: str, metric: str,
                      question: str, answer: str, contexts: List[str]):
    _pl.Path(dump_dir).mkdir(parents=True, exist_ok=True)
    path = _pl.Path(dump_dir) / f"{item_id}_{metric}.md"
    def _preview(t: str, nlines=6):
        lines = (t or "").splitlines()
        head = "\n".join(lines[:nlines])
        return head + ("\n..." if len(lines) > nlines else "")
    def _wcount(t: str): return len((t or "").split())

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# RAGAS INPUT DUMP — {item_id} ({metric})\n\n")
            f.write(f"**question** ({_wcount(question)} palavras):\n\n{_preview(question)}\n\n")
            f.write(f"**answer (avaliada)** ({_wcount(answer)} palavras):\n\n{_preview(answer)}\n\n")
            f.write(f"**contexts ({len(contexts)})**\n\n")
            for i, c in enumerate(contexts, 1):
                f.write(f"---\n\n### ctx#{i} ({_wcount(c)} palavras)\n\n{_preview(c)}\n\n")
    except Exception:
        pass

# ===============================
# Relatório
# ===============================
def _write_report(args, agg, latencies, cpu_usages, ram_usages, df_all, outdir, llm_model, run_config):
    mean_latency = statistics.mean(latencies) if latencies else 0.0
    p95_latency  = (statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20
                    else (max(latencies) if latencies else 0.0))
    mean_cpu = statistics.mean(cpu_usages) if cpu_usages else None
    mean_ram = statistics.mean(ram_usages) if ram_usages else None

    raw_path = os.path.join(outdir, f"eval_raw_{args.metric}.parquet")
    pd.DataFrame({
        "id": df_all.get("id"),
        "question": df_all.get("question"),
        "answer": df_all.get("answer"),
        "answer_draft": df_all.get("answer_draft"),
        "answer_final": df_all.get("answer_final"),
        "contexts_sample": df_all.get("contexts").apply(lambda x: x[:3] if isinstance(x, list) else x),
        "contexts_meta_sample": df_all.get("contexts_meta").apply(lambda x: x[:3] if isinstance(x, list) else x) if "contexts_meta" in df_all.columns else None,
        "tags": df_all.get("tags"),
    }).to_parquet(raw_path, index=False)

    DISPLAY = {
        "faithfulness_hhem": "Faithfulness (HHEM)",
        "faithfulness_with_hhem": "Faithfulness (HHEM)",
        "answer_relevancy": "Answer Relevancy",
    }
    lines = ["## Métricas RAGAS"]
    if agg:
        for key, val in agg.items():
            if val == val:
                label = DISPLAY.get(key, key.replace("_", " ").title())
                lines.append(f"- {label}: **{val:.3f}**")
    else:
        lines.append("- (sem itens elegíveis ou métricas indisponíveis)")

    md = f"""# Relatório de Avaliação RAG

**Modo:** `{args.mode}`  
**K (retrieve):** {args.k}  
**Métrica:** `{args.metric}`  
**LLM juiz (Ollama):** `{llm_model}`  

{chr(10).join(lines)}

## Latência & Footprint
- Latência média (s): **{mean_latency:.3f}**
- Latência p95 (s): **{p95_latency:.3f}**
- CPU média (%): **{('%.1f' % mean_cpu) if mean_cpu is not None else 'n/a'}**
- RAM média (MB): **{('%.1f' % mean_ram) if mean_ram is not None else 'n/a'}**
"""
    out_md = os.path.join(outdir, f"eval_report_{args.metric}.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print("\n==> Relatório salvo em:", out_md)
    print("==> Resultados brutos em:", raw_path)

# ===============================
# Main
# ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/eval_questions.jsonl")
    ap.add_argument("--mode", default="chat", choices=["chat","detector"])
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--k", type=int, default=8, help="apenas log; contexts reais vêm do LangGraph")
    ap.add_argument("--max_ctx_chars", type=int, default=1000)  # relevancy (opcional)
    ap.add_argument("--max_ctx_k", type=int, default=3)         # relevancy (opcional)
    ap.add_argument("--metric", required=True,
                    choices=["answer_relevancy", "faithfulness_hhem"])
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--dump-debug", action="store_true",
                    help="Salva insumos que o RAGAS/LLM recebem (por item e por chamada ao LLM).")
    ap.add_argument("--dump-dir", default="reports/_debug_prompts",
                    help="Diretório para os dumps de debug.")
    args = ap.parse_args()

    outdir = os.path.join(args.outdir, args.metric)
    _pl.Path(outdir).mkdir(parents=True, exist_ok=True)

    items = load_jsonl(args.data)

    # Reutiliza o MESMO app/langgraph da sua pipeline
    app = get_app()
    _ = get_retriever()  # mantido caso sua app dependa de side-effects

    records, latencies, cpu_usages, ram_usages = [], [], [], []
    try:
        import psutil
        have_psutil = True
        proc = psutil.Process(os.getpid())
    except Exception:
        have_psutil = False
        proc = None  # type: ignore

    # === LLM/Embeddings do juiz (RAGAS) ===
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except Exception:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    base_url = os.getenv("OLLAMA_BASE_URL", "http://150.165.75.163/ollama/").rstrip("/")
    OLLAMA_JUDGE = os.getenv("OLLAMA_JUDGE", "mistral:7b")  # você pediu mistral:7b
    OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))
    OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
    OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))

    # logger de chamadas ao juiz
    dump_llm_dir = None
    if args.dump_debug:
        dump_llm_dir = str(_pl.Path(args.dump_dir) / "llm_calls")

    llm = LoggingChatOllama(
        base_url=base_url,
        model=OLLAMA_JUDGE,
        temperature=OLLAMA_TEMPERATURE,
        num_ctx=OLLAMA_NUM_CTX,
        num_predict=OLLAMA_NUM_PREDICT,
        log_dir=dump_llm_dir,
    )
    # Embeddings multilíngues bons para PT-BR
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    # === Loop ===
    for ex in items:
        qid = ex.get("id") or f"item-{len(records)+1}"
        q   = ex["question"]
        expected = ex.get("expected_behavior", "informative")

        # 1) Invoca o LangGraph e captura EXACTAMENTE o que o modelo viu
        t0 = time.perf_counter()
        out = app.invoke(
            {"question": q, "mode": args.mode, "history": ex.get("history", [])},
            config={"configurable": {"thread_id": f"eval-{qid}"}}
        )
        dt = time.perf_counter() - t0
        latencies.append(dt)

        evidences = out.get("evidences") or []   # o que foi passado ao LLM
        draft     = (out.get("draft") or "").strip()
        final     = (out.get("final") or draft or "NÃO ENCONTREI BASE").strip()

        # 2) Contextos em texto limpo
        contexts_all = [" ".join((e.get("text") or "").split()) for e in evidences if (e.get("text") or "").strip()]

        # 3) Escolha de campos para cada métrica
        if args.metric == "faithfulness_hhem":
            # Faithfulness pelo RAGAS (HHEM): contexts = evidences reais, ambos truncados por palavras
            ctx_for_metric = _truncate_contexts_for_hhem(contexts_all)
            ans_for_metric = _truncate_answer_for_hhem(final)
        else:
            # answer_relevancy: avalia o DRAFT (relevância à pergunta)
            ctx_for_metric = [c[:args.max_ctx_chars] for c in contexts_all[:args.max_ctx_k]]
            ans_for_metric = draft

        if args.dump_debug:
            _dump_item_inputs(
                dump_dir=str(_pl.Path(args.dump_dir) / "items"),
                item_id=str(qid),
                metric=args.metric,
                question=q,
                answer=ans_for_metric,
                contexts=ctx_for_metric
            )

        if args.debug:
            print(f"[DEBUG] {qid}: evidences={len(evidences)} | ctx_eval={len(ctx_for_metric)} "
                  f"| draft_len={len(draft)} | final_len={len(final)} | dt={dt:.3f}s")

        if have_psutil and proc is not None:
            cpu = psutil.cpu_percent(interval=0.05)
            mem = proc.memory_info().rss / (1024**2)
            cpu_usages.append(cpu); ram_usages.append(mem)

        eval_mask = not (expected == "no_base")
        records.append({
            "id": qid,
            "question": q,
            "contexts": ctx_for_metric,
            "contexts_meta": evidences[:len(ctx_for_metric)],
            "answer": ans_for_metric,
            "answer_draft": draft,
            "answer_final": final,
            "tags": ex.get("tags", []),
            "evaluate": eval_mask
        })

    # === DataFrame p/ RAGAS ===
    df = pd.DataFrame(records)
    if "evaluate" not in df.columns:
        df["evaluate"] = True
    df_eval = df[df["evaluate"]].copy()
    if df_eval.empty:
        print("[RAGAS] Nenhum item elegível (evaluate==True).")
        dummy_rc = RunConfig(timeout=600, max_workers=1, max_retries=5, max_wait=30, log_tenacity=True)
        _write_report(args, {}, latencies, cpu_usages, ram_usages, df, outdir, OLLAMA_JUDGE, dummy_rc)
        return

    # Dataset para o RAGAS (NÃO mandamos ground_truth/reference nessas métricas)
    ragas_cols = ["question", "contexts", "answer"]
    ragas_ds = Dataset.from_pandas(df_eval[ragas_cols].reset_index(drop=True))

    print(f"[RAGAS] Itens avaliados: {len(df_eval)}")
    run_config = RunConfig(timeout=600, max_workers=8, max_retries=5, max_wait=30, log_tenacity=True)

    # === Seleção de métrica ===
    if args.metric == "answer_relevancy":
        metric_obj = answer_relevancy
        metric_name = "answer_relevancy"
    elif args.metric == "faithfulness_hhem":
        metric_obj = FaithfulnesswithHHEM()   # fé calculada inteiramente pelo RAGAS
        metric_name = "faithfulness_hhem"
    else:
        raise ValueError(f"Métrica não suportada: {args.metric}")

    # === Avaliação ===
    try:
        print(f"[RAGAS] Avaliando {metric_name} ...")
        res = evaluate(ragas_ds, metrics=[metric_obj], llm=llm, embeddings=emb, run_config=run_config)
        scores = _extract_scores(res)
        agg = {}
        for kcand in (metric_name, "faithfulness_with_hhem", "answer_relevancy"):
            if kcand in scores:
                agg[kcand] = scores[kcand]; break
        if not agg:
            for k, v in scores.items(): agg[k] = v
    except Exception as e:
        print(f"[WARN] métrica {metric_name} falhou: {e}")
        agg = {}

    _write_report(args, agg, latencies, cpu_usages, ram_usages, df, outdir, OLLAMA_JUDGE, run_config)

if __name__ == "__main__":
    main()
