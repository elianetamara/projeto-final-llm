# eval/eval_ragas.py
from __future__ import annotations
import os, json, time, argparse, statistics, pathlib
from typing import List, Dict, Any

from ragas import RunConfig, evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.metrics import FaithfulnesswithHHEM

import pandas as pd
from datasets import Dataset

from src.pipeline import run_pipeline
from src.retriever import PDFIndexerRetriever

DEF_SIM = float(os.getenv("SIM_THRESHOLD", "0.50"))

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Aceita JSONL (1 obj/linha) ou um array JSON. Ignora linhas vazias/comentários."""
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
        if not s or s.startswith("#"):
            continue
        try:
            rows.append(json.loads(s))
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON inválido na linha {ln}: {s[:80]} ...") from e
    return rows

def _write_report(args, agg, latencies, cpu_usages, ram_usages, df_all, outdir, llm_model, run_config):
    import os

    mean_latency = statistics.mean(latencies) if latencies else 0.0
    p95_latency = (
        statistics.quantiles(latencies, n=20)[18]
        if len(latencies) >= 20 else (max(latencies) if latencies else 0.0)
    )
    mean_cpu = statistics.mean(cpu_usages) if cpu_usages else None
    mean_ram = statistics.mean(ram_usages) if ram_usages else None

    no_base_ratio = float((df_all["answer"].astype(str).str.contains("NÃO ENCONTREI BASE")).mean())

    raw_path = os.path.join(outdir, "eval_raw.parquet")
    pd.DataFrame({
        "id": df_all.get("id"),
        "question": df_all.get("question"),
        "answer": df_all.get("answer"),
        "contexts_sample": df_all.get("contexts").apply(lambda x: x[:3] if isinstance(x, list) else x),
        "ground_truths": df_all.get("ground_truths"),
        "tags": df_all.get("tags"),
    }).to_parquet(raw_path, index=False)

    DISPLAY = {
        "faithfulness": "Faithfulness",
        "faithfulness_hhem": "Faithfulness (HHEM)",
        "faithfulness_with_hhem": "Faithfulness (HHEM)",
        "answer_relevancy": "Answer Relevancy",
    }

    lines = ["## Métricas RAGAS"]
    if agg:
        for key, val in agg.items():
            if val == val:  # não-NaN
                label = DISPLAY.get(key, key.replace("_", " ").title())
                lines.append(f"- {label}: **{val:.3f}**")
    else:
        lines.append("- (sem itens elegíveis ou métricas indisponíveis)")

    md = f"""# Relatório de Avaliação RAG

**Modo:** `{args.mode}`  
**K (retrieve):** {args.k}  
**LLM juiz:** `{llm_model}`  

{chr(10).join(lines)}

## Latência & Footprint (todas as perguntas)
- Latência média (s): **{mean_latency:.3f}**
- Latência p95 (s): **{p95_latency:.3f}**
- CPU média (%): **{('%.1f' % mean_cpu) if mean_cpu is not None else 'n/a'}**
- RAM média (MB): **{('%.1f' % mean_ram) if mean_ram is not None else 'n/a'}**

## Robustez
- % respostas “NÃO ENCONTREI BASE” (todas): **{no_base_ratio*100:.1f}%**
"""
    out_md = os.path.join(outdir, "eval_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print("\n==> Relatório salvo em:", out_md)
    print("==> Resultados brutos em:", raw_path)

def _extract_scores(result_obj):
    """
    Extrai métricas do objeto retornado por ragas.evaluate() (versões novas/antigas).
    Se vierem listas/arrays, usa a média.
    """
    import numpy as np

    def _to_float_mean(x):
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if hasattr(x, "item"): 
            try:
                return float(x.item())
            except Exception:
                pass
        if isinstance(x, (list, tuple)):
            vals = []
            for i in x:
                if isinstance(i, (int, float)):
                    vals.append(float(i))
                elif hasattr(i, "item"):
                    try:
                        vals.append(float(i.item()))
                    except Exception:
                        pass
                else:
                    try:
                        vals.append(float(str(i)))
                    except Exception:
                        continue
            if vals:
                return float(np.mean(vals))
            return None
        try:
            return float(str(x))
        except Exception:
            return None

    try:
        df_scores = result_obj.to_pandas()
        if isinstance(df_scores, pd.DataFrame):
            metric_col = "metric" if "metric" in df_scores.columns else None
            value_col = "score" if "score" in df_scores.columns else ("value" if "value" in df_scores.columns else None)
            if metric_col and value_col:
                out = {}
                for _, row in df_scores.iterrows():
                    name = str(row[metric_col])
                    val = _to_float_mean(row[value_col])
                    if val is not None:
                        out[name] = val
                if out:
                    return out
    except Exception:
        pass

    d = getattr(result_obj, "_scores_dict", None)
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            name = getattr(k, "name", str(k))
            val = _to_float_mean(v)
            if val is not None:
                out[name] = val
        if out:
            return out

    try:
        out = {}
        for m in result_obj:
            name = m.get("metric")
            score = _to_float_mean(m.get("score"))
            if name and score is not None:
                out[str(name)] = score
        if out:
            return out
    except Exception:
        pass

    d2 = getattr(result_obj, "scores", None)
    if isinstance(d2, dict):
        out = {}
        for k, v in d2.items():
            val = _to_float_mean(v)
            if val is not None:
                out[str(k)] = val
        if out:
            return out

    return {}

def _truncate_for_hhem(contexts: List[str]) -> List[str]:
    """Trunca contextos para HHEM."""
    try:
        max_chars = int(os.getenv("HHEM_CTX_CHARS", "400"))
    except Exception:
        max_chars = 600
    try:
        max_k = int(os.getenv("HHEM_CTX_K", "2"))
    except Exception:
        max_k = 2

    trimmed = []
    for c in contexts[:max_k]:
        if not isinstance(c, str):
            c = str(c)
        trimmed.append(c[:max_chars])
    return trimmed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/eval_questions.jsonl")
    ap.add_argument("--mode", default="chat", choices=["chat","detector"])
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max_ctx_chars", type=int, default=1000, help="truncate cada contexto para no máximo N caracteres")
    ap.add_argument("--max_ctx_k", type=int, default=3, help="quantos contextos por item passar ao juiz")
    args = ap.parse_args()

    pathlib.Path(args.outdir).mkdir(parents=True, exist_ok=True)

    items = load_jsonl(args.data)

    retr = PDFIndexerRetriever()
    try:
        retr.ensure_ready()
    except RuntimeError:
        retr.build_index_from_folder("data/pdfs")
        retr.ensure_ready()

    records = []
    latencies: List[float] = []
    cpu_usages: List[float] = []
    ram_usages: List[float] = []

    try:
        import psutil
        have_psutil = True
        proc = psutil.Process(os.getpid())
    except Exception:
        have_psutil = False
        proc = None  # type: ignore

    for ex in items:
        qid = ex.get("id") or f"item-{len(records)+1}"
        q = ex["question"]
        expected = ex.get("expected_behavior", "informative")

        ctx_hits = retr.retrieve(q, k=args.k)
        contexts_full = [h.get("text", "") for h in ctx_hits]
        contexts = [c[:args.max_ctx_chars] for c in contexts_full[:args.max_ctx_k]]

        t0 = time.perf_counter()
        ans = run_pipeline(mode=args.mode, user_input=q, history=[])
        dt = time.perf_counter() - t0
        latencies.append(dt)

        if have_psutil and proc is not None:
            cpu = psutil.cpu_percent(interval=0.05)
            mem = proc.memory_info().rss / (1024**2)
            cpu_usages.append(cpu); ram_usages.append(mem)

        gts: List[str] = ex.get("gt_passages", []) 
        eval_mask = not (expected == "no_base")     

        records.append({
            "id": qid,
            "question": q,
            "contexts": contexts,
            "answer": ans,
            "ground_truths": gts,
            "tags": ex.get("tags", []),
            "evaluate": eval_mask
        })

    df = pd.DataFrame(records)

    if "ground_truths" not in df.columns:
        df["ground_truths"] = [[] for _ in range(len(df))]
    else:
        df["ground_truths"] = df["ground_truths"].apply(lambda x: x if isinstance(x, list) else [])

    if "evaluate" not in df.columns:
        df["evaluate"] = True

    df_eval = df[df["evaluate"]].copy()
    if df_eval.empty:
        print("[RAGAS] Nenhum item elegível (evaluate==True).")
        dummy_rc = RunConfig(timeout=600, max_workers=1, max_retries=5, max_wait=30, log_tenacity=True)
        _write_report(args, {}, latencies, cpu_usages, ram_usages, df, args.outdir, os.getenv("OLLAMA_MODEL", "phi3:mini"), dummy_rc)
        return

    ragas_ds_base = Dataset.from_pandas(df_eval[["question","contexts","answer","ground_truths"]])

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except Exception:
        from langchain_community.embeddings import HuggingFaceEmbeddings 
    from langchain_community.chat_models import ChatOllama

    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"[RAGAS] Itens avaliados: {len(df_eval)}")

    run_config = RunConfig(
        timeout=600,
        max_workers=8,      
        max_retries=5,
        max_wait=30,
        log_tenacity=True
    )

    agg: Dict[str, float] = {}

    metrics_to_run = [
        FaithfulnesswithHHEM(),  
        answer_relevancy,
    ]

    for metric in metrics_to_run:
        try:
            metric_name = getattr(metric, "name", str(metric))
            print(f"[RAGAS] Avaliando {metric_name} ...")

            if isinstance(metric, FaithfulnesswithHHEM):
                df_hhem = df_eval.copy()
                df_hhem["contexts"] = df_hhem["contexts"].apply(_truncate_for_hhem)
                ragas_ds = Dataset.from_pandas(df_hhem[["question","contexts","answer","ground_truths"]])
            else:
                ragas_ds = ragas_ds_base

            res = evaluate(
                ragas_ds,
                metrics=[metric],
                llm=llm,
                embeddings=emb,
                run_config=run_config
            )
            scores = _extract_scores(res)
            if metric_name in scores:
                agg[metric_name] = scores[metric_name]
            else:
                for k in ("faithfulness_hhem", "faithfulness_with_hhem", "faithfulness", "answer_relevancy"):
                    if k in scores:
                        agg[k] = scores[k]
                        break
                else:
                    print(f"[WARN] métrica {metric_name} não retornou score. Scores vistos: {list(scores.keys())}")

        except Exception as e:
            print(f"[WARN] métrica {getattr(metric, 'name', str(metric))} falhou: {e}")

    _write_report(args, agg, latencies, cpu_usages, ram_usages, df, args.outdir, OLLAMA_MODEL, run_config)

if __name__ == "__main__":
    main()
