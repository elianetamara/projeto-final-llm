import csv
import time
import statistics
from pathlib import Path

from src.retriever import Retriever
from src.agents.answer_agent import generate_answer
from src.agents.selfcheck_agent import check_coverage
from src.agents.safety_agent import safety_filter, add_disclaimer

REPORT_PATH = Path("eval/report.md")
CSV_PATH = Path("eval/tests_questions.csv")

def load_tests():
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def evaluate():
    retr = Retriever()
    tests = load_tests()
    results = []

    latencies = []
    precision_list, recall_list, faithfulness_list = [], [], []

    for t in tests:
        q = t["question"]
        expected_src = t["expected_source"]

        start = time.time()
        hits = retr.retrieve(q, k=5)
        answer = generate_answer(q, hits)
        problems = check_coverage(answer, hits)
        ok, filtered = safety_filter(answer)
        if not ok:
            final_answer = filtered
        else:
            final_answer = add_disclaimer(filtered)
        latency = time.time() - start
        latencies.append(latency)

        # métricas simples
        found_sources = [h["meta"].get("source") for h in hits if h["meta"].get("source")]
        # Precision: fontes retornadas que batem com esperado
        true_pos = sum(1 for s in found_sources if expected_src in s)
        precision = true_pos / len(found_sources) if found_sources else 0.0
        recall = 1.0 if true_pos > 0 else 0.0
        faithfulness = 1.0 if not problems else 0.0

        precision_list.append(precision)
        recall_list.append(recall)
        faithfulness_list.append(faithfulness)

        results.append({
            "question": q,
            "answer": final_answer[:500],
            "expected": expected_src,
            "found_sources": found_sources,
            "precision": precision,
            "recall": recall,
            "faithfulness": faithfulness,
            "latency": latency,
        })

    # métricas agregadas
    avg_prec = statistics.mean(precision_list)
    avg_recall = statistics.mean(recall_list)
    avg_faith = statistics.mean(faithfulness_list)
    avg_latency = statistics.mean(latencies)

    # gerar relatório
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Relatório de Avaliação — Assistente Anti-Fraude PIX\n\n")
        f.write("## Métricas agregadas\n")
        f.write(f"- Precision média: {avg_prec:.2f}\n")
        f.write(f"- Recall médio: {avg_recall:.2f}\n")
        f.write(f"- Faithfulness médio: {avg_faith:.2f}\n")
        f.write(f"- Latência média: {avg_latency:.2f} s\n\n")

        f.write("## Resultados por pergunta\n")
        for r in results:
            f.write(f"### Pergunta: {r['question']}\n")
            f.write(f"- Expected source: {r['expected']}\n")
            f.write(f"- Found sources: {r['found_sources']}\n")
            f.write(f"- Precision: {r['precision']:.2f}, Recall: {r['recall']:.2f}, Faithfulness: {r['faithfulness']}\n")
            f.write(f"- Latência: {r['latency']:.2f} s\n")
            f.write(f"- Resposta (trecho): {r['answer']}\n\n")

    print("✅ Avaliação concluída. Relatório salvo em eval/report.md")

if __name__ == "__main__":
    evaluate()
