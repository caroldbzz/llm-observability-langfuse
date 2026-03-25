import json
from collections import Counter

try:
    from aula5_3 import run_monitoring_analysis
except ModuleNotFoundError:
    from docs.aula5_3 import run_monitoring_analysis


def calculate_avg_score(results: list[dict]):
    score_values = [
        item.get("final_score_value")
        for item in results
        if item.get("final_score_value") is not None
    ]
    if not score_values:
        return None
    return round(sum(score_values) / len(score_values), 2)


def calculate_severity_counts(results: list[dict]):
    num_critical = sum(1 for item in results if item.get("final_severity") == "critical")
    num_review = sum(
        1 for item in results if item.get("final_severity") in {"critical", "review"}
    )
    return num_critical, num_review


def calculate_category_summary(results: list[dict]):
    category_counter = Counter((item.get("category") or "unknown") for item in results)
    return [
        {"category": category, "count": count}
        for category, count in category_counter.most_common()
    ]


def run_final_demo():
    result = run_monitoring_analysis()
    summary = result["summary"]
    results = result["results"]

    print("\n=== RESUMO FINAL ===\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== INTERPRETAÇÃO ===\n")

    avg_score = calculate_avg_score(results)
    num_examples = summary.get("num_examples")
    num_critical, num_review = calculate_severity_counts(results)
    avg_latency = summary.get("avg_latency_ms")
    total_input_tokens = summary.get("total_input_tokens")
    total_output_tokens = summary.get("total_output_tokens")
    num_fallback_applied = summary.get(
        "num_fallback_applied",
        sum(1 for item in results if item.get("used_fallback")),
    )

    print(f"- Score médio: {avg_score}")
    print(f"- Total de exemplos: {num_examples}")
    print(f"- Casos para revisão: {num_review}")
    print(f"- Casos críticos: {num_critical}")
    print(f"- Fallback aplicado: {num_fallback_applied}")
    print(f"- Latência média (ms): {avg_latency}")
    print(f"- Tokens de entrada: {total_input_tokens}")
    print(f"- Tokens de saída: {total_output_tokens}")

    if avg_score is not None and avg_score < 3:
        print("\nAtenção: a qualidade média está baixa e indica necessidade de revisão do sistema.")
    elif avg_score is not None:
        print("\nA qualidade média geral parece aceitável, mas ainda deve ser analisada em conjunto com os casos críticos.")

    if num_critical and num_critical > 0:
        print("Existem casos críticos que precisam de análise manual ou ajuste do comportamento da aplicação.")
    else:
        print("Nenhum caso crítico foi identificado neste conjunto analisado.")

    top_categories = calculate_category_summary(results)[:3]

    if top_categories:
        print("\nCategorias mais frequentes no lote:")
        for cat in top_categories:
            print(f"- {cat['category']} ({cat['count']} caso(s))")


if __name__ == "__main__":
    run_final_demo()
