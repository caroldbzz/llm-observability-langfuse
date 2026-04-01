import json

from app import (
    MAX_REVIEW_CASES_TO_PRINT,
    N_EXAMPLES,
    calculate_avg_score,
    calculate_category_summary,
    calculate_severity_counts,
    run_batch_llm_judge_pipeline,
)


def run_final_demo():
    batch_result = run_batch_llm_judge_pipeline(N_EXAMPLES)
    summary = batch_result["summary"]
    results = batch_result["results"]
    review_cases = batch_result["review_cases"]

    print("\n=== RESUMO FINAL ===\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== INTERPRETAÇÃO ===\n")

    avg_score = calculate_avg_score(results)
    num_examples = summary["num_examples"]
    num_critical, num_review = calculate_severity_counts(results)
    total_input_tokens = summary["total_input_tokens"]
    total_output_tokens = summary["total_output_tokens"]
    num_fallback_applied = summary["num_fallback_applied"]

    print(f"- Score médio: {avg_score}")
    print(f"- Total de exemplos: {num_examples}")
    print(f"- Casos para revisão: {num_review}")
    print(f"- Casos críticos: {num_critical}")
    print(f"- Fallback aplicado: {num_fallback_applied}")
    print(f"- Tokens de entrada: {total_input_tokens}")
    print(f"- Tokens de saída: {total_output_tokens}")

    if avg_score < 3:
        print("\nAtenção: a qualidade média está baixa e indica necessidade de revisão do sistema.")
    else:
        print("\nA qualidade média geral parece aceitável, mas ainda deve ser analisada em conjunto com os casos críticos.")

    if num_critical > 0:
        print("Existem casos críticos que precisam de análise manual ou ajuste do comportamento da aplicação.")
    else:
        print("Nenhum caso crítico foi identificado neste conjunto analisado.")

    top_categories = calculate_category_summary(results)[:3]
    if top_categories:
        print("\nCategorias mais frequentes no lote:")
        for cat in top_categories:
            print(f"- {cat['category']} ({cat['count']} caso(s))")

    print(f"\nCasos totais: {len(results)}")
    print(f"Fallback aplicado: {num_fallback_applied}")
    print(f"Casos para revisão manual (final_severity != ok): {len(review_cases)}")
    print(f"Casos prioritários para revisão manual (até {MAX_REVIEW_CASES_TO_PRINT}):")

    for item in review_cases[:MAX_REVIEW_CASES_TO_PRINT]:
        print("Pergunta:", item["question"])
        print("Score inicial:", item["initial_score_value"])
        print("Score final:", item["final_score_value"])
        print("Fallback aplicado:", item["used_fallback"])
        print("Status mitigação:", item["mitigation_status"])
        print("Severidade final:", item["final_severity"])
        print("-" * 50)


if __name__ == "__main__":
    run_final_demo()
