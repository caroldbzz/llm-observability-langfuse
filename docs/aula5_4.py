import json
from aula5_3 import run_monitoring_analysis


def run_final_demo():
    result = run_monitoring_analysis()
    summary = result["summary"]

    print("\n=== RESUMO FINAL ===\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== INTERPRETAÇÃO ===\n")

    avg_score = summary.get("avg_score")
    num_examples = summary.get("num_examples")
    num_critical = summary.get("num_critical_cases")
    avg_latency = summary.get("avg_latency_ms")
    num_fallback = summary.get("num_fallback_applied")

    print(f"- Score médio: {avg_score}")
    print(f"- Total de exemplos: {num_examples}")
    print(f"- Casos críticos: {num_critical}")
    print(f"- Latência média (ms): {avg_latency}")
    print(f"- Fallback aplicado: {num_fallback} vez(es)")

    if avg_score is not None and avg_score < 3:
        print("\nAtenção: a qualidade média está baixa e indica necessidade de revisão do sistema.")
    elif avg_score is not None:
        print("\nA qualidade média geral parece aceitável, mas ainda deve ser analisada em conjunto com os casos críticos.")

    if num_critical and num_critical > 0:
        print("Existem casos críticos que precisam de análise manual ou ajuste do comportamento da aplicação.")
    else:
        print("Nenhum caso crítico foi identificado neste conjunto analisado.")

    if num_fallback and num_fallback > 0:
        print("O fallback foi acionado em parte dos casos, o que indica que a mitigação automática está atuando, mas também sinaliza oportunidades de melhoria no prompt principal.")

    worst_categories = summary.get("category_summary", [])[:3]

    if worst_categories:
        print("\nCategorias com pior desempenho:")
        for cat in worst_categories:
            print(f"- {cat['category']} (score médio: {cat['avg_score']})")


if __name__ == "__main__":
    run_final_demo()