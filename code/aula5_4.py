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

    print(f"- Score médio: {avg_score}")
    print(f"- Total de exemplos: {num_examples}")
    print(f"- Casos críticos: {num_critical}")

    # 🔹 Interpretação simples (valor da aula 5.4)
    if avg_score < 100:
        print("\n⚠️ Atenção: respostas podem estar muito curtas ou pouco informativas.")

    if num_critical > 0:
        print("⚠️ Existem casos críticos que precisam de análise manual.")

    worst_categories = summary.get("category_summary", [])[:3]

    if worst_categories:
        print("\nCategorias com pior desempenho:")
        for cat in worst_categories:
            print(f"- {cat['category']} (score médio: {cat['avg_score']})")

        print("\n💡 Sugestão:")
        print("→ Avaliar prompts específicos para essas categorias")
        print("→ Criar testes direcionados")
        print("→ Monitorar essas categorias em produção")


if __name__ == "__main__":
    run_final_demo()