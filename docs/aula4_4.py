from aula4_3 import run_batch_llm_judge_evaluation


def analyze_evaluation_results(results: list[dict]) -> dict:
    review_cases = [item for item in results if item.get("judge_score")]

    return {
        "review_cases": review_cases,
        "num_review_cases": len(review_cases),
    }


def print_priority_cases(analysis: dict):
    print("\nResumo da análise:\n")
    print(f"Casos para revisão: {analysis['num_review_cases']}")

    print("\nCasos para revisão manual:\n")

    for item in analysis["review_cases"]:
        print("Pergunta:", item.get("question"))
        print("Avaliação do juiz:", item.get("judge_score"))
        print("-" * 50)



if __name__ == "__main__":
    batch_result = run_batch_llm_judge_evaluation(N_EXAMPLES=5)
    analysis = analyze_evaluation_results(batch_result["results"])

    print_priority_cases(analysis)
