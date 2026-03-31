from app import run_batch_llm_judge_evaluation


def analyze_evaluation_results(results):
    return {
        "all_cases": results,
        "num_all_cases": len(results),
    }


def print_priority_cases(analysis):
    all_cases = analysis.get("all_cases", [])

    review_cases = []
    for item in all_cases:
        score_str = item.get("judge_score", "")
        score_val = float(score_str.split("SCORE:", 1)[1].split()[0])
        if score_val <= 3:
            review_cases.append(item)

    print("\nResumo da análise:\n")
    print(f"Casos totais: {analysis.get('num_all_cases', 0)}")
    print(f"Casos para revisão manual (score <= 3): {len(review_cases)}")

    print("\nCasos prioritários para revisão manual:\n")

    for item in review_cases:
        print("Pergunta:", item.get("question"))
        print("Avaliação do juiz:", item.get("judge_score"))



if __name__ == "__main__":
    batch_result = run_batch_llm_judge_evaluation()
    analysis = analyze_evaluation_results(batch_result["results"])

    print_priority_cases(analysis)
