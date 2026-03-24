from typing import List, Dict


def analyze_evaluation_results(results: List[Dict]) -> Dict:
    low_score_cases = []
    medium_score_cases = []
    high_score_cases = []

    for item in results:
        score = item.get("judge_score")

        if score is None:
            continue

        if score <= 2:
            low_score_cases.append(item)
        elif score == 3:
            medium_score_cases.append(item)
        else:
            high_score_cases.append(item)

    return {
        "low_score_cases": low_score_cases,
        "medium_score_cases": medium_score_cases,
        "high_score_cases": high_score_cases,
        "num_low_score_cases": len(low_score_cases),
        "num_medium_score_cases": len(medium_score_cases),
        "num_high_score_cases": len(high_score_cases),
    }


def print_priority_cases(analysis: Dict):
    print("\nResumo da análise:\n")
    print(f"Casos críticos (score <= 2): {analysis['num_low_score_cases']}")
    print(f"Casos medianos (score = 3): {analysis['num_medium_score_cases']}")
    print(f"Casos bons (score >= 4): {analysis['num_high_score_cases']}")

    print("\nCasos críticos para revisão manual:\n")

    for item in analysis["low_score_cases"]:
        print("Pergunta:", item.get("question"))
        print("Score:", item.get("judge_score"))
        print("Justificativa:", item.get("judge_reason"))
        print("-" * 50)



if __name__ == "__main__":
    mock_results = [
        {
            "question": "Como cancelar minha assinatura?",
            "judge_score": 2,
            "judge_reason": "Resposta incompleta e pouco clara.",
        },
        {
            "question": "Como atualizar meu cartão?",
            "judge_score": 4,
            "judge_reason": "Resposta clara e útil.",
        },
        {
            "question": "Meu pagamento falhou, o que faço?",
            "judge_score": 3,
            "judge_reason": "Resposta razoável, mas poderia ser mais detalhada.",
        },
        {
            "question": "Quero falar com suporte",
            "judge_score": 1,
            "judge_reason": "Resposta inadequada e não resolveu o problema.",
        },
    ]

    analysis = analyze_evaluation_results(mock_results)

    print_priority_cases(analysis)