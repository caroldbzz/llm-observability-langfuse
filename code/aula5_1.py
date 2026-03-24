import os
import json
import time
import pandas as pd
from dotenv import load_dotenv

from langfuse import get_client, propagate_attributes
from langfuse.openai import openai

load_dotenv()

langfuse = get_client()
openai.api_key = os.getenv("OPENAI_API_KEY")

OPENAI_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"

DATASET_PATH = "../data/bitext_customer_support.csv"
ANSWER_PROMPT_NAME = "customer-support-assistant"
ANSWER_PROMPT_LABEL = "production"
JUDGE_PROMPT_PATH = "prompts/answer_judge.md"

N_EXAMPLES = 5


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def build_prompt(template: str, variables: dict) -> str:
    prompt = template
    for key, value in variables.items():
        prompt = prompt.replace(f"{{{{{key}}}}}", str(value))
    return prompt


def parse_judge_response(raw_response: str) -> dict:
    try:
        parsed = json.loads(raw_response)
        score = parsed.get("score")
        reason = parsed.get("reason")
        return {
            "score": int(score) if score is not None else None,
            "reason": reason,
        }
    except Exception:
        return {
            "score": None,
            "reason": raw_response,
        }


def summarize_results(results: list[dict]) -> dict:
    valid_scores = [r["judge_score"] for r in results if r["judge_score"] is not None]

    if valid_scores:
        average_score = round(sum(valid_scores) / len(valid_scores), 2)
    else:
        average_score = None

    distribution = {}
    for score in valid_scores:
        distribution[str(score)] = distribution.get(str(score), 0) + 1

    return {
        "num_examples": len(results),
        "num_valid_scores": len(valid_scores),
        "average_score": average_score,
        "score_distribution": distribution,
    }


def run_batch_llm_judge_evaluation():
    with langfuse.start_as_current_observation(
        as_type="span",
        name="batch-llm-judge-evaluation",
    ) as root_span:

        with propagate_attributes(session_id="evaluation-session-004"):

            root_span.update(
                user_id="demo-user-alura",
                tags=["llm-as-judge-batch", OPENAI_MODEL, JUDGE_MODEL],
                metadata={
                    "dataset": DATASET_PATH,
                    "evaluation_type": "batch-open-answer-judge",
                    "judge_prompt_path": JUDGE_PROMPT_PATH,
                    "answer_prompt_name": ANSWER_PROMPT_NAME,
                    "answer_prompt_label": ANSWER_PROMPT_LABEL,
                    "num_examples": N_EXAMPLES,
                },
            )

            # Etapa 1 — carregar prompts
            with root_span.start_as_current_observation(
                as_type="span",
                name="load-prompts",
            ) as prompt_span:

                answer_prompt_client = langfuse.get_prompt(
                    name=ANSWER_PROMPT_NAME,
                    label=ANSWER_PROMPT_LABEL,
                )
                answer_system_prompt = answer_prompt_client.prompt

                judge_template = load_prompt(JUDGE_PROMPT_PATH)

                prompt_span.update(
                    output={
                        "answer_prompt_name": ANSWER_PROMPT_NAME,
                        "answer_prompt_label": ANSWER_PROMPT_LABEL,
                        "judge_prompt_path": JUDGE_PROMPT_PATH,
                    }
                )

            # Etapa 2 — carregar dataset
            with root_span.start_as_current_observation(
                as_type="span",
                name="load-dataset-sample",
            ) as dataset_span:

                df = pd.read_csv(DATASET_PATH)
                sample_df = df.head(N_EXAMPLES).copy()

                dataset_span.update(
                    output={
                        "num_loaded_examples": len(sample_df),
                    },
                    metadata={
                        "dataset_path": DATASET_PATH,
                    },
                )

            results = []

            # Etapa 3 — avaliação em lote
            for idx, row in sample_df.iterrows():
                with root_span.start_as_current_observation(
                    as_type="span",
                    name=f"evaluate-example-{idx}",
                ) as example_span:

                    question = row["instruction"]
                    expected_answer = row["response"]
                    category = row["category"]
                    flags = row["flags"]

                    # Geração da resposta da aplicação
                    answer_completion = openai.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": answer_system_prompt},
                            {"role": "user", "content": question},
                        ],
                        name=f"generate-answer-{idx}",
                    )

                    model_answer = answer_completion.choices[0].message.content.strip()

                    # Prompt do juiz
                    final_judge_prompt = build_prompt(
                        judge_template,
                        {
                            "question": question,
                            "expected_answer": expected_answer,
                            "model_answer": model_answer,
                        },
                    )

                    # Avaliação com LLM-as-Judge
                    judge_completion = openai.chat.completions.create(
                        model=JUDGE_MODEL,
                        messages=[
                            {
                                "role": "system",
                                "content": "Você é um avaliador de respostas de atendimento ao cliente.",
                            },
                            {"role": "user", "content": final_judge_prompt},
                        ],
                        name=f"judge-answer-{idx}",
                    )

                    raw_judge_response = judge_completion.choices[0].message.content.strip()
                    judge_result = parse_judge_response(raw_judge_response)

                    example_result = {
                        "question": question,
                        "expected_answer": expected_answer,
                        "model_answer": model_answer,
                        "judge_score": judge_result["score"],
                        "judge_reason": judge_result["reason"],
                        "category": category,
                        "flags": flags,
                    }

                    example_span.update(
                        input={
                            "question": question,
                            "expected_answer": expected_answer,
                        },
                        output={
                            "model_answer": model_answer,
                            "judge_score": judge_result["score"],
                            "judge_reason": judge_result["reason"],
                        },
                        metadata={
                            "category": category,
                            "flags": flags,
                        },
                    )

                    results.append(example_result)
                    time.sleep(0.1)

            # Etapa 4 — agregação
            with root_span.start_as_current_observation(
                as_type="span",
                name="aggregate-results",
            ) as aggregate_span:

                summary = summarize_results(results)

                aggregate_span.update(
                    output=summary
                )

            root_span.update(
                output={
                    "summary": summary,
                    "results": results,
                }
            )

        langfuse.flush()

        return {
            "summary": summary,
            "results": results,
        }


if __name__ == "__main__":
    result = run_batch_llm_judge_evaluation()

    print("\nResumo da avaliação:\n")
    print(result["summary"])

    print("\nResultados individuais:\n")
    for i, item in enumerate(result["results"], start=1):
        print(f"Exemplo {i}")
        print("Pergunta:", item["question"])
        print("Score:", item["judge_score"])
        print("Justificativa:", item["judge_reason"])
        print("-" * 50)