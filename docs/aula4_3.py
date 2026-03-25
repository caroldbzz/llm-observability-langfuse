import os
import time
import pandas as pd
from dotenv import load_dotenv

from langfuse import get_client, propagate_attributes
from langfuse.openai import openai

load_dotenv()

langfuse = get_client()

OPENAI_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"

DATASET_PATH = "docs/data/bitext_customer_support.csv"
ANSWER_PROMPT_NAME = "customer_support_assistant"
ANSWER_PROMPT_LABEL = "production"
JUDGE_PROMPT_PATH = "docs/prompts/answer_judge.md"

N_EXAMPLES = 5


def load_prompt(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def run_batch_llm_judge_evaluation(N_EXAMPLES: int = N_EXAMPLES):
    with langfuse.start_as_current_observation(
        as_type="span",
        name="batch-llm-judge-evaluation",
    ) as root_span:

        with propagate_attributes(
            session_id="evaluation-session-004",
            user_id="demo-user-alura",
            tags=["llm-as-judge-batch", OPENAI_MODEL, JUDGE_MODEL],
            metadata={
                "dataset": DATASET_PATH,
                "evaluation_type": "batch-open-answer-judge",
                "judge_prompt_path": JUDGE_PROMPT_PATH,
                "answer_prompt_name": ANSWER_PROMPT_NAME,
                "answer_prompt_label": ANSWER_PROMPT_LABEL,
                "num_examples": str(N_EXAMPLES),
            },
        ):

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

            for idx, row in sample_df.iterrows():
                with root_span.start_as_current_observation(
                    as_type="span",
                    name=f"evaluate-example-{idx}",
                ) as example_span:

                    question = row["instruction"]
                    expected_answer = row["response"]
                    category = row["category"]
                    flags = row["flags"]

                    answer_completion = openai.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": answer_system_prompt},
                            {"role": "user", "content": question},
                        ],
                        name=f"generate-answer-{idx}",
                    )

                    model_answer = answer_completion.choices[0].message.content.strip()

                    judge_input = (
                        f"Pergunta:\n{question}\n\n"
                        f"Resposta esperada:\n{expected_answer}\n\n"
                        f"Resposta gerada:\n{model_answer}"
                    )

                    judge_completion = openai.chat.completions.create(
                        model=JUDGE_MODEL,
                        messages=[
                            {"role": "system", "content": judge_template},
                            {"role": "user", "content": judge_input},
                        ],
                        name=f"judge-answer-{idx}",
                    )

                    raw_judge_response = judge_completion.choices[0].message.content.strip()
                    judge_score = raw_judge_response

                    example_result = {
                        "question": question,
                        "expected_answer": expected_answer,
                        "model_answer": model_answer,
                        "judge_score": judge_score,
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
                            "judge_score": judge_score,
                        },
                        metadata={
                            "category": category,
                            "flags": flags,
                        },
                    )

                    results.append(example_result)
                    time.sleep(0.1)


            with root_span.start_as_current_observation(
                as_type="span",
                name="aggregate-results",
            ) as aggregate_span:

                summary = {
                    "num_examples": len(results),
                    "num_judge_answers": len(results),
                    "judge_answers_preview": [r["judge_score"] for r in results[:3]],
                }

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
        print("Avaliação do juiz:", item["judge_score"])
        print("-" * 50)
