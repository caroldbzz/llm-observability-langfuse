import re
import time

import pandas as pd
from dotenv import load_dotenv

from langfuse import get_client, propagate_attributes
from langfuse.openai import openai

load_dotenv()

langfuse = get_client()

OPENAI_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4.1-mini"

DATASET_PATH = "docs/data/bitext_customer_support.csv"
ANSWER_PROMPT_NAME = "customer_support_assistant"
ANSWER_PROMPT_LABEL = "production"
JUDGE_PROMPT_PATH = "docs/prompts/customer_support_judge.md"
CRITICAL_SCORE_THRESHOLD = 2

N_EXAMPLES = 5


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def extract_judge_score(judge_output: str):
    match = re.search(r"SCORE:\s*([1-5])", judge_output)
    return int(match.group(1)) if match else None


def calculate_severity(score_value):
    if score_value is None:
        return "review"
    if score_value <= CRITICAL_SCORE_THRESHOLD:
        return "critical"
    if score_value == 3:
        return "review"
    return "ok"


def run_monitoring_quality():
    with langfuse.start_as_current_observation(
        as_type="span",
        name="monitoring-quality-batch",
    ) as root_span:
        with propagate_attributes(session_id="monitoring-session-002"):
            root_span.update(
                user_id="demo-user-alura",
                tags=["monitoring-quality", OPENAI_MODEL, JUDGE_MODEL],
                metadata={
                    "dataset_path": DATASET_PATH,
                    "answer_prompt_name": ANSWER_PROMPT_NAME,
                    "answer_prompt_label": ANSWER_PROMPT_LABEL,
                    "judge_prompt_path": JUDGE_PROMPT_PATH,
                    "num_examples": N_EXAMPLES,
                },
            )

            with root_span.start_as_current_observation(
                as_type="span",
                name="load-application-prompt",
            ) as answer_prompt_span:
                answer_prompt = langfuse.get_prompt(
                    name=ANSWER_PROMPT_NAME,
                    label=ANSWER_PROMPT_LABEL,
                ).prompt

                answer_prompt_span.update(
                    output={
                        "answer_prompt_name": ANSWER_PROMPT_NAME,
                        "answer_prompt_label": ANSWER_PROMPT_LABEL,
                    }
                )

            with root_span.start_as_current_observation(
                as_type="span",
                name="load-judge-prompt-local",
            ) as judge_prompt_span:
                judge_prompt_template = load_prompt(JUDGE_PROMPT_PATH)
                judge_prompt_span.update(
                    output={
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

                    start_time = time.perf_counter()
                    answer_completion = openai.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": answer_prompt},
                            {"role": "user", "content": question},
                        ],
                        name=f"generate-answer-{idx}",
                    )
                    model_answer = answer_completion.choices[0].message.content.strip()
                    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

                    usage = getattr(answer_completion, "usage", None)
                    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                    output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

                    judge_input = (
                        f"Pergunta:\n{question}\n\n"
                        f"Resposta esperada:\n{expected_answer}\n\n"
                        f"Resposta gerada:\n{model_answer}"
                    )
                    judge_completion = openai.chat.completions.create(
                        model=JUDGE_MODEL,
                        messages=[
                            {"role": "system", "content": judge_prompt_template},
                            {"role": "user", "content": judge_input},
                        ],
                        name=f"judge-answer-{idx}",
                    )
                    judge_score = judge_completion.choices[0].message.content.strip()
                    score_value = extract_judge_score(judge_score)
                    severity = calculate_severity(score_value)

                    example_span.update(
                        input={
                            "question": question,
                            "expected_answer": expected_answer,
                        },
                        output={
                            "model_answer": model_answer,
                            "judge_score": judge_score,
                            "judge_score_value": score_value,
                            "latency_ms": latency_ms,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                        },
                        metadata={
                            "category": category,
                            "flags": flags,
                            "severity": severity,
                        },
                    )

                    results.append(
                        {
                            "question": question,
                            "expected_answer": expected_answer,
                            "model_answer": model_answer,
                            "judge_score": judge_score,
                            "judge_score_value": score_value,
                            "latency_ms": latency_ms,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "severity": severity,
                            "category": category,
                            "flags": flags,
                        }
                    )
                    time.sleep(0.1)

            with root_span.start_as_current_observation(
                as_type="span",
                name="aggregate-monitoring-quality",
            ) as aggregate_span:
                num_examples = len(results)
                avg_latency_ms = (
                    round(
                        sum(item["latency_ms"] for item in results) / num_examples,
                        2,
                    )
                    if num_examples
                    else 0
                )
                summary = {
                    "num_examples": num_examples,
                    "num_judge_answers": num_examples,
                    "judge_answers_preview": [item["judge_score"] for item in results[:3]],
                    "avg_latency_ms": avg_latency_ms,
                    "total_input_tokens": sum(item["input_tokens"] for item in results),
                    "total_output_tokens": sum(item["output_tokens"] for item in results),
                }
                aggregate_span.update(output=summary)

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
    result = run_monitoring_quality()

    print("\nResumo do monitoramento:\n")
    print(result["summary"])

    print("\nResultados individuais:\n")
    for i, item in enumerate(result["results"], start=1):
        print(f"Exemplo {i}")
        print("Pergunta:", item["question"])
        print("Avaliação do juiz:", item["judge_score"])
        print("-" * 50)
