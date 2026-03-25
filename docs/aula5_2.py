# AULA 5.2 
import time
import pandas as pd
from dotenv import load_dotenv

from langfuse import get_client, propagate_attributes
from langfuse.openai import openai

load_dotenv()

langfuse = get_client()

OPENAI_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"

DATASET_PATH = "../data/bitext_customer_support.csv"
ANSWER_PROMPT_NAME = "customer_support_assistant"
ANSWER_PROMPT_LABEL = "production"
JUDGE_PROMPT_PATH = "./prompts/customer_support_judge.md"

N_EXAMPLES = 5


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_judge_input(question: str, expected_answer: str, model_answer: str) -> str:
    return (
        f"Pergunta:\n{question}\n\n"
        f"Resposta esperada:\n{expected_answer}\n\n"
        f"Resposta gerada:\n{model_answer}"
    )


def summarize_monitoring_results(results: list[dict]) -> dict:
    latencies = [r["latency_ms"] for r in results]

    return {
        "num_examples": len(results),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "judge_answers_preview": [
            {
                "question": r["question"],
                "judge_score": r["judge_score"],
            }
            for r in results[:3]
        ],
    }


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

                df = pd.read_csv(DATASET_PATH).head(N_EXAMPLES)

                dataset_span.update(
                    output={
                        "num_loaded_examples": len(df),
                    }
                )

            results = []

            for idx, row in df.iterrows():
                with root_span.start_as_current_observation(
                    as_type="span",
                    name=f"example-{idx}",
                ) as span:

                    question = row["instruction"]
                    expected_answer = row.get("response")
                    category = row.get("category")
                    flags = row.get("flags")

                    start_time = time.perf_counter()

                    completion = openai.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": answer_prompt},
                            {"role": "user", "content": question},
                        ],
                        name=f"generate-answer-{idx}",
                    )

                    answer = completion.choices[0].message.content.strip()

                    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

                    usage = getattr(completion, "usage", None)
                    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                    output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

                    judge_input = build_judge_input(
                        question,
                        expected_answer,
                        answer,
                    )

                    judge_completion = openai.chat.completions.create(
                        model=JUDGE_MODEL,
                        messages=[
                            {"role": "system", "content": judge_prompt_template},
                            {"role": "user", "content": judge_input},
                        ],
                        name=f"judge-answer-{idx}",
                    )

                    judge_raw = judge_completion.choices[0].message.content.strip()
                    judge_score = judge_raw
                    severity = "review"

                    span.update(
                        input={
                            "question": question,
                            "expected_answer": expected_answer,
                        },
                        output={
                            "answer": answer,
                            "latency_ms": latency_ms,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "judge_score": judge_score,
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
                            "judge_score": judge_score,
                            "latency_ms": latency_ms,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "severity": severity,
                            "category": category,
                            "flags": flags,
                        }
                    )

            with root_span.start_as_current_observation(
                as_type="span",
                name="aggregate-monitoring-quality",
            ) as aggregate_span:

                summary = summarize_monitoring_results(results)

                aggregate_span.update(output=summary)

            root_span.update(
                output={
                    "summary": summary,
                    "results": results,
                }
            )

        langfuse.flush()


if __name__ == "__main__":
    run_monitoring_quality()
