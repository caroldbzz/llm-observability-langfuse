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
FALLBACK_PROMPT_NAME = "customer_support_assistant"
FALLBACK_PROMPT_LABEL = "fallback"
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


def run_monitoring_analysis():
    with langfuse.start_as_current_observation(
        as_type="span",
        name="monitoring-fallback-batch",
    ) as root_span:
        with propagate_attributes(session_id="monitoring-session-003"):
            root_span.update(
                user_id="demo-user-alura",
                tags=["monitoring-fallback", OPENAI_MODEL, JUDGE_MODEL],
                metadata={
                    "dataset_path": DATASET_PATH,
                    "answer_prompt_name": ANSWER_PROMPT_NAME,
                    "answer_prompt_label": ANSWER_PROMPT_LABEL,
                    "fallback_prompt_name": FALLBACK_PROMPT_NAME,
                    "fallback_prompt_label": FALLBACK_PROMPT_LABEL,
                    "judge_prompt_path": JUDGE_PROMPT_PATH,
                    "num_examples": N_EXAMPLES,
                    "critical_score_threshold": CRITICAL_SCORE_THRESHOLD,
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
                name="load-fallback-prompt",
            ) as fallback_prompt_span:
                fallback_prompt = langfuse.get_prompt(
                    name=FALLBACK_PROMPT_NAME,
                    label=FALLBACK_PROMPT_LABEL,
                ).prompt

                fallback_prompt_span.update(
                    output={
                        "fallback_prompt_name": FALLBACK_PROMPT_NAME,
                        "fallback_prompt_label": FALLBACK_PROMPT_LABEL,
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
                    initial_answer = answer_completion.choices[0].message.content.strip()
                    initial_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

                    usage = getattr(answer_completion, "usage", None)
                    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                    output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

                    judge_input = (
                        f"Pergunta:\n{question}\n\n"
                        f"Resposta esperada:\n{expected_answer}\n\n"
                        f"Resposta gerada:\n{initial_answer}"
                    )
                    judge_completion = openai.chat.completions.create(
                        model=JUDGE_MODEL,
                        messages=[
                            {"role": "system", "content": judge_prompt_template},
                            {"role": "user", "content": judge_input},
                        ],
                        name=f"judge-answer-{idx}",
                    )
                    initial_judge_score = judge_completion.choices[0].message.content.strip()
                    initial_score_value = extract_judge_score(initial_judge_score)
                    initial_severity = calculate_severity(initial_score_value)

                    final_answer = initial_answer
                    final_judge_score = initial_judge_score
                    final_score_value = initial_score_value
                    used_fallback = False
                    mitigation_status = "original_kept"

                    fallback_latency_ms = 0.0
                    fallback_input_tokens = 0
                    fallback_output_tokens = 0

                    if initial_severity == "critical":
                        used_fallback = True
                        mitigation_status = "fallback_applied"

                        fallback_start = time.perf_counter()
                        fallback_completion = openai.chat.completions.create(
                            model=OPENAI_MODEL,
                            messages=[
                                {"role": "system", "content": fallback_prompt},
                                {"role": "user", "content": question},
                            ],
                            name=f"generate-fallback-answer-{idx}",
                        )
                        fallback_answer = fallback_completion.choices[0].message.content.strip()
                        fallback_latency_ms = round((time.perf_counter() - fallback_start) * 1000, 2)

                        fallback_usage = getattr(fallback_completion, "usage", None)
                        fallback_input_tokens = (
                            getattr(fallback_usage, "prompt_tokens", 0)
                            if fallback_usage
                            else 0
                        )
                        fallback_output_tokens = (
                            getattr(fallback_usage, "completion_tokens", 0)
                            if fallback_usage
                            else 0
                        )

                        fallback_judge_input = (
                            f"Pergunta:\n{question}\n\n"
                            f"Resposta esperada:\n{expected_answer}\n\n"
                            f"Resposta gerada:\n{fallback_answer}"
                        )
                        fallback_judge_completion = openai.chat.completions.create(
                            model=JUDGE_MODEL,
                            messages=[
                                {"role": "system", "content": judge_prompt_template},
                                {"role": "user", "content": fallback_judge_input},
                            ],
                            name=f"judge-fallback-answer-{idx}",
                        )
                        fallback_judge_score = (
                            fallback_judge_completion.choices[0].message.content.strip()
                        )
                        fallback_score_value = extract_judge_score(fallback_judge_score)

                        final_answer = fallback_answer
                        final_judge_score = fallback_judge_score
                        final_score_value = fallback_score_value

                        with example_span.start_as_current_observation(
                            as_type="event",
                            name="fallback-applied",
                            input={
                                "initial_score": initial_score_value,
                                "fallback_score": fallback_score_value,
                            },
                        ):
                            pass

                    latency_ms = round(initial_latency_ms + fallback_latency_ms, 2)
                    total_input_tokens = input_tokens + fallback_input_tokens
                    total_output_tokens = output_tokens + fallback_output_tokens
                    final_severity = calculate_severity(final_score_value)

                    example_span.update(
                        input={
                            "question": question,
                            "expected_answer": expected_answer,
                        },
                        output={
                            "initial_answer": initial_answer,
                            "final_answer": final_answer,
                            "initial_judge_score": initial_judge_score,
                            "final_judge_score": final_judge_score,
                            "initial_score_value": initial_score_value,
                            "final_score_value": final_score_value,
                            "used_fallback": used_fallback,
                            "latency_ms": latency_ms,
                            "input_tokens": total_input_tokens,
                            "output_tokens": total_output_tokens,
                        },
                        metadata={
                            "category": category,
                            "flags": flags,
                            "initial_severity": initial_severity,
                            "final_severity": final_severity,
                            "mitigation_status": mitigation_status,
                        },
                    )

                    results.append(
                        {
                            "question": question,
                            "expected_answer": expected_answer,
                            "initial_answer": initial_answer,
                            "final_answer": final_answer,
                            "initial_judge_score": initial_judge_score,
                            "final_judge_score": final_judge_score,
                            "initial_score_value": initial_score_value,
                            "final_score_value": final_score_value,
                            "used_fallback": used_fallback,
                            "latency_ms": latency_ms,
                            "input_tokens": total_input_tokens,
                            "output_tokens": total_output_tokens,
                            "initial_severity": initial_severity,
                            "final_severity": final_severity,
                            "mitigation_status": mitigation_status,
                            "category": category,
                            "flags": flags,
                        }
                    )
                    time.sleep(0.1)

            with root_span.start_as_current_observation(
                as_type="span",
                name="aggregate-monitoring-fallback",
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
                    "judge_answers_preview": [
                        item["final_judge_score"] for item in results[:3]
                    ],
                    "avg_latency_ms": avg_latency_ms,
                    "total_input_tokens": sum(item["input_tokens"] for item in results),
                    "total_output_tokens": sum(item["output_tokens"] for item in results),
                    "num_fallback_applied": sum(
                        1 for item in results if item["used_fallback"]
                    ),
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
    result = run_monitoring_analysis()

    print("\nResumo do monitoramento com fallback:\n")
    print(result["summary"])

    print("\nResultados individuais:\n")
    for i, item in enumerate(result["results"], start=1):
        print(f"Exemplo {i}")
        print("Pergunta:", item["question"])
        print("Score inicial:", item["initial_score_value"])
        print("Score final:", item["final_score_value"])
        print("Fallback aplicado:", item["used_fallback"])
        print("Status mitigação:", item["mitigation_status"])
        print("-" * 50)
