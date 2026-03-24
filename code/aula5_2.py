import os
import time
import json
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

# Prompt principal continua no Langfuse
ANSWER_PROMPT_NAME = "customer_support_assistant"
ANSWER_PROMPT_LABEL = "production"

# Prompt de judge passa a ser local
JUDGE_PROMPT_PATH = "./prompts/customer_support_judge.md"

N_EXAMPLES = 5


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_judge_prompt(template: str, question: str, expected_answer: str, model_answer: str) -> str:
    return (
        template
        .replace("{{question}}", question)
        .replace("{{expected_answer}}", str(expected_answer))
        .replace("{{model_answer}}", model_answer)
    )


def summarize_monitoring_results(results: list[dict]) -> dict:
    scores = [r["judge_score"] for r in results if r["judge_score"] is not None]
    low_score_cases = [
        {
            "question": r["question"],
            "judge_score": r["judge_score"],
            "judge_reason": r["judge_reason"],
        }
        for r in results
        if r["judge_score"] is not None and r["judge_score"] <= 2
    ]

    return {
        "num_examples": len(results),
        "avg_score": round(sum(scores) / len(scores), 2) if scores else None,
        "low_score_cases": low_score_cases[:3],
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

            # Etapa 1 — carregar prompt principal do Langfuse
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

            # Etapa 2 — carregar prompt de judge local
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

            # Etapa 3 — carregar dataset
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

            # Etapa 4 — processar exemplos
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

                    # Judge local
                    judge_prompt = build_judge_prompt(
                        judge_prompt_template,
                        question,
                        expected_answer,
                        answer,
                    )

                    judge_completion = openai.chat.completions.create(
                        model=JUDGE_MODEL,
                        messages=[
                            {"role": "system", "content": "Você é um avaliador de respostas de atendimento ao cliente."},
                            {"role": "user", "content": judge_prompt},
                        ],
                        response_format={"type": "json_object"},
                        name=f"judge-answer-{idx}",
                    )

                    judge_raw = judge_completion.choices[0].message.content.strip()

                    try:
                        judge_data = json.loads(judge_raw)
                        judge_score = int(judge_data.get("score"))
                        judge_reason = judge_data.get("reason")
                    except Exception:
                        judge_score = None
                        judge_reason = judge_raw

                    severity = "critical" if judge_score is not None and judge_score <= 2 else "ok"

                    if severity == "critical":
                        with span.start_as_current_observation(
                            as_type="event",
                            name="low-quality-detected",
                            input={
                                "judge_score": judge_score,
                                "judge_reason": judge_reason,
                            },
                        ):
                            pass

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
                            "judge_reason": judge_reason,
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
                            "judge_reason": judge_reason,
                            "latency_ms": latency_ms,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "severity": severity,
                            "category": category,
                            "flags": flags,
                        }
                    )

            # Etapa 5 — consolidação
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