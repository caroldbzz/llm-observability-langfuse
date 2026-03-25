import os
import time
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv

from langfuse import get_client, propagate_attributes
from langfuse.openai import openai

load_dotenv()

langfuse = get_client()

OPENAI_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4.1-mini"

DATASET_PATH = "data/bitext_customer_support.csv"
ANSWER_PROMPT_NAME = "customer_support_assistant"
ANSWER_PROMPT_LABEL = "production"

FALLBACK_PROMPT_NAME = "customer_support_assistant"
FALLBACK_PROMPT_LABEL = "fallback"

JUDGE_PROMPT_PATH = "code/prompts/customer_support_judge.md"

N_EXAMPLES = 8


def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_judge_input(question, expected_answer, model_answer):
    return (
        f"Pergunta:\n{question}\n\n"
        f"Resposta esperada:\n{expected_answer}\n\n"
        f"Resposta gerada:\n{model_answer}"
    )


def evaluate_answer(question, expected_answer, answer, judge_template):
    judge_input = build_judge_input(
        question, expected_answer, answer
    )

    judge_resp = openai.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": judge_template},
            {"role": "user", "content": judge_input},
        ],
    )

    return judge_resp.choices[0].message.content.strip()


def calculate_category_summary(results):
    category_counts = defaultdict(int)

    for r in results:
        category = r.get("category") or "unknown"
        category_counts[category] += 1

    summary = []
    for category, count in category_counts.items():
        summary.append({
            "category": category,
            "count": count,
            "avg_score": None,
        })

    return sorted(summary, key=lambda x: x["count"], reverse=True)


def summarize_monitoring_results(results):
    latencies = [r["latency_ms"] for r in results]

    total_input_tokens = sum(r["input_tokens"] for r in results)
    total_output_tokens = sum(r["output_tokens"] for r in results)

    num_fallback_applied = sum(1 for r in results if r.get("used_fallback"))

    return {
        "num_examples": len(results),
        "avg_score": None,
        "score_distribution": {},
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "total_tokens": total_input_tokens + total_output_tokens,
        "num_critical_cases": 0,
        "low_score_cases": [],
        "judge_answers_preview": [r["judge_score"] for r in results[:5]],
        "category_summary": calculate_category_summary(results),
        "num_fallback_applied": num_fallback_applied,
    }


def run_monitoring_analysis():
    with langfuse.start_as_current_observation(
        as_type="span",
        name="monitoring-analysis-batch",
    ) as root:

        with propagate_attributes(session_id="monitoring-session-003"):

            answer_prompt = langfuse.get_prompt(
                name=ANSWER_PROMPT_NAME,
                label=ANSWER_PROMPT_LABEL
            ).prompt

            judge_template = load_prompt(JUDGE_PROMPT_PATH)

            df = pd.read_csv(DATASET_PATH).head(N_EXAMPLES)

            results = []

            for idx, row in df.iterrows():
                with root.start_as_current_observation(
                    as_type="span",
                    name=f"example-{idx}",
                ) as span:

                    question = row["instruction"]
                    expected = row.get("response")
                    category = row.get("category")

                    start = time.perf_counter()

                    completion = openai.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": answer_prompt},
                            {"role": "user", "content": question},
                        ],
                    )

                    answer = completion.choices[0].message.content.strip()
                    latency = (time.perf_counter() - start) * 1000

                    usage = completion.usage
                    input_tokens = usage.prompt_tokens
                    output_tokens = usage.completion_tokens

                    judge_score = evaluate_answer(
                        question, expected, answer, judge_template
                    )

                    used_fallback = False

                    span.update(
                        output={
                            "answer": answer,
                            "latency_ms": latency,
                            "judge_score": judge_score,
                            "used_fallback": used_fallback,
                        },
                        metadata={
                            "category": category,
                            "mitigation_status": "original_kept",
                        }
                    )

                    results.append({
                        "question": question,
                        "judge_score": judge_score,
                        "used_fallback": used_fallback,
                        "latency_ms": latency,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "category": category,
                    })

            summary = summarize_monitoring_results(results)

            root.update(output={"summary": summary, "results": results})

        langfuse.flush()
        return {"summary": summary, "results": results}


def run():
    return run_monitoring_analysis()


if __name__ == "__main__":
    run()
