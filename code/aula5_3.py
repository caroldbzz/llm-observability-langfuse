import os
import time
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from langfuse import get_client, propagate_attributes
from langfuse.openai import openai

load_dotenv()

langfuse = get_client()
openai.api_key = os.getenv("OPENAI_API_KEY")

OPENAI_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent / "data" / "bitext_customer_support.csv"

ANSWER_PROMPT_NAME = "customer-support-assistant"
ANSWER_PROMPT_LABEL = "production"

JUDGE_PROMPT_PATH = BASE_DIR / "prompts" / "customer_support_judge.md"

N_EXAMPLES = 8


def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_judge_prompt(template, question, expected_answer, model_answer):
    return (
        template
        .replace("{{question}}", question)
        .replace("{{expected_answer}}", str(expected_answer))
        .replace("{{model_answer}}", model_answer)
    )


def parse_judge_response(raw):
    try:
        data = json.loads(raw)
        return int(data.get("score")), data.get("reason")
    except Exception:
        return None, raw


def calculate_category_summary(results):
    category_scores = defaultdict(list)

    for r in results:
        category = r.get("category") or "unknown"
        if r.get("judge_score") is not None:
            category_scores[category].append(r["judge_score"])

    summary = []
    for category, scores in category_scores.items():
        summary.append({
            "category": category,
            "count": len(scores),
            "avg_score": round(sum(scores)/len(scores), 2) if scores else None
        })

    return sorted(summary, key=lambda x: x["avg_score"] or 999)


def summarize_monitoring_results(results):
    valid_scores = [r["judge_score"] for r in results if r["judge_score"] is not None]
    latencies = [r["latency_ms"] for r in results]

    total_input_tokens = sum(r["input_tokens"] for r in results)
    total_output_tokens = sum(r["output_tokens"] for r in results)

    score_distribution = dict(Counter(valid_scores))

    low_score_cases = [
        r for r in results
        if r["judge_score"] is not None and r["judge_score"] <= 2
    ]

    return {
        "num_examples": len(results),
        "avg_score": round(sum(valid_scores)/len(valid_scores), 2) if valid_scores else None,
        "score_distribution": score_distribution,
        "avg_latency_ms": round(sum(latencies)/len(latencies), 2),
        "total_tokens": total_input_tokens + total_output_tokens,
        "num_critical_cases": len(low_score_cases),
        "low_score_cases": low_score_cases[:5],
        "category_summary": calculate_category_summary(results),
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

                    judge_prompt = build_judge_prompt(
                        judge_template, question, expected, answer
                    )

                    judge_resp = openai.chat.completions.create(
                        model=JUDGE_MODEL,
                        messages=[{"role": "user", "content": judge_prompt}],
                        response_format={"type": "json_object"},
                    )

                    raw = judge_resp.choices[0].message.content
                    score, reason = parse_judge_response(raw)

                    span.update(
                        output={
                            "answer": answer,
                            "latency_ms": latency,
                            "judge_score": score,
                        }
                    )

                    results.append({
                        "question": question,
                        "judge_score": score,
                        "judge_reason": reason,
                        "latency_ms": latency,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "category": category,
                    })

            summary = summarize_monitoring_results(results)

            root.update(output={"summary": summary})

        langfuse.flush()
        return {"summary": summary, "results": results}


def run():
    return run_monitoring_analysis()


if __name__ == "__main__":
    run()
