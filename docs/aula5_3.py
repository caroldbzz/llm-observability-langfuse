import time
from collections import Counter

import pandas as pd
from dotenv import load_dotenv
from langfuse import get_client, propagate_attributes
from langfuse.openai import openai

load_dotenv()

langfuse = get_client()

DATASET_PATH = "data/bitext_customer_support.csv"
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_MODEL_JUDGE = "gpt-4.1-mini"
JUDGE_PROMPT_PATH = "prompts/customer_support_answer_judge.md"
ANSWER_PROMPT_NAME = "customer_support_assistant"
ANSWER_PROMPT_LABEL = "production"
FALLBACK_PROMPT_NAME = "customer_support_assistant"
FALLBACK_PROMPT_LABEL = "fallback"
CRITICAL_SCORE_THRESHOLD = 2

N_EXAMPLES = 100
MAX_REVIEW_CASES_TO_PRINT = 2


def load_prompt(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def extract_judge_score(judge_output):
    return int(judge_output.split("SCORE:", 1)[1].split()[0])


def calculate_severity(score_value):
    if score_value <= CRITICAL_SCORE_THRESHOLD:
        return "critical"
    if score_value == 3:
        return "review"
    return "ok"


def build_judge_input(question, expected_answer, generated_answer):
    return (
        f"Pergunta:\n{question}\n\n"
        f"Resposta esperada:\n{expected_answer}\n\n"
        f"Resposta gerada:\n{generated_answer}"
    )


def generate_answer(system_prompt, question, request_name):
    completion = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        name=request_name,
    )

    answer = completion.choices[0].message.content.strip()
    usage = completion.usage

    return answer, usage.prompt_tokens, usage.completion_tokens


def judge_answer(judge_prompt, question, expected_answer, generated_answer, request_name):
    judge_input = build_judge_input(question, expected_answer, generated_answer)

    completion = openai.chat.completions.create(
        model=OPENAI_MODEL_JUDGE,
        messages=[
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": judge_input},
        ],
        name=request_name,
    )

    judge_score = completion.choices[0].message.content.strip()
    score_value = extract_judge_score(judge_score)
    severity = calculate_severity(score_value)

    return judge_score, score_value, severity


def evaluate_example(row, idx, answer_system_prompt, fallback_system_prompt, judge_prompt, evaluation_span):
    question = row["instruction"]
    expected_answer = row["response"]
    category = row["category"]
    flags = row["flags"]

    with evaluation_span.start_as_current_observation(
        as_type="span",
        name=f"generate-answer-{idx}",
    ):
        initial_answer, input_tokens, output_tokens = generate_answer(
            answer_system_prompt,
            question,
            f"generate-answer-{idx}",
        )

    with evaluation_span.start_as_current_observation(
        as_type="span",
        name=f"judge-answer-{idx}",
    ):
        initial_judge_score, initial_score_value, initial_severity = judge_answer(
            judge_prompt,
            question,
            expected_answer,
            initial_answer,
            f"judge-answer-{idx}",
        )

    final_answer = initial_answer
    final_judge_score = initial_judge_score
    final_score_value = initial_score_value
    used_fallback = False
    mitigation_status = "original_kept"

    fallback_input_tokens = 0
    fallback_output_tokens = 0

    if initial_severity == "critical":
        used_fallback = True
        mitigation_status = "fallback_applied"

        with evaluation_span.start_as_current_observation(
            as_type="span",
            name=f"generate-fallback-answer-{idx}",
        ):
            fallback_answer, fallback_input_tokens, fallback_output_tokens = generate_answer(
                fallback_system_prompt,
                question,
                f"generate-fallback-answer-{idx}",
            )

        with evaluation_span.start_as_current_observation(
            as_type="span",
            name=f"judge-fallback-answer-{idx}",
        ):
            fallback_judge_score, fallback_score_value, _ = judge_answer(
                judge_prompt,
                question,
                expected_answer,
                fallback_answer,
                f"judge-fallback-answer-{idx}",
            )

        final_answer = fallback_answer
        final_judge_score = fallback_judge_score
        final_score_value = fallback_score_value

        with evaluation_span.start_as_current_observation(
            as_type="event",
            name="fallback-applied",
            input={
                "initial_score": initial_score_value,
                "fallback_score": fallback_score_value,
            },
        ):
            pass

    final_severity = calculate_severity(final_score_value)
    total_input_tokens = input_tokens + fallback_input_tokens
    total_output_tokens = output_tokens + fallback_output_tokens

    result = {
        "question": question,
        "expected_answer": expected_answer,
        "initial_answer": initial_answer,
        "final_answer": final_answer,
        "initial_judge_score": initial_judge_score,
        "final_judge_score": final_judge_score,
        "initial_score_value": initial_score_value,
        "final_score_value": final_score_value,
        "used_fallback": used_fallback,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "initial_severity": initial_severity,
        "final_severity": final_severity,
        "mitigation_status": mitigation_status,
        "category": category,
        "flags": flags,
    }

    evaluation_span.update(
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

    return result


def build_summary(results):
    total_examples = len(results)
    num_fallback_applied = sum(1 for item in results if item["used_fallback"])

    return {
        "num_examples": total_examples,
        "num_judge_answers": total_examples + num_fallback_applied,
        "judge_answers_preview": [item["final_judge_score"] for item in results[:3]],
        "total_input_tokens": sum(item["input_tokens"] for item in results),
        "total_output_tokens": sum(item["output_tokens"] for item in results),
        "num_fallback_applied": num_fallback_applied,
    }


def calculate_avg_score(results):
    score_values = [item["final_score_value"] for item in results]
    return round(sum(score_values) / len(score_values), 2)


def calculate_severity_counts(results):
    num_critical = sum(1 for item in results if item["final_severity"] == "critical")
    num_review = sum(1 for item in results if item["final_severity"] in {"critical", "review"})
    return num_critical, num_review


def calculate_category_summary(results):
    category_counter = Counter((item["category"] or "unknown") for item in results)
    return [
        {"category": category, "count": count}
        for category, count in category_counter.most_common()
    ]


def run_batch_llm_judge_pipeline(n_examples=N_EXAMPLES):
    with langfuse.start_as_current_observation(
        as_type="span",
        name="customer-support-llm-pipeline",
    ) as root_span:
        with propagate_attributes(
            session_id="llm-judge-session-123",
            user_id="id-123",
            tags=["customer-support", "llm-as-judge-batch", OPENAI_MODEL, OPENAI_MODEL_JUDGE],
            metadata={
                "environment": "develop",
                "version": "V1",
                "dataset_path": DATASET_PATH,
                "judge_prompt_path": JUDGE_PROMPT_PATH,
                "answer_prompt_name": ANSWER_PROMPT_NAME,
                "answer_prompt_label": ANSWER_PROMPT_LABEL,
                "fallback_prompt_name": FALLBACK_PROMPT_NAME,
                "fallback_prompt_label": FALLBACK_PROMPT_LABEL,
                "num_examples": n_examples,
            },
        ):
            with root_span.start_as_current_observation(
                as_type="span",
                name="load-dataset",
            ) as dataset_span:
                df = pd.read_csv(DATASET_PATH)
                sample_df = df.head(n_examples).copy()

                dataset_span.update(
                    input={"dataset_path": DATASET_PATH},
                    output={"num_samples": len(sample_df)},
                )

            with root_span.start_as_current_observation(
                as_type="event",
                name="dataset-loaded",
            ):
                pass

            with root_span.start_as_current_observation(
                as_type="span",
                name="load-prompts",
            ) as prompt_load_span:
                judge_prompt = load_prompt(JUDGE_PROMPT_PATH)

                answer_system_prompt = langfuse.get_prompt(
                    name=ANSWER_PROMPT_NAME,
                    label=ANSWER_PROMPT_LABEL,
                ).prompt

                fallback_system_prompt = langfuse.get_prompt(
                    name=FALLBACK_PROMPT_NAME,
                    label=FALLBACK_PROMPT_LABEL,
                ).prompt

                prompt_load_span.update(
                    output={
                        "judge_prompt_path": JUDGE_PROMPT_PATH,
                        "answer_prompt_name": ANSWER_PROMPT_NAME,
                        "answer_prompt_label": ANSWER_PROMPT_LABEL,
                        "fallback_prompt_name": FALLBACK_PROMPT_NAME,
                        "fallback_prompt_label": FALLBACK_PROMPT_LABEL,
                    }
                )

            results = []
            for idx, row in sample_df.iterrows():
                with root_span.start_as_current_observation(
                    as_type="span",
                    name=f"evaluation-{idx}",
                ) as evaluation_span:
                    result = evaluate_example(
                        row,
                        idx,
                        answer_system_prompt,
                        fallback_system_prompt,
                        judge_prompt,
                        evaluation_span,
                    )
                    results.append(result)

            with root_span.start_as_current_observation(
                as_type="span",
                name="aggregate-results",
            ) as aggregate_span:
                summary = build_summary(results)
                aggregate_span.update(output=summary)

            with root_span.start_as_current_observation(
                as_type="span",
                name="analyze-priority-case",
            ) as review_span:
                review_cases = [item for item in results if item["final_severity"] != "ok"]

                review_span.update(
                    input={
                        "total_cases": len(results),
                        "critical_threshold": CRITICAL_SCORE_THRESHOLD,
                    },
                    output={
                        "review_cases_count": len(review_cases),
                        "review_cases": review_cases,
                    },
                )

        root_span.update(
            input={"dataset_path": DATASET_PATH, "num_evaluated_cases": len(results)},
            output={
                "summary": summary,
                "results": results,
                "review_cases_count": len(review_cases),
                "review_cases": review_cases,
            },
        )

    langfuse.flush()

    return {
        "summary": summary,
        "results": results,
        "review_cases_count": len(review_cases),
        "review_cases": review_cases,
    }
