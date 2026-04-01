import time

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

N_EXAMPLES = 5


def load_prompt(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def run_batch_llm_judge_pipeline():
    with langfuse.start_as_current_observation(
        as_type="span",
        name="batch-llm-judge-pipeline",
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
                "num_examples": N_EXAMPLES,
            },
        ):
            with root_span.start_as_current_observation(
                as_type="span",
                name="load-dataset",
            ) as dataset_span:
                df = pd.read_csv(DATASET_PATH)
                sample_df = df.head(N_EXAMPLES).copy()

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
                prompt_client = langfuse.get_prompt(
                    name=ANSWER_PROMPT_NAME,
                    label=ANSWER_PROMPT_LABEL,
                )
                answer_system_prompt = prompt_client.prompt
                fallback_prompt_client = langfuse.get_prompt(
                    name=FALLBACK_PROMPT_NAME,
                    label=FALLBACK_PROMPT_LABEL,
                )
                fallback_system_prompt = fallback_prompt_client.prompt

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
                    question = row["instruction"]
                    expected_answer = row["response"]
                    category = row["category"]
                    flags = row["flags"]

                    completion = openai.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": answer_system_prompt},
                            {"role": "user", "content": question},
                        ],
                        name=f"generate-answer-{idx}",
                    )
                    model_answer = completion.choices[0].message.content.strip()

                    usage = completion.usage
                    input_tokens = usage.prompt_tokens
                    output_tokens = usage.completion_tokens

                    judge_input = (
                        f"Pergunta:\n{question}\n\n"
                        f"Resposta esperada:\n{expected_answer}\n\n"
                        f"Resposta gerada:\n{model_answer}"
                    )

                    completion_judge = openai.chat.completions.create(
                        model=OPENAI_MODEL_JUDGE,
                        messages=[
                            {"role": "system", "content": judge_prompt},
                            {"role": "user", "content": judge_input},
                        ],
                        name=f"judge-answer-{idx}",
                    )
                    judge_score = completion_judge.choices[0].message.content.strip()
                    score_value = int(judge_score.split("SCORE:", 1)[1].split()[0])

                    with evaluation_span.start_as_current_observation(
                        as_type="span",
                        name="calculate-initial-severity",
                    ) as initial_severity_span:
                        if score_value <= CRITICAL_SCORE_THRESHOLD:
                            severity = "critical"
                        elif score_value == 3:
                            severity = "review"
                        else:
                            severity = "ok"

                        initial_severity_span.update(
                            input={"score_value": score_value},
                            output={"severity": severity},
                        )

                    final_answer = model_answer
                    final_judge_score = judge_score
                    final_score_value = score_value
                    used_fallback = False
                    mitigation_status = "original_kept"
                    fallback_input_tokens = 0
                    fallback_output_tokens = 0

                    if severity == "critical":
                        used_fallback = True
                        mitigation_status = "fallback_applied"

                        fallback_completion = openai.chat.completions.create(
                            model=OPENAI_MODEL,
                            messages=[
                                {"role": "system", "content": fallback_system_prompt},
                                {"role": "user", "content": question},
                            ],
                            name=f"generate-fallback-answer-{idx}",
                        )
                        fallback_answer = fallback_completion.choices[0].message.content.strip()
                        fallback_usage = fallback_completion.usage
                        fallback_input_tokens = fallback_usage.prompt_tokens
                        fallback_output_tokens = fallback_usage.completion_tokens

                        fallback_judge_input = (
                            f"Pergunta:\n{question}\n\n"
                            f"Resposta esperada:\n{expected_answer}\n\n"
                            f"Resposta gerada:\n{fallback_answer}"
                        )
                        fallback_judge_completion = openai.chat.completions.create(
                            model=OPENAI_MODEL_JUDGE,
                            messages=[
                                {"role": "system", "content": judge_prompt},
                                {"role": "user", "content": fallback_judge_input},
                            ],
                            name=f"judge-fallback-answer-{idx}",
                        )
                        fallback_judge_score = (
                            fallback_judge_completion.choices[0].message.content.strip()
                        )
                        fallback_score_value = int(
                            fallback_judge_score.split("SCORE:", 1)[1].split()[0]
                        )

                        final_answer = fallback_answer
                        final_judge_score = fallback_judge_score
                        final_score_value = fallback_score_value

                        with root_span.start_as_current_observation(
                            as_type="event",
                            name="fallback-applied",
                            input={
                                "example_index": idx,
                                "initial_score": score_value,
                                "fallback_score": fallback_score_value,
                            },
                        ):
                            pass

                    with evaluation_span.start_as_current_observation(
                        as_type="span",
                        name="calculate-final-severity",
                    ) as final_severity_span:
                        if final_score_value <= CRITICAL_SCORE_THRESHOLD:
                            final_severity = "critical"
                        elif final_score_value == 3:
                            final_severity = "review"
                        else:
                            final_severity = "ok"

                        final_severity_span.update(
                            input={"final_score_value": final_score_value},
                            output={"final_severity": final_severity},
                        )

                    total_input_tokens = input_tokens + fallback_input_tokens
                    total_output_tokens = output_tokens + fallback_output_tokens

                    evaluation_result = {
                        "question": question,
                        "expected_answer": expected_answer,
                        "initial_answer": model_answer,
                        "final_answer": final_answer,
                        "initial_judge_score": judge_score,
                        "final_judge_score": final_judge_score,
                        "initial_score_value": score_value,
                        "final_score_value": final_score_value,
                        "used_fallback": used_fallback,
                        "input_tokens": total_input_tokens,
                        "output_tokens": total_output_tokens,
                        "initial_severity": severity,
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
                            "initial_answer": model_answer,
                            "final_answer": final_answer,
                            "initial_judge_score": judge_score,
                            "final_judge_score": final_judge_score,
                            "initial_score_value": score_value,
                            "final_score_value": final_score_value,
                            "used_fallback": used_fallback,
                            "input_tokens": total_input_tokens,
                            "output_tokens": total_output_tokens,
                        },
                        metadata={
                            "category": category,
                            "flags": flags,
                            "initial_severity": severity,
                            "final_severity": final_severity,
                            "mitigation_status": mitigation_status,
                        },
                    )

                    results.append(evaluation_result)
                    time.sleep(0.1)

            with root_span.start_as_current_observation(
                as_type="span",
                name="aggregate-results",
            ) as aggregate_span:
                total_examples = len(results)
                num_fallback_applied = sum(1 for r in results if r["used_fallback"])
                summary = {
                    "num_examples": total_examples,
                    "num_judge_answers": total_examples + num_fallback_applied,
                    "judge_answers_preview": [r["final_judge_score"] for r in results[:3]],
                    "total_input_tokens": sum(r["input_tokens"] for r in results),
                    "total_output_tokens": sum(r["output_tokens"] for r in results),
                    "num_fallback_applied": num_fallback_applied,
                }
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


if __name__ == "__main__":
    batch_result = run_batch_llm_judge_pipeline()
    review_cases = batch_result["review_cases"]

    print(f"Casos totais: {len(batch_result['results'])}")
    print(f"Fallback aplicado: {batch_result['summary']['num_fallback_applied']}")
    print(f"Casos para revisão manual (final_severity != ok): {len(review_cases)}")
    print("Casos prioritários para revisão manual:")

    for item in review_cases:
        print("Pergunta:", item.get("question"))
        print("Score inicial:", item.get("initial_score_value"))
        print("Score final:", item.get("final_score_value"))
        print("Fallback aplicado:", item.get("used_fallback"))
        print("Status mitigação:", item.get("mitigation_status"))
        print("Severidade final:", item.get("final_severity"))
