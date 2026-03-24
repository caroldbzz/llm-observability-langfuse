import os
import json
import time
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
PROJECT_ROOT = BASE_DIR.parent
DATASET_PATH = PROJECT_ROOT / "data" / "bitext_customer_support.csv"
ANSWER_PROMPT_NAME = "customer-support-assistant"
ANSWER_PROMPT_LABEL = "production"
JUDGE_PROMPT_PATH = BASE_DIR / "prompts" / "answer_judge.md"


def load_prompt(path: str | Path) -> str:
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def build_prompt(template: str, variables: dict) -> str:
    prompt = template
    for key, value in variables.items():
        prompt = prompt.replace(f"{{{{{key}}}}}", str(value))
    return prompt


def parse_judge_response(raw_response: str) -> dict:
    try:
        parsed = json.loads(raw_response)
        return {
            "score": parsed.get("score"),
            "reason": parsed.get("reason"),
        }
    except Exception:
        return {
            "score": None,
            "reason": raw_response,
        }


def run_llm_judge_evaluation():
    with langfuse.start_as_current_observation(
        as_type="span",
        name="llm-judge-evaluation-pipeline",
    ) as root_span:

        with propagate_attributes(session_id="evaluation-session-003"):

            root_span.update(
                user_id="demo-user-alura",
                tags=["llm-as-judge", OPENAI_MODEL, JUDGE_MODEL],
                metadata={
                    "dataset": DATASET_PATH,
                    "evaluation_type": "open-answer-judge",
                    "judge_prompt_path": JUDGE_PROMPT_PATH,
                    "answer_prompt_name": ANSWER_PROMPT_NAME,
                    "answer_prompt_label": ANSWER_PROMPT_LABEL,
                },
            )

            with root_span.start_as_current_observation(
                as_type="span",
                name="load-dataset-example",
            ) as dataset_span:

                df = pd.read_csv(DATASET_PATH)
                row = df.iloc[0]

                question = row["instruction"]
                expected_answer = row["response"]
                category = row["category"]
                flags = row["flags"]

                dataset_span.update(
                    output={
                        "question": question,
                        "expected_answer": expected_answer,
                    },
                    metadata={
                        "category": category,
                        "flags": flags,
                    },
                )

                time.sleep(0.2)

            with root_span.start_as_current_observation(
                as_type="span",
                name="get-answer-prompt",
            ) as answer_prompt_span:

                prompt_client = langfuse.get_prompt(
                    name=ANSWER_PROMPT_NAME,
                    label=ANSWER_PROMPT_LABEL,
                )

                system_prompt = prompt_client.prompt

                answer_prompt_span.update(
                    output={
                        "prompt_name": ANSWER_PROMPT_NAME,
                        "prompt_label": ANSWER_PROMPT_LABEL,
                    }
                )

            with root_span.start_as_current_observation(
                as_type="span",
                name="generate-answer",
            ) as answer_span:

                completion = openai.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ],
                    name="customer-support-answer",
                )

                model_answer = completion.choices[0].message.content.strip()

                answer_span.update(
                    input={"question": question},
                    output={"model_answer": model_answer},
                )

                time.sleep(0.1)

            with root_span.start_as_current_observation(
                as_type="span",
                name="load-judge-prompt",
            ) as judge_prompt_span:

                judge_template = load_prompt(JUDGE_PROMPT_PATH)

                judge_prompt_span.update(
                    output={
                        "judge_prompt_path": JUDGE_PROMPT_PATH,
                        "judge_prompt_preview": judge_template[:200],
                    }
                )

            with root_span.start_as_current_observation(
                as_type="span",
                name="judge-answer",
            ) as judge_span:

                final_judge_prompt = build_prompt(
                    judge_template,
                    {
                        "question": question,
                        "expected_answer": expected_answer,
                        "model_answer": model_answer,
                    },
                )

                judge_completion = openai.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[
                        {"role": "system", "content": "Você é um avaliador de respostas de atendimento ao cliente."},
                        {"role": "user", "content": final_judge_prompt},
                    ],
                    name="llm-answer-judge",
                )

                raw_judge_response = judge_completion.choices[0].message.content.strip()
                judge_result = parse_judge_response(raw_judge_response)

                judge_span.update(
                    input={
                        "question": question,
                        "expected_answer": expected_answer,
                        "model_answer": model_answer,
                    },
                    output={
                        "judge_score": judge_result["score"],
                        "judge_reason": judge_result["reason"],
                        "raw_judge_response": raw_judge_response,
                    },
                    metadata={
                        "judge_prompt_path": JUDGE_PROMPT_PATH,
                    },
                )

                time.sleep(0.1)

            root_span.update(
                input={"question": question},
                output={
                    "expected_answer": expected_answer,
                    "model_answer": model_answer,
                    "judge_score": judge_result["score"],
                    "judge_reason": judge_result["reason"],
                },
                metadata={
                    "category": category,
                    "flags": flags,
                },
            )

        langfuse.flush()

        return {
            "question": question,
            "expected_answer": expected_answer,
            "model_answer": model_answer,
            "judge_score": judge_result["score"],
            "judge_reason": judge_result["reason"],
        }


if __name__ == "__main__":
    result = run_llm_judge_evaluation()

    print("\nPergunta:\n")
    print(result["question"])

    print("\nResposta esperada do dataset:\n")
    print(result["expected_answer"])

    print("\nResposta gerada pelo modelo:\n")
    print(result["model_answer"])

    print("\nScore do juiz:\n")
    print(result["judge_score"])

    print("\nJustificativa do juiz:\n")
    print(result["judge_reason"])
