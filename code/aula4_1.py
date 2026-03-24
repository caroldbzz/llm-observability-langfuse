import os
import time
import pandas as pd
from dotenv import load_dotenv

from langfuse import get_client, propagate_attributes
from langfuse.openai import openai

load_dotenv()

langfuse = get_client()
openai.api_key = os.getenv("OPENAI_API_KEY")

OPENAI_MODEL = "gpt-4o-mini"
DATASET_PATH = "../data/bitext_customer_support.csv"
PROMPT_PATH = "prompts/intent_classifier.md"


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def build_prompt(template: str, variables: dict) -> str:
    prompt = template
    for key, value in variables.items():
        prompt = prompt.replace(f"{{{{{key}}}}}", str(value))
    return prompt


def run_intent_evaluation():
    with langfuse.start_as_current_observation(
        as_type="span",
        name="intent-evaluation-pipeline",
    ) as root_span:

        with propagate_attributes(session_id="evaluation-session-002"):

            root_span.update(
                user_id="demo-user-alura",
                tags=["intent-evaluation", OPENAI_MODEL],
                metadata={
                    "dataset": DATASET_PATH,
                    "evaluation_type": "intent-classification",
                    "prompt_path": PROMPT_PATH,
                },
            )


            with root_span.start_as_current_observation(
                as_type="span",
                name="load-dataset-example",
            ) as dataset_span:

                df = pd.read_csv(DATASET_PATH)
                row = df.iloc[0]

                question = row["instruction"]
                true_intent = row["intent"]
                category = row["category"]
                flags = row["flags"]

                dataset_span.update(
                    output={
                        "question": question,
                        "true_intent": true_intent,
                    },
                    metadata={
                        "category": category,
                        "flags": flags,
                    },
                )


            with root_span.start_as_current_observation(
                as_type="span",
                name="load-prompt-template",
            ) as prompt_span:

                prompt_template = load_prompt(PROMPT_PATH)

                prompt_span.update(
                    output={
                        "prompt_path": PROMPT_PATH,
                        "prompt_preview": prompt_template[:200],
                    }
                )


            with root_span.start_as_current_observation(
                as_type="span",
                name="predict-intent",
            ) as predict_span:

                final_prompt = build_prompt(
                    prompt_template,
                    {"question": question},
                )

                completion = openai.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "Você é um classificador de intenção."},
                        {"role": "user", "content": final_prompt},
                    ],
                    name="intent-classification",
                )

                predicted_intent = completion.choices[0].message.content.strip()

                predict_span.update(
                    input={"question": question},
                    output={"predicted_intent": predicted_intent},
                    metadata={
                        "prompt_path": PROMPT_PATH,
                    },
                )

            with root_span.start_as_current_observation(
                as_type="span",
                name="evaluate-intent",
            ) as eval_span:

                is_correct = predicted_intent == true_intent
                score = 1 if is_correct else 0

                eval_span.update(
                    input={
                        "true_intent": true_intent,
                        "predicted_intent": predicted_intent,
                    },
                    output={
                        "is_correct": is_correct,
                        "score": score,
                    },
                )

                time.sleep(0.1)

            root_span.update(
                input={"question": question},
                output={
                    "true_intent": true_intent,
                    "predicted_intent": predicted_intent,
                    "score": score,
                },
                metadata={
                    "category": category,
                    "flags": flags,
                    "prompt_path": PROMPT_PATH,
                },
            )

        langfuse.flush()

        return {
            "question": question,
            "true_intent": true_intent,
            "predicted_intent": predicted_intent,
            "score": score,
        }


if __name__ == "__main__":
    result = run_intent_evaluation()

    print("\nPergunta:\n", result["question"])
    print("\nIntent esperada:", result["true_intent"])
    print("Intent prevista:", result["predicted_intent"])
    print("\nScore:", result["score"])