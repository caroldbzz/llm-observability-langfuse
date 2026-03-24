import os
import pandas as pd
from dotenv import load_dotenv

from langfuse import get_client, propagate_attributes
from langfuse.openai import openai

load_dotenv()

langfuse = get_client()
openai.api_key = os.getenv("OPENAI_API_KEY")

OPENAI_MODEL = "gpt-4o-mini"
DATASET_PATH = "../data/bitext_customer_support.csv"

PROMPT_FILES = {
    "baseline": "prompts/customer_support_baseline.md",
    "improved": "prompts/customer_support_improved.md",
}


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def build_prompt(template: str, variables: dict) -> str:
    prompt = template
    for key, value in variables.items():
        prompt = prompt.replace(f"{{{{{key}}}}}", str(value))
    return prompt


def simple_score(answer: str, expected_answer: str) -> dict:
    return {
        "length": len(answer),
        "has_steps": "passo" in answer.lower(),
        "is_long": len(answer) > 200,
        "mentions_expected": any(
            word in answer.lower()
            for word in expected_answer.lower().split()[:5]
        ),
    }


def run_prompt_experiment():
    with langfuse.start_as_current_observation(
        as_type="span",
        name="prompt-comparison-structured",
    ) as root_span:

        with propagate_attributes(session_id="experiment-session-002"):

            root_span.update(
                user_id="demo-user-alura",
                tags=["prompt-structured-comparison", OPENAI_MODEL],
                metadata={
                    "experiment": "prompt-ranking",
                    "dataset": DATASET_PATH,
                },
            )

            df = pd.read_csv(DATASET_PATH)
            row = df.iloc[4]

            question = row["instruction"]
            expected_answer = row["response"]

            results = []

            for label, prompt_path in PROMPT_FILES.items():

                with root_span.start_as_current_observation(
                    as_type="span",
                    name=f"run-{label}",
                ) as experiment_span:

                    prompt_template = load_prompt(prompt_path)

                    final_prompt = build_prompt(
                        prompt_template,
                        {"question": question},
                    )

                    completion = openai.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": "Você é um assistente de atendimento ao cliente."},
                            {"role": "user", "content": final_prompt},
                        ],
                        name=f"llm-generation-{label}",
                    )

                    answer = completion.choices[0].message.content

                    score = simple_score(answer, expected_answer)

                    experiment_span.update(
                        input={
                            "question": question,
                            "expected_answer": expected_answer,
                        },
                        output={"answer": answer},
                        metadata={
                            "prompt_label": label,
                            "prompt_path": prompt_path,
                            "score": score,
                        },
                    )

                    results.append({
                        "label": label,
                        "answer": answer,
                        "expected_answer": expected_answer,
                        "score": score,
                    })

            root_span.update(
                output={
                    "question": question,
                    "expected_answer": expected_answer,
                    "comparison": results,
                }
            )

        langfuse.flush()

        return results


if __name__ == "__main__":
    results = run_prompt_experiment()

    print("\nComparação estruturada dos prompts:\n")

    for r in results:
        print(f"[{r['label']}]")
        print(r["answer"])
        print("Expected answer:", r["expected_answer"])
        print("Score:", r["score"])
        print("-" * 40)