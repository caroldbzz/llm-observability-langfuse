import pandas as pd
from dotenv import load_dotenv

from langfuse import get_client, propagate_attributes
from langfuse.openai import openai

load_dotenv()

langfuse = get_client()

OPENAI_MODEL = "gpt-4o-mini"
DATASET_PATH = "docs/data/bitext_customer_support.csv"

PROMPT_FILES = {
    "baseline": "docs/prompts/customer_support_baseline.md",
    "improved": "docs/prompts/customer_support_improved.md",
}

def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


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

        with propagate_attributes(
            session_id="experiment-session-002",
            user_id="demo-user-alura",
            tags=["prompt-structured-comparison", OPENAI_MODEL],
            metadata={
                "experiment": "prompt-ranking",
                "dataset": str(DATASET_PATH),
            },
        ):
            with root_span.start_as_current_observation(
                as_type="span",
                name="load-dataset-example",
            ) as dataset_span:
                df = pd.read_csv(DATASET_PATH)
                row = df.iloc[4]

                question = row["instruction"]
                expected_answer = row["response"]
                category = row["category"]
                intent = row["intent"]

                dataset_span.update(
                    output={
                        "question": question,
                        "expected_answer": expected_answer,
                    },
                    metadata={
                        "category": category,
                        "intent": intent,
                    },
                )

            results = []

            for label, prompt_path in PROMPT_FILES.items():
                with root_span.start_as_current_observation(
                    as_type="span",
                    name=f"run-{label}",
                ) as experiment_span:

                    system_prompt = load_prompt(prompt_path)

                    completion = openai.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question},
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
