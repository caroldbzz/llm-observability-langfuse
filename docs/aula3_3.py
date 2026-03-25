import time
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


def run_file_prompt_experiment():
    with langfuse.start_as_current_observation(
        as_type="span",
        name="file-based-prompt-experiment",
    ) as root_span:

        with propagate_attributes(
            session_id="prompt-file-session-001",
            user_id="demo-user-alura",
            tags=["prompt-files", "prompt-experiment", OPENAI_MODEL],
            metadata={
                "dataset_source": DATASET_PATH,
                "experiment_type": "file-based-prompts",
                "pipeline_version": "v1",
            },
        ):

            with root_span.start_as_current_observation(
                as_type="span",
                name="load-dataset-example",
            ) as dataset_span:

                df = pd.read_csv(DATASET_PATH)
                row = df.iloc[0]

                question = row["instruction"]
                expected_answer = row["response"]
                category = row["category"]
                intent = row["intent"]
                flags = row["flags"]

                dataset_span.update(
                    output={
                        "question": question,
                        "expected_answer": expected_answer,
                    },
                    metadata={
                        "category": category,
                        "intent": intent,
                        "flags": flags,
                    },
                )

                time.sleep(0.2)

            results = []

            for label, prompt_path in PROMPT_FILES.items():
                with root_span.start_as_current_observation(
                    as_type="span",
                    name=f"run-{label}",
                ) as experiment_span:

                    system_prompt = load_prompt(prompt_path)

                    experiment_span.update(
                        input={
                            "question": question,
                        },
                        metadata={
                            "prompt_source": "file",
                            "prompt_label": label,
                            "prompt_path": prompt_path,
                            "prompt_template_preview": system_prompt[:250],
                        },
                    )

                    completion = openai.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question},
                        ],
                        name=f"llm-generation-{label}",
                    )

                    answer = completion.choices[0].message.content

                    experiment_span.update(
                        output={
                            "answer": answer,
                        }
                    )

                    results.append(
                        {
                            "prompt_label": label,
                            "prompt_path": prompt_path,
                            "answer": answer,
                        }
                    )

            root_span.update(
                input={"question": question},
                output={
                    "expected_answer": expected_answer,
                    "results": results,
                },
                metadata={
                    "category": category,
                    "intent": intent,
                    "flags": flags,
                },
            )

    langfuse.flush()

    return {
        "question": question,
        "expected_answer": expected_answer,
        "results": results,
    }


if __name__ == "__main__":
    result = run_file_prompt_experiment()

    print("\nPergunta:\n")
    print(result["question"])

    print("\nResposta esperada do dataset:\n")
    print(result["expected_answer"])

    print("\nResultados por prompt:\n")
    for item in result["results"]:
        print(f"[{item['prompt_label']}] - {item['prompt_path']}")
        print(item["answer"])
