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
PROMPT_NAME = "customer-support-assistant"

PROMPT_LABELS = ["baseline", "improved"]


def run_prompt_experiment():
    with langfuse.start_as_current_observation(
        as_type="span",
        name="prompt-experiment",
    ) as root_span:

        with propagate_attributes(session_id="experiment-session-001"):

            root_span.update(
                user_id="demo-user-alura",
                tags=["prompt-experiment", OPENAI_MODEL],
                metadata={
                    "experiment": "prompt-comparison",
                    "dataset": DATASET_PATH,
                },
            )

            df = pd.read_csv(DATASET_PATH)
            question = df.iloc[4]["instruction"]

            results = []

            for label in PROMPT_LABELS:

                with root_span.start_as_current_observation(
                    as_type="span",
                    name=f"run-{label}",
                ) as experiment_span:

                    prompt_client = langfuse.get_prompt(
                        name=PROMPT_NAME,
                        label=label,
                    )

                    system_prompt = prompt_client.prompt

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
                        input={"question": question},
                        output={"answer": answer},
                        metadata={
                            "prompt_label": label,
                            "prompt_version": prompt_client.version,
                        },
                    )

                    results.append(
                        {
                            "label": label,
                            "answer": answer,
                        }
                    )

            root_span.update(
                output={"comparison": results}
            )

        langfuse.flush()

        return results


if __name__ == "__main__":
    results = run_prompt_experiment()

    print("\nResultados do experimento:\n")

    for r in results:
        print(f"[{r['label']}]")
        print(r["answer"])
        print("-" * 40)