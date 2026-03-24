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
PROMPT_LABEL = "production"


def run_customer_support_pipeline() -> str:
    with langfuse.start_as_current_observation(
        as_type="span",
        name="customer-support-pipeline",
    ) as root_span:

        with propagate_attributes(session_id="chat-session-123"):

            root_span.update(
                user_id="demo-user-alura",
                tags=["customer-support", "prompt-versioning", OPENAI_MODEL],
                metadata={
                    "environment": "development",
                    "dataset_source": DATASET_PATH,
                    "pipeline_version": "v2",
                },
            )

            with root_span.start_as_current_observation(
                as_type="span",
                name="load-dataset",
            ) as dataset_span:

                df = pd.read_csv(DATASET_PATH)
                question = df.iloc[0]["instruction"]

                dataset_span.update(
                    output={"selected_question": question},
                    metadata={"dataset_path": DATASET_PATH},
                )

                time.sleep(0.2)

            with root_span.start_as_current_observation(
                as_type="span",
                name="text-preprocessing",
            ) as preprocess_span:

                cleaned_question = question.strip()

                preprocess_span.update(
                    input={"raw_question": question},
                    output={"cleaned_question": cleaned_question},
                    metadata={
                        "char_count": len(cleaned_question),
                        "is_empty": len(cleaned_question) == 0,
                    },
                )

                time.sleep(0.1)

            with root_span.start_as_current_observation(
                as_type="span",
                name="get-prompt-version",
            ) as prompt_span:

                prompt_client = langfuse.get_prompt(
                    name=PROMPT_NAME,
                    label=PROMPT_LABEL,
                )

                system_prompt = prompt_client.prompt

                prompt_span.update(
                    output={
                        "prompt_name": PROMPT_NAME,
                        "prompt_label": PROMPT_LABEL,
                        "system_prompt": system_prompt,
                    }
                )

            completion = openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": cleaned_question},
                ],
                name="llm-generation",
            )

            answer = completion.choices[0].message.content

            root_span.update(
                input={"question": question},
                output={"answer": answer},
                metadata={
                    "environment": "development",
                    "dataset_source": DATASET_PATH,
                    "pipeline_version": "v2",
                    "prompt_name": PROMPT_NAME,
                    "prompt_label": PROMPT_LABEL,
                },
            )

        langfuse.flush()

        return answer


if __name__ == "__main__":
    resposta = run_customer_support_pipeline()

    print("\nResposta do Suporte:\n")
    print(resposta)