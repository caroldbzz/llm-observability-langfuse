import time
import pandas as pd
from dotenv import load_dotenv

from langfuse import get_client
from langfuse.openai import openai

load_dotenv()

langfuse = get_client()

OPENAI_MODEL = "gpt-4o-mini"
DATASET_PATH = "docs/data/bitext_customer_support.csv"


def run_customer_support_pipeline() -> str:
    with langfuse.start_as_current_observation(
        as_type="span",
        name="customer-support-pipeline",
    ) as root_span:

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
            as_type="event",
            name="dataset-loaded",):
            pass
        
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

        completion = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Você é um assistente de atendimento ao cliente. Responda de forma clara, objetiva e educada.",
                },
                {"role": "user", "content": cleaned_question},
            ],
            name="llm-generation",
        )

        answer = completion.choices[0].message.content

        root_span.update(
            input={"question": question},
            output={"answer": answer},
        )

    langfuse.flush()

    return answer


if __name__ == "__main__":
    resposta = run_customer_support_pipeline()

    print("\nResposta do Suporte:\n")
    print(resposta)