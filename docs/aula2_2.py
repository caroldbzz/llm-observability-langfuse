import pandas as pd
from dotenv import load_dotenv

from langfuse import get_client
from langfuse.openai import openai

load_dotenv()

langfuse = get_client()

OPENAI_MODEL = "gpt-4o-mini"
DATASET_PATH = "docs/data/bitext_customer_support.csv"


def get_customer_question() -> str:
    df = pd.read_csv(DATASET_PATH)
    question = df.iloc[0]["instruction"]
    return question


def ask_llm(question: str) -> str:
    with langfuse.start_as_current_observation(
        as_type="span",
        name="customer-support-demo",
        input={"question": question},
    ) as root_span:

        with root_span.start_as_current_observation(
            as_type="event",
            name="dataset-loaded",):
            pass


        completion = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Você é um assistente de atendimento ao cliente. Responda de forma clara, objetiva e educada.",
                },
                {"role": "user", "content": question},
            ],
            name="openai-generation-step",
        )

        answer = completion.choices[0].message.content

        root_span.update(
            input={"question": question},
            output={"answer": answer},
            )


    langfuse.flush()

    return answer


if __name__ == "__main__":
    pergunta_dataset = get_customer_question()

    print(f"\nQuestão selecionada: '{pergunta_dataset}'")

    resposta = ask_llm(pergunta_dataset)

    print(f"\nResposta do Suporte:\n{resposta}")
