import os
from dotenv import load_dotenv

from langfuse import get_client
from langfuse.openai import openai

load_dotenv()

langfuse = get_client()
openai.api_key = os.getenv("OPENAI_API_KEY")


def ask_llm(question: str) -> str:
    with langfuse.start_as_current_observation(
        as_type="span",
        name="chat-request",
        input={"question": question},
    ) as root_span:


        with root_span.start_as_current_observation(
            as_type="event",
            name="input-received",
        ):
            pass

        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um assistente objetivo e didático."},
                {"role": "user", "content": question},
            ],
            name="openai-chat",
        )

        answer = completion.choices[0].message.content 

        root_span.update(output={"answer": answer})

    langfuse.flush()

    return answer


if __name__ == "__main__":
    q = input("Pergunte algo: ").strip()
    print("\nResposta:\n")
    print(ask_llm(q))