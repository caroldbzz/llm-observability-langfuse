import os
import time
from dotenv import load_dotenv

from langfuse import get_client
from langfuse.openai import openai

load_dotenv()

langfuse = get_client()
openai.api_key = os.getenv("OPENAI_API_KEY")


def ask_llm(question: str) -> str:
    # ROOT SPAN = início do trace
    with langfuse.start_as_current_observation(
        as_type="span",
        name="chat-request",
        input={"question": question},
    ) as root_span:

        # TRACE-level configuration
        root_span.update_trace(
            name="demo-langfuse-chat",
            user_id="demo-user",
            tags=["demo", "openai", "langfuse", "chat"],
            metadata={
                "environment": "local-demo",
                "feature": "basic-chat",
            },
        )

        # SPAN de pré-processamento (etapa com duração)
        with root_span.start_as_current_observation(
            as_type="span",
            name="pre-processing",
        ) as preprocessing_span:

            preprocessing_span.update(
                metadata={
                    "question_length_chars": len(question),
                    "question_word_count": len(question.split()),
                }
            )

            time.sleep(0.2)

        # EVENT (marco pontual, sem duração)
        with root_span.start_as_current_observation(
            as_type="event",
            name="input-received",
        ):
            pass

        # GENERATION (LLM call instrumentada automaticamente)
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um assistente objetivo e didático."},
                {"role": "user", "content": question},
            ],
            name="openai-chat",
        )

        answer = completion.choices[0].message.content or ""

        # Atualiza root span com output consolidado
        root_span.update(output={"answer": answer})

    # Garante envio das observações antes do script encerrar
    langfuse.flush()

    return answer


if __name__ == "__main__":
    q = input("Pergunte algo: ").strip()
    print("\nResposta:\n")
    print(ask_llm(q))