import pandas as pd
from dotenv import load_dotenv
from langfuse import get_client


from langfuse import propagate_attributes
from langfuse.openai import openai


load_dotenv()


langfuse = get_client()


DATASET_PATH = "data/bitext_customer_support.csv"
OPENAI_MODEL = "gpt-4o-mini"
PROMPT_LABELS= ["baseline", "improved"]
PROMPT_NAME = "customer_support_assistant"




def get_customer_question():
    df = pd.read_csv(DATASET_PATH)
    question = df.iloc[4]["instruction"]


    return question


def run_prompt_experiment():
    with langfuse.start_as_current_observation(
        as_type="span",
        name="prompt-experiment"
    ) as root_span:
       
        with propagate_attributes(
            session_id="chat-session-123",
            user_id="id-123",
            tags=["customer-support", "prompt-experiment", OPENAI_MODEL],
            metadata={
                "environment": "develop",
                "version": "V1",
                "dataset_path": DATASET_PATH,
            }
        ):
       
            with root_span.start_as_current_observation(
                as_type="span",
                name="load_dataset"
            ) as dataset_span:
               
                question = get_customer_question()
                dataset_span.update(input= {"dataset_path": DATASET_PATH},
                output={"selected_question": question})
           
            with root_span.start_as_current_observation(
                as_type="event",
                name="dataset-loaded"
            ):
                pass


            with root_span.start_as_current_observation(
                as_type="span",
                name="preprocessing-question"
            ) as preprocessing_span:
               
                cleaned_question = question.strip()
                preprocessing_span.update(input= {"question": question},
                output={"cleaned_question": cleaned_question},
                metadata={"len_question": len(cleaned_question)})


            results = []


            for label in PROMPT_LABELS:
                with root_span.start_as_current_observation(
                    as_type="span",
                    name=f"run-{label}"
                ) as expertiment_span:
                   
                    prompt_client = langfuse.get_prompt(
                        name=PROMPT_NAME,
                        label=label
                    )


                    system_prompt = prompt_client.prompt


                completion = openai.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "system", "content": system_prompt},
                            {"role": "user", "content": cleaned_question}],
                    name=f"llm-generation-{label}"
                    )
               
                answer = completion.choices[0].message.content


                expertiment_span.update(
                    input= {"question": question},
                    output={"answer": answer},
                    metadata={"PROMPT_LABEL": label,
                            "PROMPT_VERSION": prompt_client.version})
               
                results.append({"label": label,
                                "answer": answer})
               
    root_span.update(metadata={"comparison": results})
   
    langfuse.flush()


    return answer




if __name__ == "__main__":
    results = run_prompt_experiment()
   
    