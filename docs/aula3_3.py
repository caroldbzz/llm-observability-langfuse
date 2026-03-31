import pandas as pd
from dotenv import load_dotenv
from langfuse import get_client


from langfuse import propagate_attributes
from langfuse.openai import openai


load_dotenv()


langfuse = get_client()


DATASET_PATH = "data/bitext_customer_support.csv"
OPENAI_MODEL = "gpt-4o-mini"
PROMPT_FILES = {
    "baseline": "prompts/customer_support_baseline.md",
    "improved": "prompts/customer_support_improved.md",
}


def load_prompt(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def get_customer_question():
    df = pd.read_csv(DATASET_PATH)
    question = df.iloc[4]["instruction"]

    return question


def run_prompt_file_experiment():
    with langfuse.start_as_current_observation(
        as_type="span",
        name="prompt-file-experiment"
    ) as root_span:
       
        with propagate_attributes(
            session_id="prompt-session-123",
            user_id="id-123",
            tags=["customer-support", "prompt-file-experiment", OPENAI_MODEL],
            metadata={
                "environment": "develop",
                "version": "V1",
                "dataset_path": DATASET_PATH,
                "experiment_type": "file-based-prompts",
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


            for label, prompt_path in PROMPT_FILES.items():
                with root_span.start_as_current_observation(
                    as_type="span",
                    name=f"run-{label}"
                ) as expertiment_span:
                   
                    system_prompt = load_prompt(prompt_path)

                    completion = openai.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": cleaned_question},
                        ],
                        name=f"llm-generation-{label}",
                    )
               
                    answer = completion.choices[0].message.content


                    expertiment_span.update(
                        input={"question": cleaned_question},
                        output={"answer": answer},
                        metadata={
                            "prompt_source": "file",
                            "prompt_label": label,
                            "prompt_path": prompt_path,
                        })
               
                    results.append({"question": cleaned_question,
                                    "label": label,
                                    "prompt_path": prompt_path,
                                    "answer": answer})
               
        root_span.update(metadata={"comparison": results})
   
    langfuse.flush()


    return results




if __name__ == "__main__":
    results = run_prompt_file_experiment()

    for item in results:
        print(item)