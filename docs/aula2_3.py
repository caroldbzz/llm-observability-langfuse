import time
import pandas as pd
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.openai import openai

load_dotenv()

langfuse = get_client()

DATASET_PATH = "docs/data/bitext_customer_support.csv"
OPENAI_MODEL = "gpt-4o-mini"

def get_customer_question():
    df = pd.read_csv(DATASET_PATH)
    question = df.iloc[0]["instruction"]
    return question 

def run_customer_support_pipeline():
    with langfuse.start_as_current_observation(
        as_type="span",
        name="customer-support-pipeline"
    ) as root_span:
        
        with root_span.start_as_current_observation(
            as_type="span",
            name = "load_dataset",
        ) as dataset_span:
            
            question = get_customer_question()
            dataset_span.update(input={"dataset_path": DATASET_PATH},
                output={"selected_question": question})
        

        time.sleep(0.2)

        with root_span.start_as_current_observation(
            as_type="event",
            name="dataset-loaded",
        ):
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
            },)
            time.sleep(0.1)

       
        completion = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Você é um assistente de atendimento ao cliente. Responda de forma objetiva e educada."},
                {"role": "user", "content": cleaned_question},
            ],
            name="llm-generation",
        )

        answer = completion.choices[0].message.content  
        root_span.update(input={"question": cleaned_question},
            output={"answer": answer,})
    
    langfuse.flush()

    return answer


if __name__ == "__main__":
   resposta = run_customer_support_pipeline()
   print("Resposta do Suporte: ")
   print(resposta)
