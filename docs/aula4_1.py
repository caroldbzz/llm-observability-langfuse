import pandas as pd
from dotenv import load_dotenv
from langfuse import get_client


from langfuse import propagate_attributes
from langfuse.openai import openai


load_dotenv()


langfuse = get_client()


DATASET_PATH = "data/bitext_customer_support.csv"
OPENAI_MODEL = "gpt-4o-mini"
# PROMPT_FILES = {
#     "baseline": "prompts/customer_support_baseline.md",
#     "improved": "prompts/customer_support_improved.md",
# }
PROMPT_PATH = "prompts/intent_classifier.md"


def load_prompt(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def get_customer_question():
    df = pd.read_csv(DATASET_PATH)
    row = df.iloc[4]
    question = row["instruction"]
    true_intent = row["intent"]
    category = row["category"]

    return question, true_intent, category


def run_intent_evaluation():
    with langfuse.start_as_current_observation(
        as_type="span",
        name="intent-evaluation-pipeline"
    ) as root_span:
       
        with propagate_attributes(
            session_id="intent-evaluation-123",
            user_id="id-123",
            tags=["customer-support", "intent-evaluation", OPENAI_MODEL],
            metadata={
                "environment": "develop",
                "version": "V1",
                "dataset_path": DATASET_PATH,
            }
        ):
       
            with root_span.start_as_current_observation(
                as_type="span",
                name="load-dataset"
            ) as dataset_span:
               
                question, true_intent, category = get_customer_question()
               
                dataset_span.update(
                    output={
                        "selected_question": question,
                        "true_intent": true_intent},
                    metadata={
                        "category": category
                        })
           
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
            
            with root_span.start_as_current_observation(
                as_type="span",
                name="predict-intent"
            ) as predict_span:
            
                system_prompt = load_prompt(PROMPT_PATH)

                completion = openai.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": cleaned_question},
                    ],
                    name="intent-classification",
                )
        
                predicted_intent = completion.choices[0].message.content

                predict_span.update(
                    input={"question": cleaned_question,},
                    output={"predicted_intent": predicted_intent},
                    metadata={
                        "prompt_path": PROMPT_PATH,
                    })
        
            with root_span.start_as_current_observation(
                as_type="span",
                name="evaluate-intent",
            ) as eval_span:
                
                is_correct = predicted_intent == true_intent.lower()
                score = 1 if is_correct else 0

                eval_span.update(
                    input={"true_intent": true_intent,
                           "predicted_intent": predicted_intent},
                    output={"is_correct": is_correct, 
                            "score": score},
                )
               
        root_span.update(input={"question": cleaned_question},
                         output={"true_intent": true_intent,
                                 "predicted_intent": predicted_intent,  
                                 "score": score},
                        metadata={
                            "category": category,
                            "prompt_path": PROMPT_PATH,
                        })
   
    langfuse.flush()


    return {"question": cleaned_question,
            "true_intent": true_intent,
            "predicted_intent": predicted_intent,
            "score": score}




if __name__ == "__main__":
    results = run_intent_evaluation()

    print("Pergunta:", results["question"])
    print("Intenção esperada:", results["true_intent"])
    print("Intenção prevista:", results["predicted_intent"])
    print("Score:", results["score"])