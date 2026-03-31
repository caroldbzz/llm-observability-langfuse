import pandas as pd
from dotenv import load_dotenv
from langfuse import get_client


from langfuse import propagate_attributes
from langfuse.openai import openai


load_dotenv()


langfuse = get_client()


DATASET_PATH = "data/bitext_customer_support.csv"

OPENAI_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4.1-mini"

JUDGE_PROMPT_PATH = "prompts/answer_judge.md"
ANSWER_PROMPT_NAME = "customer_support_assistant"
ANSWER_PROMPT_LABEL = "production"


def load_prompt(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def get_customer_question():
    df = pd.read_csv(DATASET_PATH)
    row = df.iloc[4]
    question = row["instruction"]
    expected_answer = row["response"]
    category = row["category"]

    return question, expected_answer, category


def run_llm_judge_evaluation():
    with langfuse.start_as_current_observation(
        as_type="span",
        name="llm-judge-evaluation-pipeline"
    ) as root_span:
       
        with propagate_attributes(
            session_id="evaluation-evaluation-123",
            user_id="id-123",
            tags=["customer-support", "llm-as-judge", OPENAI_MODEL],
            metadata={
                "dataset": DATASET_PATH,
                "evaluation_type": "open-answer-judge",
                "judge_prompt_path": JUDGE_PROMPT_PATH,
                "answer_prompt_name": ANSWER_PROMPT_NAME,
                "answer_prompt_label": ANSWER_PROMPT_LABEL,
            }
        ):
       
            with root_span.start_as_current_observation(
                as_type="span",
                name="load-dataset"
            ) as dataset_span:
               
                question, expected_answer, category = get_customer_question()
               
                dataset_span.update(
                    output={
                        "selected_question": question,
                        "expected_answer": expected_answer},
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
                name="get-answer-prompt",
            ) as answer_prompt_span:

                prompt_client = langfuse.get_prompt(
                    name=ANSWER_PROMPT_NAME,
                    label=ANSWER_PROMPT_LABEL,
                )

                system_prompt = prompt_client.prompt

                answer_prompt_span.update(
                    output={
                        "prompt_name": ANSWER_PROMPT_NAME,
                        "prompt_label": ANSWER_PROMPT_LABEL,
                    }
                )

            with root_span.start_as_current_observation(
                as_type="span",
                name="generate-answer"
            ) as answer_span:

                completion = openai.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": cleaned_question},
                    ],
                    name="customer-support-answer",
                )
        
                model_answer = completion.choices[0].message.content

                answer_span.update(
                    input={"question": cleaned_question,},
                    output={"model_answer": model_answer})
        
            with root_span.start_as_current_observation(
                as_type="span",
                name="load-judge-prompt",
            ) as judge_prompt_span:
                
                judge_prompt = load_prompt(JUDGE_PROMPT_PATH)

                judge_prompt_span.update(
                    output={
                        "judge_prompt_path": JUDGE_PROMPT_PATH,
                    }
                )
            
            with root_span.start_as_current_observation(
                as_type="span",
                name="judge-answer",
            ) as judge_span:
                
                judge_input = (
                    f"Pergunta: {cleaned_question}\n\n"
                    f"Resposta esperada: {expected_answer}\n\n"
                    f"Resposta do modelo: {model_answer}"
                )

                judge_completion = openai.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[
                        {"role": "system", "content": judge_prompt},
                        {"role": "user", "content": judge_input},
                    ],
                    name="llm-answer-judge",
                )

                judge_score = judge_completion.choices[0].message.content

                judge_span.update(
                    input={ "question": question,
                            "expected_answer": expected_answer,
                            "model_answer": model_answer,},
                    output={"judge_score": judge_score}
                )
               
        root_span.update(input={"question": cleaned_question},
                         output={"expected_answer": expected_answer,
                                 "model_answer": model_answer,
                                 "judge_score": judge_score},
                        metadata={
                            "category": category,
                        })
   
    langfuse.flush()


    return {"question": cleaned_question,
            "expected_answer": expected_answer,
            "model_answer": model_answer,
            "judge_score": judge_score}




if __name__ == "__main__":
    results = run_llm_judge_evaluation()

    print("Pergunta:", results["question"])
    print("Resposta esperada:", results["expected_answer"])
    print("Resposta do modelo:", results["model_answer"])
    print("Score:", results["judge_score"])
