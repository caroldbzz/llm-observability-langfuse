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

N_EXAMPLES = 5


def load_prompt(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


# def get_customer_question():
#     df = pd.read_csv(DATASET_PATH)
#     row = df.iloc[4]
#     question = row["instruction"]
#     expected_answer = row["response"]
#     category = row["category"]

#     return question, expected_answer, category


def run_batch_llm_judge_evaluation():
    with langfuse.start_as_current_observation(
        as_type="span",
        name="batch-llm-judge-evaluation"
    ) as root_span:
       
        with propagate_attributes(
            session_id="evaluation-session-123",
            user_id="id-123",
            tags=["customer-support", "llm-as-judge-batch", OPENAI_MODEL],
            metadata={
                "dataset": DATASET_PATH,
                "evaluation_type": "batch-open-answer-judge",
                "judge_prompt_path": JUDGE_PROMPT_PATH,
                "answer_prompt_name": ANSWER_PROMPT_NAME,
                "answer_prompt_label": ANSWER_PROMPT_LABEL,
                "num_examples": str(N_EXAMPLES),
            }
        ):
       
            with root_span.start_as_current_observation(
                as_type="span",
                name="load-dataset-sample",
            ) as dataset_span:
               
                df = pd.read_csv(DATASET_PATH)
                sample_df = df.head(N_EXAMPLES).copy()

                dataset_span.update(
                        output={
                            "num_examples": len(sample_df),
                        },
                        metadata={
                            "dataset_path": DATASET_PATH,
                        },
                    )
           
            with root_span.start_as_current_observation(
                as_type="event",
                name="dataset-sample-loaded"
            ):
                pass
            

            with root_span.start_as_current_observation(
                as_type="span",
                name="load-prompts",
            ) as prompt_span:

                answer_prompt_client = langfuse.get_prompt(
                    name=ANSWER_PROMPT_NAME,
                    label=ANSWER_PROMPT_LABEL,
                )
                answer_system_prompt = answer_prompt_client.prompt

                judge_prompt = load_prompt(JUDGE_PROMPT_PATH)

                prompt_span.update(
                    output={
                        "answer_prompt_name": ANSWER_PROMPT_NAME,
                        "answer_prompt_label": ANSWER_PROMPT_LABEL,
                        "judge_prompt_path": JUDGE_PROMPT_PATH,
                    }
                )

            results = []

            for idx, row in sample_df.iterrows():
                with root_span.start_as_current_observation(
                    as_type="span",
                    name=f"evaluate-{idx}"
                ) as evaluate_span:

                    question = row["instruction"]
                    expected_answer = row["response"]
                    category = row["category"]

                    completion = openai.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": answer_system_prompt},
                            {"role": "user", "content": question},
                        ],
                        name=f"generate-answer-{idx}",
                    )
            
                    model_answer = completion.choices[0].message.content

        
                    judge_input = (
                        f"Pergunta: {question}\n\n"
                        f"Resposta esperada: {expected_answer}\n\n"
                        f"Resposta do modelo: {model_answer}"
                    )

                    judge_completion = openai.chat.completions.create(
                        model=JUDGE_MODEL,
                        messages=[
                            {"role": "system", "content": judge_prompt},
                            {"role": "user", "content": judge_input},
                        ],
                        name=f"llm-answer-{idx}",
                    )

                    judge_score = judge_completion.choices[0].message.content

                    evaluation_result = {
                        "question": question,
                        "expected_answer": expected_answer,
                        "model_answer": model_answer,
                        "judge_score": judge_score,
                        "category": category,
                    }

                    evaluate_span.update(
                        input={ "question": question,
                                "expected_answer": expected_answer,},
                        output={"judge_score": judge_score,
                                "model_answer": model_answer}
                    )

                    results.append(evaluation_result)
                
            with root_span.start_as_current_observation(
                as_type="span",
                name="aggregate-results",
            ) as aggregate_span:
                
                summary = {
                    "num_examples": len(results),
                    "num_judge_answers": len(results),
                    "judge_answers_preview": [r["judge_score"] for r in results[:2]],
                }

                aggregate_span.update(
                    output=summary
                )
                
            root_span.update(
                    output={
                        "summary": summary,
                        "results": results,
                    }
                )

   
    langfuse.flush()


    return {
        "summary": summary,
        "results": results,
    }




if __name__ == "__main__":
    results = run_batch_llm_judge_evaluation()

    print("\nResumo da avaliação:\n")
    print(results["summary"])

    print("Resultados individuais: ")
    for item in results["results"]:
        print("Pergunta:", item["question"])
        print("Avaliação do juiz:", item["judge_score"])
        print("-" * 50)

