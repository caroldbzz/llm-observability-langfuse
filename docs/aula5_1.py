import pandas as pd
from dotenv import load_dotenv
from langfuse import get_client


from langfuse import propagate_attributes
from langfuse.openai import openai


load_dotenv()


langfuse = get_client()


DATASET_PATH = "data/bitext_customer_support.csv"
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_MODEL_JUDGE = "gpt-4.1-mini"
JUDGE_PROMPT_PATH = "prompts/customer_support_answer_judge.md"
ANSWER_PROMPT_NAME = "customer_support_assistant"
ANSWER_PROMPT_LABEL ="production"


N_EXAMPLES = 5




def load_prompt(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def run_batch_llm_judge_pipeline():
    with langfuse.start_as_current_observation(
        as_type="span",
        name="batch-llm-judge-pipeline"
    ) as root_span:
       
        with propagate_attributes(
            session_id="llm-judge-session-123",
            user_id="id-123",
            tags=["customer-support", "llm-as-judge-batch"],
            metadata={
                "environment": "develop",
                "version": "V1",
                "dataset_path": DATASET_PATH,
                "judge_prompt_path": JUDGE_PROMPT_PATH,
                "answer_prompt_path": ANSWER_PROMPT_NAME,
                "answer_prompt_label":ANSWER_PROMPT_LABEL,
                "num_examples": N_EXAMPLES
            }
        ):
       
            with root_span.start_as_current_observation(
                as_type="span",
                name="load-dataset"
            ) as dataset_span:
               
                df = pd.read_csv(DATASET_PATH)
                sample_df = df.head(N_EXAMPLES).copy()


                dataset_span.update(input={"dataset_path": DATASET_PATH},
                                    output={"num_samples": N_EXAMPLES,})
           
            with root_span.start_as_current_observation(
                as_type="event",
                name="dataset-loaded"
            ):
                pass
           
            with root_span.start_as_current_observation(
                as_type="span",
                name="load-prompts"
            ) as prompt_load_span:
               
                judge_prompt = load_prompt(JUDGE_PROMPT_PATH)


                prompt_client = langfuse.get_prompt(
                    name=ANSWER_PROMPT_NAME,
                    label=ANSWER_PROMPT_LABEL
                )


                answer_system_prompt = prompt_client.prompt


                prompt_load_span.update(output={"JUDGE_PROMPT_PATH": JUDGE_PROMPT_PATH,
                                               "ANSWER_PROMPT_NAME": ANSWER_PROMPT_NAME})
            results = []
            for idx, row in sample_df.iterrows():
                with root_span.start_as_current_observation(
                    as_type="span",
                    name=f"evaluation-{idx}"
                ) as evaluation_span:
                   
                    question = row["instruction"]
                    expected_answer = row["response"]
                    category = row["category"]


                    completion = openai.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[{"role": "system", "content": answer_system_prompt},
                                {"role": "user", "content": question}],
                        name=f"generate-answer-{idx}"
                    )
                   
                    model_answer = completion.choices[0].message.content
                   
                    judge_input = (
                        f"Pergunta: {question}/n/n"
                        f"Resposta esperada: {expected_answer} /n/n"
                        f"Resposta gerada: {model_answer}"
                    )


                    completion_judge = openai.chat.completions.create(
                        model=OPENAI_MODEL_JUDGE,
                        messages=[{"role": "system", "content": judge_prompt},
                                {"role": "user", "content": judge_input}],
                        name=f"judge-answer-{idx}"
                    )


                    judge_score = completion_judge.choices[0].message.content


                    evaluation_result = {
                        "question": question,
                        "expected_answer": expected_answer,
                        "model_answer": model_answer,
                        "judge_score": judge_score,
                        "category": category


                    }


                    evaluation_span.update(input={"question": question,
                                            "expected_answer": expected_answer},
                                    output={"model_answer": model_answer,
                                            "judge_score": judge_score})


                    results.append(evaluation_result)

            with root_span.start_as_current_observation(
                as_type="span",
                name="aggregate-results",
            ) as aggregate_span:
                
                summary = {
                    "num_examples": len(results),
                    "judge_answers_preview": [r["judge_score"] for r in results[:2]],
                }

                aggregate_span.update(
                    output=summary
                )

            with root_span.start_as_current_observation(
                as_type="span",
                name="analyze-priority-case"
            ) as review_span:
                review_cases = []
                for item in results:
                    score_str = item.get("judge_score", "")
                    score_val = float(score_str.split("SCORE:", 1)[1].split()[0])

                    if score_val <= 3:
                        review_cases.append(item)

                review_span.update(
                    input={
                        "total_cases": len(results),
                        "threshold": 3
                    },
                    output={
                        "review_cases_count": len(review_cases),
                        "review_cases": review_cases
                    }
                )
                
        root_span.update(
                    input={"dataset_path": DATASET_PATH, "num_evaluated_cases": len(results)},
                    output={
                        "summary": summary,
                        "results": results,
                        "review_cases_count": len(review_cases),
                        "review_cases": review_cases
                    }
                )
   
    langfuse.flush()


    return {
        "summary": summary,
        "results": results,
        "review_cases_count": len(review_cases),
        "review_cases": review_cases,
    }



if __name__ == "__main__":
    batch_result = run_batch_llm_judge_pipeline()
    review_cases = batch_result["review_cases"]

    print(f"Casos totais: {len(batch_result['results'])}")
    print(f"Casos para revisão manual (score <= 3): {len(review_cases)}")
    print("Casos prioritários para revisão manual:")

    for item in review_cases:
        print("Pergunta:", item.get("question"))
        print("Avaliação do juiz:", item.get("judge_score"))
