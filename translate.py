import pandas as pd
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import time

if os.getenv("OPENAI_API_KEY") is None:
    load_dotenv()

N = 5
MODEL_ID = "gpt-4.1"
TRANSLATE_INSTRUCTION = "請將以下的題目翻譯為繁體中文，並注意以下規則\n請注意，若為人名則不需翻譯"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

for subset in ["Language", "Gender", "Emotion", "Animal"]:
    df = pd.read_csv(f"data/{subset}/metadata.csv")
    
    if N == -1:
        N = len(df)

    batch_inputs = ""

    for i in range(N):
        question_1 = df.iloc[i]["single_instruction"]
        answer_1 = df.iloc[i]["single_answer"]
        question_2 = df.iloc[i]["multi_instruction"]
        answer_2 = df.iloc[i]["multi_answer"]
        file_name = df.iloc[i]["file"]

        body = {
        "model": MODEL_ID,
        "input": [
            {
            "role": "system",
            "content": [
                {
                "type": "input_text",
                "text": TRANSLATE_INSTRUCTION
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "input_text",
                "text": f"# question_1\n{question_1}\n\n# answer_1\n{answer_1}\n\n# question_2\n{question_2}\n\n# answer_2\n{answer_2}"
                }
            ]
            }
        ],
        "text": {
            "format": {
            "type": "json_schema",
            "name": "translation_results",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                "question_1": {
                    "type": "string"
                },
                "answer_1": {
                    "type": "string"
                },
                "question_2": {
                    "type": "string"
                },
                "answer_2": {
                    "type": "string"
                }
                },
                "required": [
                "question_1",
                "answer_1",
                "question_2",
                "answer_2"
                ],
                "additionalProperties": False
            }
            }
        },
        "reasoning": {},
        "tools": [],
        "temperature": 1,
        "max_output_tokens": 2048,
        "top_p": 1,
        "store": True
        }

        request_obj = {
            "custom_id": file_name,
            "method": "POST",
            "url": "/v1/responses",
            "body": body
        }

        batch_inputs += json.dumps(request_obj, ensure_ascii=False) + "\n"
    
    with open(f"batch_file_{subset}.jsonl", "w") as f:
        f.write(batch_inputs)
    
    # Upload batch file
    batch_input_file = client.files.create(
        file=open(f"batch_file_{subset}.jsonl", "rb"),
        purpose="batch"
    )

    # Create batch job
    batch_input_file_id = batch_input_file.id
    batch_job = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "description": f"{subset} subset, {N} samples"
        }
    )

    # Wait for batch job to complete
    batch = client.batches.retrieve(batch_job.id)
    while batch.status != "completed":
        print(f"{subset} batch status: ", batch.status)
        if batch.status == "failed":
            print(batch)
            raise Exception(f"{subset} batch failed")
        time.sleep(10)
        batch = client.batches.retrieve(batch_job.id)

    # Download batch output
    file_response = client.files.content(batch.output_file_id)
    with open(f"batch_output_{subset}.jsonl", "w") as f:
        f.write(file_response.text)
    
    # Create metadata_tw.csv with Traditional Chinese instructions
    tw_translate_output = []
    with open(f"batch_output_{subset}.jsonl", "r") as f:
        for line in f:
            json_obj = json.loads(line)
            trans_res = json_obj["response"]["body"]["output"][0]["content"][0]["text"]
            tw_translate_output.append(json.loads(trans_res))
    
    for i in range(N):
        df.loc[df.index[i], "single_instruction"] = tw_translate_output[i]["question_1"]
        df.loc[df.index[i], "multi_instruction"] = tw_translate_output[i]["question_2"]
        df.loc[df.index[i], "single_answer"] = tw_translate_output[i]["answer_1"]
        df.loc[df.index[i], "multi_answer"] = tw_translate_output[i]["answer_2"]

    df.to_csv(f"data/{subset}/metadata_tw.csv", index=False)

    print(f"{subset} batch completed")
