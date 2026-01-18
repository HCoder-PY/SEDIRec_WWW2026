import json
import os
import time
import requests
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# Please fill in your DeepSeek API.
API_KEY = "123456789"

# Choose the dataset to process.
DATASET = "book-crossing"
# DATASET = "dbbook2014"
# DATASET = "ml1m"

INPUT_JSONL = f"batch_input/{DATASET}_user_max_his30_deepseek-chat_input.jsonl"
MAX_HIS_NUM = 30
OUTPUT_DIR = "batch_output"
OUTPUT_JSONL = f"{OUTPUT_DIR}/{DATASET}_max{MAX_HIS_NUM}_output.jsonl"
TEMP_JSONL = f"{OUTPUT_JSONL}.temp"

MAX_RETRIES = 3
REQUEST_TIMEOUT = (100, 200)


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def call_deepseek_api(payload):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    return response.json()


def load_input_data(input_file):
    input_data = {}
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                custom_id = data["custom_id"]
                input_data[custom_id] = data
    return input_data


def check_and_fix_errors(output_file, input_data):
    with open(output_file, "r", encoding="utf-8") as f:
        output_lines = [line.strip() for line in f if line.strip()]

    error_custom_ids = set()
    for line in output_lines:
        data = json.loads(line)
        if "error" in data:
            error_custom_ids.add(data["custom_id"])

    if not error_custom_ids:
        print("No errors found in the output file. Nothing to fix.")
        return

    print(f"Found {len(error_custom_ids)} errors. Reprocessing...")

    fixed_results = {}
    for custom_id in tqdm(error_custom_ids, desc="Reprocessing errors"):
        if custom_id in input_data:
            payload = input_data[custom_id]["body"]
            try:
                api_response = call_deepseek_api(payload)
                fixed_results[custom_id] = {
                    "custom_id": custom_id,
                    "response": {
                        "body": {
                            "choices": [{
                                "message": {
                                    "content": api_response["choices"][0]["message"]["content"]
                                }
                            }]
                        }
                    }
                }
            except Exception as e:
                print(f"Failed to reprocess custom_id {custom_id}: {str(e)}")
                fixed_results[custom_id] = None

    with open(TEMP_JSONL, "w", encoding="utf-8") as temp_file:
        for line in tqdm(output_lines, desc="Updating output file"):
            data = json.loads(line)
            custom_id = data["custom_id"]
            if custom_id in fixed_results and fixed_results[custom_id]:
                temp_file.write(json.dumps(fixed_results[custom_id], ensure_ascii=False) + "\n")
            else:
                temp_file.write(line + "\n")

    os.replace(TEMP_JSONL, output_file)
    print("\nFile fixed successfully!")


if __name__ == "__main__":
    start_time = time.time()
    input_data = load_input_data(INPUT_JSONL)
    check_and_fix_errors(OUTPUT_JSONL, input_data)
    print(f"\nTotal time: {time.time() - start_time:.2f} seconds.")
