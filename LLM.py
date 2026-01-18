import json
import time
import requests
from tqdm import tqdm
import os
import concurrent.futures
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
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_JSONL = f"{OUTPUT_DIR}/{DATASET}_max{MAX_HIS_NUM}_output.jsonl"

MAX_WORKERS = 30
REQUEST_TIMEOUT = (100, 200)
MAX_RETRIES = 3


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
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        print("Connection error, retrying...")
        time.sleep(5)
        raise
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {str(e)}")
        raise


def process_line(line):
    try:
        data = json.loads(line.strip())
        payload = data["body"]
        api_response = call_deepseek_api(payload)

        return {
            "custom_id": int(data["custom_id"]),
            "data": {
                "custom_id": data["custom_id"],
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
        }
    except Exception as e:
        print(f"Error processing line: {str(e)}")
        return {
            "custom_id": int(data.get("custom_id", 0)),
            "data": {
                "custom_id": data.get("custom_id", "error"),
                "error": str(e),
                "original_line": line.strip()
            }
        }


def main():
    with open(INPUT_JSONL, "r", encoding="utf-8") as infile:
        lines = list(infile)

    results = []

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=MAX_WORKERS,
            thread_name_prefix="api_worker"
    ) as executor:
        futures = [executor.submit(process_line, line) for line in lines]

        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(lines),
                           desc="Processing"):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Unexpected error: {str(e)}")

    results.sort(key=lambda x: x["custom_id"])

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as outfile:
        for result in results:
            outfile.write(json.dumps(result["data"], ensure_ascii=False) + "\n")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nProcessing completed in {time.time() - start_time:.2f} seconds.")
