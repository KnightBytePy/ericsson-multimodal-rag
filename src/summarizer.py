import os
import json
import time
import google.generativeai as genai
import PIL.Image

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("GOOGLE_API_KEY not found in environment.")
    GOOGLE_API_KEY = input("Please paste your API Key here: ").strip()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

MODEL_NAME = "gemini-2.5-flash-lite"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "pages")
SAVE_FILE = os.path.join(SCRIPT_DIR, "image_summaries.json")

genai.configure(api_key=GOOGLE_API_KEY)

print(f"Connecting to {MODEL_NAME}...")
try:
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    print(f"Error setting up model: {e}")
    exit(1)


def summarize_image(image_path):
    filename = os.path.basename(image_path)
    print(f"Analyzing: {filename}...", end=" ", flush=True)

    try:
        img = PIL.Image.open(image_path)

        prompt = """
        Analyze this chart/figure from the Ericsson Mobility Report.
        1. Identify the key metric.
        2. Read specific numbers/forecasts for 2024-2030.
        3. Note regional trends.
        Output a concise summary.
        """

        response = model.generate_content([prompt, img])

        if not response.text:
            return "Error: Empty response."

        # Sleep to handle Rate Limits (4s is usually enough for 15 RPM, keeping 5s for safety)
        time.sleep(5)
        print("Done")
        return response.text.strip()

    except Exception as e:
        error_str = str(e)
        # Handle Quota limits
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            print("\n⚠️ QUOTA HIT. Cooling down for 60 seconds...")
            time.sleep(60)
            print("   Retrying...")
            return summarize_image(image_path)

        print(f"\nError: {e}")
        return f"Error analyzing image: {e}"


def get_image_summaries():
    # Verify Image Directory
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Image directory not found at {IMAGE_DIR}")
        return []

    existing_data = []
    processed_map = {}

    # 1. Load existing progress
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r") as f:
                existing_data = json.load(f)
                for item in existing_data:
                    processed_map[item['image_path']] = item['description']
            print(f"Resuming... Found {len(existing_data)} entries.")
        except:
            print("Starting fresh.")

    # 2. Find files to process
    all_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if
                 f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files_to_process = []

    for full_path in all_files:
        if full_path not in processed_map:
            files_to_process.append(full_path)
        elif "error analyzing image" in processed_map[full_path].lower() or "error:" in processed_map[
            full_path].lower():
            print(f"⚠️ Retrying previously failed image: {os.path.basename(full_path)}")
            files_to_process.append(full_path)
            # Remove bad entry from data
            existing_data = [x for x in existing_data if x['image_path'] != full_path]

    print(f"🚀 Found {len(files_to_process)} images to analyze using {MODEL_NAME}.")

    # 3. Process loop
    for i, full_path in enumerate(files_to_process):
        desc = summarize_image(full_path)

        record = {"image_path": full_path, "description": desc}
        existing_data.append(record)

        # Save frequently
        if (i + 1) % 5 == 0:
            with open(SAVE_FILE, "w") as f:
                json.dump(existing_data, f, indent=4)
            print("Saved progress.")

    # Final Save
    with open(SAVE_FILE, "w") as f:
        json.dump(existing_data, f, indent=4)
    print("Process finished.")


if __name__ == "__main__":
    get_image_summaries()