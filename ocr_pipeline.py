import time
import logging
from PIL import Image
import os
import cv2
from base64 import b64decode
import numpy as np
import base64
from ollama import ChatResponse, chat, Client
import csv

# ---------------- Logging Configuration ---------------- #
p = os.path.basename(__file__).replace('.py', '').replace(' ', '_')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"Logs/{p}.log"),
        logging.StreamHandler()
    ]
)

# ---------------- Environment Variables ---------------- #
try:
    logging.info('Loading environment variables...')
    client = Client(host=os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434'))
    source_folder = os.environ.get('SOURCE_FOLDER', "Source")
    target_folder = os.environ.get('TARGET_FOLDER', "Target")
    model = os.environ.get('MODEL', 'llama3.2-vision:11b')
except KeyError as e:
    logging.error(f'Missing environment variable: {e}')
    exit(1)

processed_folder = os.path.join(source_folder, "Processed")

# ---------------- Prompt ---------------- #
prompt = """ 
You are the world's best image-to-text converter. Your task is to extract **all** text content from an image **exactly as it appears**, without modification, summarization, or omission.

## Instructions:
1. **Extract text exactly as it is displayed in the image**—do not interpret, modify, or infer any content.
2. Ensure the following key fields are **captured accurately**:
   - **Programme:** (Degree type such as B.Tech, M.Tech, PhD, DIIT, etc.)
   - **Branch:** (Department name such as Mechanical, Civil, Aerospace, etc.)
   - **Date of Convocation:** (Exact date format as in the image)
   - **Sr. No.:** (Serial Number as listed)
   - **Roll No.:** (Exact roll number from the image)
   - **Name of the Student:** (Indian names should be extracted as they appear; do not modify them)
   - **Absent or Present:** (Clearly indicate if the student was absent or present)
3. **If any text is unclear**, extract as much as possible while maintaining accuracy.
4. Try to **classify** the text whether it's a degree, department, name of the student, date fields or roll number based on your understanding.
5. Ignore data that doesn't fall into any of the key fields.
6. Don't repeat values if not present in the image.

## Output Format:
Your response must be structured as a markdown table with the following format:

| **Sr. No.** | **Programme** | **Branch** | **Date of Convocation** | **Roll No.** | **Name of the student** | **Absent or Present** |
| ----------- | ------------- | ---------- | ----------------------- | ------------ | ----------------------- | --------------------- |
|             |               |            |                         |              |                         |                       |


## Additional Constraints:
- **Do not generate any additional text or explanations.** The output should only be the structured table.
- **Do not summarize or categorize the data beyond the fields given.** Maintain exact textual integrity.
- **Preserve formatting and spacing** to ensure readability in a structured table format.

If the extracted text does not fit within the given table format due to missing values, leave the cell empty instead of making assumptions.
"""

# ---------------- Core Functions ---------------- #
def create_processed_dir():
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
        logging.info(f"Created '{processed_folder}' directory.")

def get_files(directory, extension='.jpg'):
    file_names = [item for item in os.listdir(directory)
                  if os.path.isfile(os.path.join(directory, item)) and item.endswith(extension)]
    return file_names

def find_missing_files(source, target):
    target = [x.replace('.csv', '.jpg') for x in target]
    missing_files = list(set(source) - set(target))
    logging.info(f"{len(missing_files)} files missing in target directory.")
    return missing_files

def rotate_and_save_image(file_path):
    logging.info(f'Rotating and saving image: {file_path}')
    img = Image.open(os.path.join(source_folder, file_path))
    rotated_img = img.rotate(90, expand=True)
    dest_path = os.path.join(processed_folder, os.path.basename(file_path))
    rotated_img.save(dest_path)
    logging.info(f"Saved rotated image to: {dest_path}")

def image_to_csv(file):
    logging.info(f'Extracting data from image: {file}')
    base64_code = encode_image(os.path.join(processed_folder, file))
    base64_code = pre_process_image(base64_code)
    llm_response = ask_ollama(model, prompt, base64_code)
    return llm_response

def ask_ollama(model, prompt, image):
    logging.info('Sending image to Ollama model...')
    response: ChatResponse = client.chat(
        model=model,
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [image]
        }]
    )
    return response.message.content

def encode_image(image_path: str) -> str:
    logging.info(f'Encoding image to base64: {image_path}')
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def pre_process_image(image_base64: str) -> bytes:
    logging.info('Preprocessing image for OCR...')
    image_bytes = b64decode(image_base64)
    nparr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        logging.error("Could not decode base64 string into an image")
        raise ValueError("Image decode failed")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced)
    thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.bitwise_not(thresh)
    _, img_encoded = cv2.imencode('.jpg', thresh)
    return img_encoded.tobytes()

def parse_markdown_table(markdown_text):
    logging.info('Parsing markdown table...')
    lines = markdown_text.split('\n')
    headers = [h.strip() for h in lines[0].split('|')[1:-1]]
    rows = []
    for line in lines[2:]:
        if '|' in line:
            cells = [c.strip() for c in line.split('|')[1:-1]]
            rows.append(cells)
    return headers, rows

def write_to_csv(headers, rows, file):
    csv_path = os.path.join(target_folder, file.replace('.jpg', '') + '.csv')
    logging.info(f'Writing data to CSV: {csv_path}')
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)
    return csv_path

# ---------------- Main Script ---------------- #
def main():
    logging.info("Starting image-to-markdown pipeline...")
    create_processed_dir()
    source_files = get_files(source_folder)
    converted_files = get_files(target_folder, extension='csv')
    missing_files = find_missing_files(source_files, converted_files)

    for file in missing_files:
        try:
            rotate_and_save_image(file)
            response = image_to_csv(file)
            headers, rows = parse_markdown_table(response)
            csv_file_path = write_to_csv(headers, rows, file)
            logging.info(f"Markdown table saved to: {csv_file_path}")
            logging.info("Cooling down GPU for 2 minutes...")
            time.sleep(120)
        except Exception as e:
            logging.error(f"Error processing file {file}: {e}")

if __name__ == "__main__":
    main()
