# Document to Markdown Converter

## Overview

The **Document to Markdown Converter** is a Streamlit-based web application that allows users to convert PDFs, images, and other document formats into Markdown. The application leverages **Docling** as the backend for document processing and **Ollama** to utilize vision models for tasks such as text extraction, image description, and document conversion. This tool is particularly useful for generating structured Markdown content from documents, which can be used for various purposes, including RAG (Retrieval-Augmented Generation) document preparation.

## Features

- **Document Conversion**: Convert PDFs, DOCX, PPTX, and images (JPEG, JPG, PNG) into Markdown format.
- **OCR Support**: Extract text from scanned documents or images using Optical Character Recognition (OCR).
- **Image Description**: Generate detailed descriptions of images within documents using AI models.
- **LLM Integration**: Use Large Language Models (LLMs) via Ollama for advanced text extraction and image analysis.
- **Image Enhancement**: Preprocess images to improve OCR accuracy and image description quality.
- **Custom Prompts**: Provide custom prompts to guide the AI in generating more accurate and relevant outputs.
- **Markdown Export**: Download the converted document as a Markdown file.

## How It Works

1. **Upload Document**: Users upload a document (PDF, DOCX, PPTX) or an image (JPEG, JPG, PNG) through the Streamlit interface.
2. **Select Vision Model**: Choose a vision model from the available options (e.g., LLaVA, Granite, Gemma) to process the document.
3. **Configure Settings**: Enable or disable OCR, image enhancement, and image description extraction based on your needs.
4. **Provide Custom Prompt**: Optionally, provide a custom prompt to guide the AI in generating the output.
5. **Convert to Markdown**: The application processes the document using the selected model and settings, converting it into Markdown format.
6. **Download Result**: The converted Markdown content is displayed and can be downloaded as a `.md` file.

## Installation

### To run the application locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/khamsakamal48/Document-to-Markdown.git
   cd Document-to-Markdown
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.9 or higher installed. Then, install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   Start the Streamlit app by running:
   ```bash
   streamlit run app.py
   ```

4. **Access the Application**:
   Open your web browser and navigate to `http://localhost:8501` to access the application.

### To run the application inside a contained, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/khamsakamal48/Document-to-Markdown.git
   cd Document-to-Markdown

2. **Use Podman or Docker**
    ```bash
   # Using Podman
   mv compose.yml docker-compose.yml
   podman-compose up -d
   
    # Using Docker
   docker-compose up -d -f compose.yml
   

## Usage

1. **Upload a Document**: Use the file uploader to upload a document or image.
2. **Select a Vision Model**: Choose a vision model from the dropdown menu in the sidebar.
3. **Configure Settings**: Adjust the settings (OCR, image enhancement, etc.) as needed.
4. **Provide a Custom Prompt**: Optionally, enter a custom prompt in the text area.
5. **Convert**: Click the "Convert to Markdown Format" button to start the conversion process.
6. **Download**: Once the conversion is complete, download the Markdown file using the provided button.

## Backend Technologies

- **Docling**: Used for document processing, OCR, and image description extraction.
- **Ollama**: Provides access to vision models and LLMs for advanced text and image analysis.
- **Streamlit**: Powers the web interface, making it easy to interact with the application.

## Supported Formats

- **Documents**: PDF, DOCX, PPTX
- **Images**: JPEG, JPG, PNG

## Custom Prompts

You can provide custom prompts to guide the AI in generating more accurate and relevant outputs. For example:

- **Text Extraction Prompt**: 
  ```plaintext
  Extract all text from the image while preserving the original formatting and structure.
  ```
- **Image Description Prompt**: 
  ```plaintext
  Describe the image in detail, including any text, objects, and their spatial relationships.
  ```

## Examples

### Example 1: Converting a PDF to Markdown

1. Upload a PDF document.
2. Select a vision model (e.g., LLaVA).
3. Enable OCR and image enhancement.
4. Click "Convert to Markdown Format".
5. Download the resulting Markdown file.

### Example 2: Extracting Text from an Image

1. Upload an image (JPEG, JPG, PNG).
2. Select a vision model (e.g., Granite).
3. Enable OCR and provide a custom prompt for text extraction.
4. Click "Convert to Markdown Format".
5. Download the extracted text in Markdown format.

## Troubleshooting

- **Ollama Not Installed**: If you encounter an error indicating that Ollama is not installed, ensure that Ollama is properly installed and running on your system.
- **Image Processing Issues**: If the image processing fails, try enabling the "Enhance Image" option to improve the quality of the image before processing.

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and create a pull request with your changes. Ensure that your code follows the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- **Streamlit** for the web framework.
- **Docling** for document processing capabilities.
- **Ollama** for providing access to powerful vision models and LLMs.

## Contact

For any questions or issues, please open an issue on the [GitHub repository](https://github.com/khamsakamal48/Document-to-Markdown/issues).

---

**Note**: This application is designed for educational and experimental purposes. Ensure that you have the necessary rights and permissions to process and convert the documents and images you upload.