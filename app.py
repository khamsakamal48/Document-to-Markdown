import streamlit as st  # Import Streamlit for creating web applications.

# Import Path from pathlib for filesystem path manipulations.
from pathlib import Path

import base64  # Module to encode and decode data in base64 format.

import ollama  # Library to interact with the Ollama AI model.

import tempfile  # Provides utilities to create temporary files/directories.

from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
# Import specific classes/functions from 'ollama' package for chat interactions.
from ollama import ChatResponse, chat, Client

from httpx import ConnectError  # Exception class for handling HTTP connection errors in 'httpx'.

# Import ImageRefMode enum from the docling_core library to specify how images should be referenced.
from docling_core.types.doc import ImageRefMode

# Import InputFormat enumeration from docling for defining input formats.
from docling.datamodel.base_models import InputFormat

# Import SimplePipeline class from docling.pipeline for processing documents in a simple manner.
from docling.pipeline.simple_pipeline import SimplePipeline

# Import granite_picture_description options for configuring picture description tasks.
from docling.datamodel.pipeline_options import granite_picture_description, TableFormerMode
from docling_core.types.doc.document import PictureDescriptionData  # Data structure for picture descriptions.

# Import various classes and enumerations related to document conversion from the docling library.
from docling.document_converter import (
    DocumentConverter,  # Class to convert documents between different formats.
    PdfFormatOption,  # Enum for specifying PDF-related format options.
    WordFormatOption,  # Enum for specifying Word-related format options.
    ImageFormatOption
)

# Import additional pipeline options related to PDF processing from the docling library.
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,  # Options specifically for configuring PDF pipelines.
    AcceleratorDevice,
    AcceleratorOptions,
)

from docling.models.tesseract_ocr_cli_model import TesseractCliOcrOptions

import cv2  # OpenCV library for image and video processing.

import numpy as np  # NumPy library for numerical operations on arrays.

from base64 import b64decode  # Function to decode base64 encoded data.

import os  # Module to interact with the operating system.

# Initializes a `Client` instance with the host set to the value of the environment variable 'OLLAMA_BASE_URL'.
try:
    client = Client(
        host=os.environ['OLLAMA_BASE_URL']  # Retrieves the base URL from an environment variable.
    )
except KeyError:
    client = Client(
        host=''
    )

# Remove all classes from 'torch.classes' to prevent class conflicts or errors.
import torch
torch.classes.__path__ = []

# Set up the configuration for the Streamlit page with specific settings:
# - `page_title` sets the title of the webpage.
# - `page_icon` specifies an emoji to be used as the icon in the browser tab.
# - `layout='wide'` configures the layout to display content in a wider format, suitable for desktop views.

st.set_page_config(
    page_title='Document to Markdown converter',  # Title displayed on the browser tab.
    page_icon=':document:',  # Use the document emoji as an icon in the browser tab.
    layout='wide'  # Set layout option to wide for a broader display area.
)

# Display the main title of the web application within Streamlit's UI:
st.title('Document to Markdown converter')
# Add a horizontal line (divider) to separate the title from subsequent content.
st.divider()

########################################################################################################################
#                                                   FUNCTIONS                                                          #
########################################################################################################################
def convert_to_markdown(file, ocr, extract_image_desc, extract_via_llm, ollama_model, ollama_llm, llm_prompt,
                        enhance_image):
    """
    Converts a document file to Markdown format with optional OCR and image description extraction.

    :param file: Path or URI of the input file (PDF, DOCX, etc.).
    :type file: str
    :param ocr: Boolean indicating whether to perform OCR on PDFs.
    :type ocr: bool
    :param extract_image_desc: Boolean indicating if image descriptions should be extracted and included.
    :type extract_image_desc: bool
    :param extract_via_llm: Boolean indicating if extraction should be done via LLM (Large Language Model).
    :type extract_via_llm: bool
    :param ollama_model: The model identifier for Ollama to use in processing.
    :type ollama_model: str
    :param ollama_llm: Boolean indicating whether to use Ollama's LLM for image descriptions.
    :type ollama_llm: bool
    :param llm_prompt: The prompt used for querying the LLM if extract_via_llm is True.
    :type llm_prompt: str
    :param enhance_image: Boolean flag indicating whether to preprocess images before extraction.
    :type enhance_image: bool
    :return: A string containing the Markdown version of the document with optional image descriptions.
    :rtype: str
    """

    # Set up Vision Model through Ollama
    model = ollama_model

    # Check if ollama is available
    if ollama_llm:
        try:
            ollama.ps()

        except ConnectionError:
            ollama_llm  = False


    if extract_via_llm:
        # Check if the file is an image type suitable for LLM processing
        if file.lower().endswith('.jpeg') or file.lower().endswith('.jpg') or file.lower().endswith('.png'):

            # Convert images to base64 string format
            base64_code = encode_image(file)

            # Optionally enhance the image before further processing
            if enhance_image:
                base64_code = pre_process_image(base64_code)

            prompt = llm_prompt

            # Ask Ollama to process the document using LLM
            conv_doc = ask_ollama(model, prompt, base64_code)

        else:
            st.error('Upload an Image only (.jpeg, .jpg, .png) for performing complete text extraction through an LLM.')
            conv_doc = ''  # Initialize as empty if invalid file type

    else:
        input_doc_path = Path(file)

        # Set up options for document conversion
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
            accelerator_options=AcceleratorOptions(
                num_threads=8,
                device=AcceleratorDevice.AUTO
            )
        )
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        # # Configure OCR options, though other features like table structure detection are commented out here
        ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True, lang=['auto'])
        pipeline_options.ocr_options = ocr_options

        if extract_image_desc:
            pipeline_options.images_scale = 2
            pipeline_options.generate_picture_images = True
            pipeline_options.do_picture_classification = True
            pipeline_options.do_picture_description = True
            pipeline_options.do_picture_classification = True
            pipeline_options.generate_page_images = True
            pipeline_options.generate_table_images = True

            # Allow connection to remote services for additional processing
            pipeline_options.enable_remote_services = True

            if not ollama_llm:
                pipeline_options.picture_description_options = granite_picture_description
                pipeline_options.picture_description_options.prompt = (
                    "Describe the image in less than three sentences. Be concise and accurate."
                )

        # Initialize a document converter with specified formats and options
        doc_converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.IMAGE,
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.PPTX,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options, backend=DoclingParseV2DocumentBackend),
                InputFormat.DOCX: WordFormatOption(pipeline_cls=SimplePipeline),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options)
            }
        )

        # Convert the input document using the specified options
        conv_res = doc_converter.convert(input_doc_path)

        if extract_image_desc:
            image_extracted = True
            annotations = []

            if not ollama_llm:
                annotations = []
                for pic in conv_res.document.pictures:
                    for annotation in pic.annotations:
                        if not isinstance(annotation, PictureDescriptionData):
                            continue
                        annotations.append(annotation.text)
            else:
                base64_code = []
                try:
                    # Extract base64-encoded image data from document pictures
                    for pic in conv_res.document.pictures:
                        base64_code.append(
                            str(pic.image.uri).replace('AnyUrl(\'data:image/png;base64,', '').replace(
                                'data:image/png;base64,',
                                ''))
                except AttributeError:
                    image_extracted = False  # Fallback if images can't be extracted
                    # Convert main file to base64 as a fallback approach
                    base64_code = [encode_image(file)]

                for code in base64_code:
                    # Enhance the image before sending it to LLM, if required
                    if enhance_image:
                        code = pre_process_image(code)

                    content = llm_prompt

                    annotations.append(ask_ollama(model, content, code))

            if image_extracted:
                # Format extracted annotations for Markdown inclusion as images
                annotations = ['<picture>\n\n' + a + '\n\n</picture>\n\n' for a in annotations]

                conv_doc = replace_with_img_desc(
                    conv_res.document.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER), annotations)
            else:
                # If only one image, use its description directly
                conv_doc = annotations[0]

        else:
            # Export document to Markdown without image descriptions
            conv_doc = conv_res.document.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)

    return conv_doc


def pre_process_image(image_base64: str) -> bytes:
    """
    Preprocesses an image for better OCR results by enhancing contrast and reducing noise.

    Decodes a base64-encoded string into an image, applies preprocessing techniques such as converting to grayscale,
    contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization), and denoising. The processed
    image is then re-encoded back to bytes in JPEG format.

    :param image_base64: A base64 encoded string representation of the image.
    :type image_base64: str
    :return: Byte stream of the preprocessed image in JPEG format.
    :rtype: bytes

    :raises ValueError: If the input base64 string cannot be decoded into an image.

    Note:
        This function assumes that the necessary libraries such as OpenCV, NumPy, and a suitable base64 decoder
        are already imported.
    """

    # Decode the base64-encoded image string into bytes
    image_bytes = b64decode(image_base64)

    # Convert byte data to numpy array and decode it into an OpenCV image
    nparr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode base64 string into an image")

    # Convert the color image to grayscale for further processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE to enhance contrast in the grayscale image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Reduce noise using Non-Local Means Denoising algorithm
    denoised = cv2.fastNlMeansDenoising(enhanced)

    # Otsu thresholding
    thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.bitwise_not(thresh)

    # Re-encode the processed image into JPEG format and convert to bytes
    _, img_encoded = cv2.imencode('.jpg', thresh)
    processed_image_bytes = img_encoded.tobytes()

    return processed_image_bytes


def ask_ollama(model, prompt, image):
    """
    Generates a response from the Ollama model using a provided text prompt and an optional image.

    This function interacts with the Ollama API to process the input prompt and image, returning the content of
    the generated message. It is used for tasks that may involve analyzing both textual data and images within
    a single query context.

    :param model: The identifier of the Ollama model to use.
    :type model: str
    :param prompt: Text input provided as a prompt to generate responses.
    :type prompt: str
    :param image: Base64-encoded string of an image, if applicable.
    :type image: str

    :return: The content of the response generated by the Ollama model.
    :rtype: str

    Note:
        This function assumes that a `chat` method is available from an imported module for interacting with
        the Ollama API and requires its result to be cast as `ChatResponse`.
    """

    # Send request to the Ollama chat interface with the model, prompt, and image
    response: ChatResponse = client.chat(
        model=model,
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [image]  # Include the image as part of the message context if provided
        }]
    )

    # Return the content of the generated message from the Ollama response
    return response.message.content


def encode_image(image_path: str) -> str:
    """
    Converts an image file into a base64-encoded string.

    This function reads the binary content of an image from the specified file path and encodes it as a
    base64 string. The resulting string can be used for data transfer or embedding in various formats that
    support base64 encoding (e.g., HTML, JSON).

    :param image_path: Path to the image file to be encoded.
    :type image_path: str

    :return: A base64-encoded string representing the contents of the image file.
    :rtype: str

    Note:
        This function assumes that the necessary `base64` module is imported for encoding purposes.
    """

    # Open the image file in binary read mode
    with open(image_path, "rb") as image_file:
        # Read the entire content of the file and encode it to base64
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    return encoded_image


def replace_with_img_desc(doc, annotations):
    """
    Replaces placeholder comments in a document with corresponding image descriptions.

    This function processes a Markdown or text-based document by identifying specific placeholders marked as
    `<!-- image -->`. It replaces these placeholders with provided image descriptions. The function ensures that
    each placeholder is replaced sequentially using the annotations list.

    :param doc: A string representing the original document content.
    :type doc: str
    :param annotations: A list of strings containing image descriptions to replace the placeholders.
    :type annotations: list[str]

    :return: The modified document with image placeholders replaced by their respective descriptions.
    :rtype: str

    Note:
        If there are more placeholders than available annotations, excess placeholders will remain unchanged.
        Similarly, extra annotations that do not correspond to a placeholder will be ignored.
    """

    # Split the document into lines for processing
    lines = doc.split('\n')
    current_index = 0
    new_lines = []

    for line in lines:
        parts = line.split('<!-- image -->')

        if len(parts) > 1 and current_index < len(annotations):
            # If there is a placeholder and an annotation available, replace it
            replacement_str = annotations[current_index]
            current_index += 1
            modified_line = replacement_str.join(parts)
        else:
            # Otherwise, keep the line unchanged
            modified_line = line

        new_lines.append(modified_line)

    # Join all lines back into a single document string
    return '\n'.join(new_lines)


# Function to extract and cache vision-related model names from the Ollama list of models
@st.cache_resource
def get_vision_models():
    """
    Retrieves a tuple containing names of vision-related models available in the Ollama installation.

    This function attempts to fetch a list of all models registered with Ollama, filtering for those whose
    names contain specific keywords indicative of their relevance to vision tasks (e.g., 'vision', 'llava',
    'gemma3', 'moondream'). The results are cached using Streamlit's caching mechanism to avoid redundant
    fetching on subsequent calls.

    :return: A tuple containing two elements:
             - model_names: A tuple of strings, each representing a vision-related model name.
             - ollama_avbl: A boolean indicating whether Ollama is installed and accessible (True) or not (False).
    :rtype: tuple[tuple[str], bool]

    Note:
        This function uses Streamlit's caching decorator to store the results. It handles exceptions that may
        occur if Ollama is not properly installed or accessible, in which case it warns the user via the
        Streamlit interface and returns an empty model name tuple along with False for availability status.
    """

    try:
        # Attempt to list all models from Ollama
        models_info = client.list()

        # Extract model names containing specific vision-related keywords
        model_names = tuple(
            model.model for model in models_info['models']
            if 'vision' in model.model or
            'llava' in model.model or
            'gemma3' in model.model or
            'moondream' in model.model
        )

        ollama_avbl = True  # Set availability to True if the above operation succeeds

    except (ConnectError, ConnectionError):
        # Catch connection errors indicating Ollama is not installed or accessible
        st.warning('Ollama not installed or not accessible.')
        model_names = ()  # Return an empty tuple for unavailable models
        ollama_avbl = False  # Set availability to False due to the error

    return model_names, ollama_avbl

########################################################################################################################
#                                               STREAMLIT WEB APP                                                      #
########################################################################################################################

with st.sidebar:
    # Extract model names for selection using cached function
    available_models, ollama_avb = get_vision_models()

    # Dictionary mapping model identifiers to their respective documentation URLs
    model_desc = {
        'llava:latest': 'https://llava-vl.github.io',
        'granite3.2-vision:latest': 'https://www.ibm.com/new/announcements/ibm-granite-3-2-open-source-reasoning-and-vision',
        'llama3.2-vision:latest': 'https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/',
        'llama3.2-vision:11b': 'https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/',
        'gemma3:12b': 'https://blog.google/technology/developers/gemma-3/'
    }

    # Sidebar header for selecting a vision model
    st.header('👁️ Select Vision model:')

    # Dropdown menu to select from available vision models
    vision_model = st.sidebar.selectbox(
        '**👁️ Select Vision model**',
        available_models,
        label_visibility='collapsed'  # Hide the label for a cleaner UI
    )

    # Provide info for selected model
    if vision_model:
        try:
            # Link button with documentation URL based on selected model
            st.link_button(
                f'Read More about `{vision_model}`',
                model_desc[vision_model],
                type='tertiary'  # Set link button style to tertiary for consistency
            )
        except KeyError:
            # Fallback link if the selected model is not in the predefined dictionary
            st.link_button(
                f'Read More about `{vision_model}`',
                'https://ollama.com/search',
                type='tertiary'
            )

    # Empty header for spacing
    st.header('')

    # Sidebar header for manual prompt entry
    st.header('💬 Provide a prompt for better results (optional)')

    # Text area for users to input a custom prompt
    manual_prompt = st.text_area(
        '💬 **Provide a prompt for better results (optional)**',
        value='',  # Default empty string
        help='Enter a prompt manually for better results. Leave empty to use the default prompt.',
        label_visibility='collapsed'  # Hide the label for cleaner UI
    )

# Streamlit file uploader for document conversion with specific supported formats
uploaded_file = st.file_uploader(
    'Upload Document to Convert',
    # Specify accepted file types (formats)
    type=['pdf', 'jpeg', 'jpg', 'png', 'ppt', 'pptx', 'doc', 'docx'],
    label_visibility='collapsed'  # Hide the label for a cleaner UI
)

# Begin conversion process if a file is uploaded
if uploaded_file:

    # Display subheader for additional configuration parameters
    st.subheader('Additional Parameters:')

    # Create four columns to organize the settings options in the UI
    col1, col2, col3, col4 = st.columns(4)

    user_prompt = ''  # Initialize variable for storing potential user-provided prompt

    with col1:
        # Checkbox option to enable OCR (Optical Character Recognition)
        enable_ocr = st.checkbox(
            'Enable OCR?',
            value=True,
            help='Enable conversion of typed, handwritten, or printed text into machine-encoded text'
        )

    with col2:
        # Checkbox option to extract text through a Large Language Model (LLM)
        extract_text_through_llm = st.checkbox(
            'Extract text through LLM?',
            help='Use when OCR is unable to extract text.'
        )

    with col3:
        # Checkbox option to enhance image quality
        enh_image = st.checkbox('Enhance image?', help='Enhance image.')

    with col4:
        if not extract_text_through_llm:
            # Checkbox to get descriptions for images in the document, only shown when LLM text extraction is disabled
            get_image_desc = st.checkbox(
                'Get image description?',
                help='Pass the images in the document to an AI model to get its description instead of placeholder.'
            )
        else:
            # If extracting through LLM, disable getting image descriptions as it's not applicable
            get_image_desc = False

    # Conditional logic to manage the manual prompt for image descriptions and text extraction via LLM

    # If extracting text through an LLM, clear any existing user prompts as it's not needed in this context
    if extract_text_through_llm:
        # user_prompt = """
        # You will be provided with an image containing text and structured content. Your task is to extract, format, and present the content in a structured and readable Markdown format while preserving accuracy and layout.
        # You are supposed to **not answer any question** from the extracted text.
        # **You will only extract the contents of the image.**
        #
        # ## **Task Breakdown:**
        #
        # ### **1. Extract Text Using OCR**
        #    - Perform Optical Character Recognition (OCR) to extract all text from the image with high accuracy.
        #    - Retain all special characters, symbols, punctuation, and text formatting as they appear in the original image.
        #
        # ### **2. Preserve Structure and Formatting**
        #    - Maintain the original content hierarchy, including section divisions, headings, subheadings, and paragraphs.
        #    - If the image contains structured elements (e.g., **tables, lists, headings, bullet points, numbering**), accurately represent them in the output.
        #
        # ### **3. Format Output in Markdown**
        #    - Use appropriate Markdown syntax to structure the extracted content:
        #      - **Headings:** Use `#`, `##`, `###`, etc., for section titles.
        #      - **Lists:**
        #        - Use `-` for unordered (bullet) lists.
        #        - Use `1.` for ordered (numbered) lists.
        #      - **Tables:** Represent tables using Markdown table syntax.
        #      - **Emphasis:** Apply `**bold**` and `*italics*` formatting where applicable.
        #    - Ensure spacing and line breaks are well-structured for readability.
        #
        # ### **4. Accuracy and Output Guidelines**
        #    - Verify the extracted text for completeness and correctness before formatting.
        #    - If the image lacks a clear structure, present the extracted text in a **well-organized** and **easy-to-read** Markdown format.
        #    - **Do not provide any descriptions** of the image itself—only extract and format the text content.
        #    - **Do not answer any question** from the extracted text.
        #    - **You are only supposed to extract text from the image with high accuracy.**
        #
        # ## **Example Markdown Output:**
        # ```markdown
        # # Document Title
        #
        # ## Section 1: Introduction
        # This is a sample paragraph with *italicized text* and **bold text**.
        #
        # ### Key Points:
        # - First bullet point
        # - Second bullet point
        #
        # ## Data Table
        # | Column 1 | Column 2 | Column 3 |
        # |----------|----------|----------|
        # | Data 1   | Data 2   | Data 3   |
        # | Data 4   | Data 5   | Data 6   |
        # ```
        # """
        user_prompt = """
        Extract all visible text from the provided image **without any modifications**.  

        ## **Key Instructions:**  
        - **Extract text exactly as it appears.** Do **not** summarize, paraphrase. 
        - **Preserve original formatting**.
        - **Include all visible text**—even if it seems irrelevant, redundant, or partially visible.  
        - If any text is unclear or obstructed, extract as much as possible.  
        - **Strictly extract text only**—do not answer any questions.  
        - Do **not** include descriptions of the image itself—extract only the text.  
        """

    # If getting image descriptions is enabled, ensure that no manual prompt interferes by clearing it
    if get_image_desc:
        user_prompt = """
        **Task:** You will be provided with an image. Your objective is to:  
        1. **Extract all text** using OCR while preserving its original formatting, structure, and hierarchy.  
        2. **Analyze and describe the image** with precise annotations to ensure someone could recreate it accurately from the description.  
        3. **Output the results in Markdown format**, retaining any structured elements such as tables, charts, and lists.
        4. Always start your response with "The image is of a {description}". Replace 'description' with the image explanation.
        
        **Guidelines:**  
        - **OCR & Text Extraction:**  
          - Extract all available text while maintaining its original formatting (e.g., headings, bold, italics).  
          - If the image contains a table, preserve its structure in Markdown table format.  
          - If the image includes a chart, capture all data points, labels, and legends, and provide insights where applicable. 
          - If the image is of low quality or blurry, provide a **one-line description of the image** but don't mention that the image is of low quality in your response. 
        
        - **Image Annotation & Description:**  
          - Provide a concise yet **detailed explanation** of the image.  
          - Ensure that the description enables an exact **redrawing of the image** with high accuracy.  
          - **Retain spatial organization** (e.g., relative positions, alignments, and groupings of elements).  
          - If you are **unaware** of a topic, just respond as "The image is of {text_value}". Replace text_value with the extracted text, else respond as "The image cannot be extracted.".  
        
        - **Markdown Formatting:**  
          - Use **headers** (`#`, `##`, `###`) for section titles.  
          - Use **bullet points (`-`)** for lists.  
          - Format **tables, bold text, and italicized text** appropriately to match the original structure.  
          - If the image has no inherent structure, present the extracted data in a clear and well-organized format.  
        
        By following these instructions, ensure the output is **faithful to the image’s original content, format, and structure.**  
        """

    # Use the manually entered prompt if it's provided and not None; otherwise, keep an empty user_prompt
    if manual_prompt is not None:
        user_prompt = manual_prompt

    # Button to trigger document conversion when clicked
    if st.button('**Convert to Markdown Format**', type='primary', use_container_width=True, icon="🚀"):
        # Create a temporary directory for the uploaded file
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)

        # Write the uploaded file content to a new file in the temporary directory
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Display a loading spinner while processing
        with st.spinner('Converting...'):
            # Call the conversion function with various parameters based on user settings and file information
            result = convert_to_markdown(
                f.name,
                ocr=enable_ocr,  # Whether OCR is enabled or not
                extract_image_desc=get_image_desc,  # If image descriptions should be extracted
                extract_via_llm=extract_text_through_llm,  # If text extraction through LLM is chosen
                ollama_model=vision_model,  # The selected vision model from the sidebar dropdown
                ollama_llm=ollama_avb,  # Boolean indicating if Ollama LLM is available and accessible
                llm_prompt=user_prompt,  # User-entered prompt or empty string based on conditions above
                enhance_image=enh_image  # Whether image enhancement should be applied before processing
            )

            # Display the converted Markdown content in the Streamlit app
            st.markdown(result)

            # Provide a download button for the user to save the result as a Markdown file
            st.download_button(
                '📥 Download Result in Markdown',
                result,
                file_name=f'{uploaded_file.name[:uploaded_file.name.rfind(".")]}.md',  # Change the extension to .md
                mime='text/markdown',
                use_container_width=True  # Use full container width for the button
            )
