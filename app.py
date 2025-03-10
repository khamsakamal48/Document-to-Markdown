import streamlit as st
from pathlib import Path
import base64
import ollama
import tempfile
from ollama import ChatResponse, chat
from httpx import ConnectError
import pandas as pd
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.base_models import InputFormat
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.pipeline_options import PictureDescriptionApiOptions
from docling.datamodel.pipeline_options import granite_picture_description
from docling_core.types.doc.document import PictureDescriptionData
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
)
from tempfile import NamedTemporaryFile

import os
import torch
torch.classes.__path__ = []

from sympy.codegen.ast import continue_

# Configure the Streamlit page settings with a title, icon, and layout.
st.set_page_config(
    page_title='Document to Markdown converter',
    page_icon=':document:',
    layout='wide'
)

# Set the main title for the web application.
st.title('Document to Markdown converter')
st.divider()

########################################################################################################################
#                                                   FUNCTIONS                                                          #
########################################################################################################################

def convert_to_markdown(file, ocr, extract_image_desc, extract_via_llm, ollama_model, ollama_llm, llm_prompt):
    # Vision Model through Ollama
    model = ollama_model

    if extract_via_llm:
        if file.lower().endswith('.jpeg') or file.lower().endswith('.jpg') or file.lower().endswith('.png'):

            # Convert images to base64
            base64_code = encode_image(file)

            prompt = llm_prompt

            conv_doc = ask_ollama(model, prompt, base64_code)

        else:
            st.error('Upload an Image only (.jpeg, .jpg, .png) for performing complete text extraction through an LLM.')

            conv_doc = ''

    else:
        input_doc_path = Path(file)


        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = ocr
        # pipeline_options.do_table_structure = True
        # pipeline_options.table_structure_options.do_cell_matching = True
        ocr_options = RapidOcrOptions(force_full_page_ocr=ocr)
        pipeline_options.ocr_options = ocr_options
        # pipeline_options.images_scale = 2
        # pipeline_options.generate_picture_images = True

        if extract_image_desc:
            pipeline_options.images_scale = 2
            pipeline_options.generate_picture_images = True
            pipeline_options.generate_page_images = True
            pipeline_options.do_picture_classification = True
            pipeline_options.do_picture_description = True

            # Enable connections to remote services
            pipeline_options.enable_remote_services = True

            if not ollama_llm:
                pipeline_options.picture_description_options = granite_picture_description
                pipeline_options.picture_description_options.prompt = (
                    "Describe the image in less than three sentences. Be concise and accurate."
                )

        doc_converter = (
            DocumentConverter(  # all of the below is optional, has internal defaults.
                allowed_formats = [
                    InputFormat.PDF,
                    InputFormat.IMAGE,
                    InputFormat.DOCX,
                    InputFormat.HTML,
                    InputFormat.PPTX,
                ],  # whitelist formats, non-matching files are ignored.
                format_options = {
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,  # pipeline options go here.
                        # backend=PyPdfiumDocumentBackend  # optional: pick an alternative backend
                    ),
                    InputFormat.DOCX: WordFormatOption(
                        pipeline_cls=SimplePipeline  # default for office formats and HTML
                    ),
                },
            )
        )

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
                    for pic in conv_res.document.pictures:
                        base64_code.append(
                            str(pic.image.uri).replace('AnyUrl(\'data:image/png;base64,', '').replace('data:image/png;base64,',
                                                                                                      ''))
                except AttributeError:
                    image_extracted = False

                    # Convert images to base64
                    base64_code = [encode_image(file)]

                for code in base64_code:
                    # if image_extracted:
                    #     # Describe the image when an image is found
                    #     content = 'Extract text from the image and then Describe the image in less than ten sentences. Be concise and accurate.'
                    #
                    # else:
                    #     # Use the entire document as page
                    #     content = """
                    #                 Please look at this image and extract all the text content. Format the output in markdown:
                    #                     - Use headers (# ## ###) for titles and sections
                    #                     - Use bullet points (-) for lists
                    #                     - Use proper markdown formatting for emphasis and structure
                    #                     - Preserve the original text hierarchy and formatting as much as possible
                    #               """
                    content = llm_prompt

                    annotations.append(ask_ollama(model, content, code))


            if image_extracted:
                annotations = ['<picture>\n\n'+a+'\n\n</picture>\n\n' for a in annotations]

                conv_doc = replace_with_img_desc(conv_res.document.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER), annotations)

            else:
                conv_doc = annotations[0]

        else:
            conv_doc = conv_res.document.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)

    return conv_doc


def ask_ollama(model, prompt, image):
    response: ChatResponse = chat(
        model=model,
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [image]
        },
    ])

    return response.message.content

def encode_image(image_path: str) -> str:
    """Convert image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def replace_with_img_desc(doc, annotations):
    lines = doc.split('\n')

    current_index = 0

    new_lines = []
    for line in lines:
        parts = line.split('<!-- image -->')
        if len(parts) > 1 and current_index < len(annotations):
            replacement_str = annotations[current_index]
            current_index += 1
            modified_line = replacement_str.join(parts)
        else:
            modified_line = line

        new_lines.append(modified_line)

    return '\n'.join(new_lines)

# Function to extract model names from the provided model information
@st.cache_resource
def get_vision_models():

    try:
        models_info = ollama.list()
        model_names = tuple(model.model for model in models_info['models'] if 'vision' in model.model or 'llava' in model.model)
        ollama_avbl = True

    # Handle error when Ollama is not installed or accessible over default port
    except ConnectError:
        st.warning('Ollama not installed or not accessible.')
        model_names = ()
        ollama_avbl = False

    return model_names, ollama_avbl

with st.sidebar:

    # Extract model names for selection
    available_models, ollama_avb = get_vision_models()

    model_desc = {
        'llava:latest': 'https://llava-vl.github.io',
        'granite3.2-vision:latest': 'https://www.ibm.com/new/announcements/ibm-granite-3-2-open-source-reasoning-and-vision',
        'llama3.2-vision:latest': 'https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/',
        'llama3.2-vision:11b': 'https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/'
    }

    # LLM model selection
    st.header('👁️ Select Vision model:')
    vision_model = st.sidebar.selectbox('**👁️ Select Vision model**', available_models, index=2, label_visibility='collapsed')

    st.link_button(f'Read More about `{vision_model}`', model_desc[vision_model], type='tertiary')

    st.header('')

    # Manual Prompting
    st.header('💬 Provide a prompt for better results (optional)')
    manual_prompt = st.text_area(
        '💬 **Provide a prompt for better results (optional)**',
        value='',
        help='Enter a prompt manually for better results. Leave empty to use the default prompt.',
        label_visibility='collapsed'
    )

# Upload Document to convert
uploaded_file = st.file_uploader(
    'Upload Document to Convert',
    type=['pdf', 'jpeg', 'jpg', 'png', 'ppt', 'pptx', 'doc', 'docx'],
    label_visibility='collapsed'
)

# Start conversion
if uploaded_file:

    st.subheader('Additional Parameters:')
    col1, col2, col3 = st.columns(3)

    user_prompt = ''

    with col1:
        # Enable OCR
        enable_ocr = st.checkbox('Enable OCR?',
                                 value=True,
                                 help='Enable conversion of of typed, handwritten, or printed text into machine-encoded text')

    with col2:
        # Extract text through LLM
        extract_text_through_llm = st.checkbox('Extract text through LLM?', help='Use when OCR is unable to extract text.')

    with col3:
        if not extract_text_through_llm:
            # Get Image description?
            get_image_desc = st.checkbox('Get image description?', help='Pass the images in the document to an AI model to get its description instead of placeholder.')

        else:
            get_image_desc = False

    if extract_text_through_llm:
        user_prompt = """
        You will be provided with an image containing text and structured content. Your task is to extract, format, and present the content in a structured and readable Markdown format while preserving accuracy and layout.

        ## **Task Breakdown:**  
        
        ### **1. Extract Text Using OCR**  
           - Perform Optical Character Recognition (OCR) to extract all text from the image with high accuracy.  
           - Retain all special characters, symbols, punctuation, and text formatting as they appear in the original image.  
        
        ### **2. Preserve Structure and Formatting**  
           - Maintain the original content hierarchy, including section divisions, headings, subheadings, and paragraphs.  
           - If the image contains structured elements (e.g., **tables, lists, headings, bullet points, numbering**), accurately represent them in the output.  
        
        ### **3. Format Output in Markdown**  
           - Use appropriate Markdown syntax to structure the extracted content:  
             - **Headings:** Use `#`, `##`, `###`, etc., for section titles.  
             - **Lists:**  
               - Use `-` for unordered (bullet) lists.  
               - Use `1.` for ordered (numbered) lists.  
             - **Tables:** Represent tables using Markdown table syntax.  
             - **Emphasis:** Apply `**bold**` and `*italics*` formatting where applicable.  
           - Ensure spacing and line breaks are well-structured for readability.  
        
        ### **4. Accuracy and Output Guidelines**  
           - Verify the extracted text for completeness and correctness before formatting.  
           - If the image lacks a clear structure, present the extracted text in a **well-organized** and **easy-to-read** Markdown format.  
           - **Do not provide any descriptions** of the image itself—only extract and format the text content.  
        
        ## **Example Markdown Output:**  
        ```markdown  
        # Document Title  
        
        ## Section 1: Introduction  
        This is a sample paragraph with *italicized text* and **bold text**.  
        
        ### Key Points:  
        - First bullet point  
        - Second bullet point  
        
        ## Data Table  
        | Column 1 | Column 2 | Column 3 |  
        |----------|----------|----------|  
        | Data 1   | Data 2   | Data 3   |  
        | Data 4   | Data 5   | Data 6   |  
        ```
        """

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

    if manual_prompt is not None:
        user_prompt = manual_prompt

    if st.button('**Convert to Markdown Format**', type='primary', use_container_width=True, icon="🚀"):
        # Call convert_to_markdown function using the created temp path
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())

        with st.spinner('Converting...'):
            result = convert_to_markdown(f.name, ocr=enable_ocr, extract_image_desc=get_image_desc,
                                         extract_via_llm=extract_text_through_llm, ollama_model=vision_model,
                                         ollama_llm=ollama_avb, llm_prompt=user_prompt)
            st.markdown(result)



            # Download the conversion
            st.download_button(
                '📥 Download Result in Markdown',
                result,
                file_name=f'{uploaded_file.name[:uploaded_file.name.rfind('.')]}.md',
                mime='text/markdown',
                use_container_width=True
            )
