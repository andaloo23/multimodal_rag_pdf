"""
PDF Q&A System - Multimodal RAG
This script processes a PDF and creates an interactive Q&A system.
"""

import os
import base64
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from pydantic import BaseModel, Field
import anthropic
import instructor
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.anthropic import Anthropic
from tqdm import tqdm
from pdf2image import pdfinfo_from_path
import time
import sys
import threading

# Load environment variables
load_dotenv(verbose=True, dotenv_path=".env")

# Configuration
PDF_PATH = "data/introduction_to_computing_systems.pdf"
MODEL_ID_HAIKU = "claude-3-haiku-20240307"
MODEL_ID_SONNET = "claude-3-5-sonnet-20241022"
USE_MODEL = MODEL_ID_HAIKU  # Using cheaper model for faster processing

# Helper function for progress indication
class ProgressSpinner:
    def __init__(self, message="Processing"):
        self.message = message
        self.spinning = False
        self.thread = None
        
    def spin(self):
        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        idx = 0
        while self.spinning:
            sys.stdout.write(f'\r  {spinner[idx]} {self.message}')
            sys.stdout.flush()
            idx = (idx + 1) % len(spinner)
            time.sleep(0.1)
        sys.stdout.write('\r' + ' ' * (len(self.message) + 10) + '\r')
        sys.stdout.flush()
    
    def start(self):
        self.spinning = True
        self.thread = threading.Thread(target=self.spin)
        self.thread.start()
    
    def stop(self):
        self.spinning = False
        if self.thread:
            self.thread.join()

print("="*80)
print("PDF Q&A SYSTEM - MULTIMODAL RAG")
print("="*80)

# Step 1: Extract Content from PDF
print("\n[1/6] Extracting content from PDF")
print(f"Processing: {PDF_PATH}")

# Get PDF page count for progress estimation
try:
    pdf_info = pdfinfo_from_path(PDF_PATH)
    num_pages = pdf_info.get("Pages", "unknown")
    print(f"PDF has {num_pages} pages")
    
    if isinstance(num_pages, int) and num_pages > 100:
        print(f"Large document detected. This may take 10-30 minutes")
        print(f"Estimated time: {num_pages * 1:.0f}-{num_pages * 2:.0f} seconds")
    
    # Always use 'auto' strategy - it's smart about when to use OCR
    # For large docs, we'll skip OCR with the ocr_languages=None parameter
    use_strategy = "auto"
    use_ocr = num_pages < 100 if isinstance(num_pages, int) else True
    
except Exception as e:
    print(f"Could not determine page count: {e}")
    use_strategy = "auto"
    use_ocr = True
    num_pages = "unknown"

print(f"Starting extraction (strategy: {use_strategy})")
print(f"This will process the entire PDF and may take a while")
start_time = time.time()

# Start progress spinner
spinner = ProgressSpinner(f"Extracting content from {num_pages} pages" if isinstance(num_pages, int) else "Extracting content")
spinner.start()

try:
    chunks = partition_pdf(
        filename=PDF_PATH,
        infer_table_structure=False,  # Disabled for speed
        strategy=use_strategy,  # Auto strategy intelligently chooses extraction method
        extract_image_block_types=[],  # Skip images for now
        extract_image_block_to_payload=False,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )
finally:
    spinner.stop()

elapsed_time = time.time() - start_time
print(f"Extraction completed in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")

print(f"DEBUG: Extracted {len(chunks)} chunks")
if len(chunks) > 0:
    print(f"DEBUG: First chunk type: {type(chunks[0])}")
    print(f"DEBUG: First chunk has {len(chunks[0].text) if hasattr(chunks[0], 'text') else 0} characters")
else:
    print(f"No chunks extracted")
    print(f"Attempting alternative extraction method")
    
    # Fallback: try with different settings
    spinner2 = ProgressSpinner("Trying alternative extraction")
    spinner2.start()
    try:
        chunks = partition_pdf(
            filename=PDF_PATH,
            strategy="hi_res",  # Force hi_res
            infer_table_structure=False,
            extract_image_block_types=[],
            extract_image_block_to_payload=False,
            chunking_strategy="basic",  # Simpler chunking
        )
    finally:
        spinner2.stop()
    
    print(f"Alternative method extracted {len(chunks)} chunks")

print(f"Extracted {len(chunks)} chunks from the PDF")

# Step 2: Separate Text, Tables, and Images
print("\n[2/6] Separating text, tables, and images")

# Extract text chunks
text_chunks = []
for chunk in chunks:
    if "CompositeElement" in str(type(chunk)):
        text_chunks.append(chunk)

# Extract tables
def get_tables_info(chunks):
    tables_info = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_elements = chunk.metadata.orig_elements
            for el in chunk_elements:
                if "Table" in str(type(el)):
                    tables_info.append({
                        "text_as_html": el.metadata.text_as_html,
                        "image_base64": el.metadata.image_base64,
                    })
    return tables_info

# Extract images
def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

tables_info = get_tables_info(chunks)
images = get_images_base64(chunks)

print(f"Found {len(text_chunks)} text chunks")
print(f"Found {len(tables_info)} tables")
print(f"Found {len(images)} images")

# Step 3: Set Up LLM Client
print("\n[3/6] Setting up LLM client")

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

instructor_client = instructor.from_anthropic(
    client,
    max_tokens=4096,
    model=USE_MODEL
)

print(f"Using model: {USE_MODEL}")

# Step 4: Generate Summaries for All Content
print("\n[4/6] Generating summaries and descriptions")
print("This will take several minutes depending on document size")

class NodeSummaryResponse(BaseModel):
    """Response model for summary object of a node"""
    summary: str = Field(
        ...,
        description="A detailed summary of the table or text. 8 minimum to 10 sentences max.")

class NodeSummariser:
    """Document summariser"""
    def __init__(self, client):
        self.client = client

    def summarise(self, document_text: str) -> NodeSummaryResponse:
        """Summarise a document text"""
        try:
            summary, _ = self.client.chat.completions.create_with_completion(
                messages=[{
                    "role": "user",
                    "content": f"Give a very detailed summary of the following text or table: {document_text}. "
                               f"Respond only with the summary, no additional comment."
                }],
                response_model=NodeSummaryResponse
            )
            return summary
        except Exception as e:
            print(f"Error: {e}")
            return NodeSummaryResponse(summary="Error generating summary")

class ImageDescriptionResponse(BaseModel):
    """Response model for description of an image"""
    description: str = Field(
        ...,
        description="A detailed description of a given image. 8 minimum to 10 sentences max.")

class ImageDescriber:
    """Image describer"""
    def __init__(self, client):
        self.client = client

    def describe(self, base64_image: str) -> ImageDescriptionResponse:
        """Return the description of an image"""
        try:
            # Skip very small images
            if len(base64_image) < 10000:
                return ImageDescriptionResponse(description="Image too small to process")
            
            input_image = instructor.Image.from_raw_base64(base64_image)
            description, _ = self.client.chat.completions.create_with_completion(
                messages=[{
                    "role": "user",
                    "content": ["Describe in detail what is in this image?", input_image]
                }],
                response_model=ImageDescriptionResponse
            )
            return description
        except Exception as e:
            print(f"Error: {e}")
            return ImageDescriptionResponse(description="Error generating image description")

# Generate summaries for text chunks (limit to first 30 for speed)
print("Summarizing text chunks")
summariser = NodeSummariser(instructor_client)
text_summaries = []

for chunk in tqdm(text_chunks[:30], desc="Text chunks"):
    summary = summariser.summarise(chunk.text)
    text_summaries.append({
        "original_text": chunk.text,
        "summary": summary.summary,
        "type": "text"
    })

# Generate summaries for tables
print("Summarizing tables")
table_summaries = []

for table in tqdm(tables_info, desc="Tables"):
    summary = summariser.summarise(table["text_as_html"])
    table_summaries.append({
        "original_text": table["text_as_html"],
        "summary": summary.summary,
        "type": "table"
    })

# Generate descriptions for images (limit to first 10 for speed)
print("Describing images")
describer = ImageDescriber(instructor_client)
image_descriptions = []

for img in tqdm(images[:10], desc="Images"):
    description = describer.describe(img)
    if description.description != "Image too small to process":
        image_descriptions.append({
            "image_base64": img,
            "description": description.description,
            "type": "image"
        })

print(f"Generated {len(text_summaries)} text summaries")
print(f"Generated {len(table_summaries)} table summaries")
print(f"Generated {len(image_descriptions)} image descriptions")

# Step 5: Build Vector Database for Semantic Search
print("\n[5/6] Building vector database")

# Set up LlamaIndex with Anthropic
Settings.llm = Anthropic(model=USE_MODEL, api_key=os.getenv("ANTHROPIC_API_KEY"))

# Use HuggingFace embeddings (free, no API key required)
print("Using HuggingFace embeddings (this may download a model on first run)")
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Create documents for indexing
documents = []

# Add text summaries
for item in text_summaries:
    doc = Document(
        text=f"Summary: {item['summary']}\n\nOriginal Content: {item['original_text']}",
        metadata={"type": "text", "content_type": "text_chunk"}
    )
    documents.append(doc)

# Add table summaries
for item in table_summaries:
    doc = Document(
        text=f"Table Summary: {item['summary']}\n\nTable Content: {item['original_text']}",
        metadata={"type": "table", "content_type": "table"}
    )
    documents.append(doc)

# Add image descriptions
for item in image_descriptions:
    doc = Document(
        text=f"Image Description: {item['description']}",
        metadata={"type": "image", "content_type": "image"}
    )
    documents.append(doc)

print(f"Created {len(documents)} documents for indexing")

# Build the vector index
print("Building vector index")
index = VectorStoreIndex.from_documents(documents, show_progress=False)
print("Vector index built")

# Step 6: Create Query Engine and Start Q&A
print("\n[6/6] Setting up query engine")

query_engine = index.as_query_engine(
    similarity_top_k=5,  # Retrieve top 5 most relevant chunks
    response_mode="compact"
)

print("Query engine ready\n")

# Interactive Q&A Function
def ask_question(question: str):
    """Ask a question about the PDF"""
    print(f"\n{'='*80}")
    print(f"QUESTION: {question}")
    print(f"{'='*80}\n")
    
    response = query_engine.query(question)
    
    print("ANSWER:")
    print(response.response)
    print(f"\n{'='*80}\n")
    
    return response

# Main Interactive Loop
print("="*80)
print("READY TO ANSWER QUESTIONS!")
print("="*80)
print("\nYou can now ask questions about your PDF.")
print("Type 'quit', 'exit', or 'q' to stop.\n")

# Interactive loop
while True:
    try:
        question = input("Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using the PDF Q&A System!")
            break
        
        if not question:
            continue
        
        ask_question(question)
        
    except KeyboardInterrupt:
        print("\n\nExiting")
        break
    except Exception as e:
        print(f"\nError: {e}")
        print("Please try again.\n")

