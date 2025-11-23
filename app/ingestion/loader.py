import uuid
from typing import List
from app.logger import logging
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredImageLoader
)
from app.dataclasses import RawDocument
import fitz
from PIL import Image
import io,os
import pytesseract

class Documentloader:
    """LangChain-powered loader but controlled by OUR code
    """
    def __init__(self,data_path:str):
        self.data_path=data_path
    
    @staticmethod
    def extract_text_from_pdf_images(pdf_path:str)->str:
        """
        Extract images from PDF pages and run OCR on them
        """
        try:
            logging.info("This is the helper function to extract text from images")
            text=""
            doc=fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page=doc[page_num]

                image_list=page.get_images(full=True)

                for image_index,img in enumerate(image_list):
                    xref=img[0]
                    base_image=doc.extract_image(xref)

                    image_bytes=base_image["image"]

                    image=Image.open(io.BytesIO(image_bytes))

                    text+=pytesseract.image_to_string(image)
                    text += pytesseract.image_to_string(image)
                    # print("OCR SAMPLE:", text[:200])  

            return text.strip()
        except Exception as e:
            logging.error(f"Error in extract_text_from_pdf_images function with {e}")

    
    def load(self)->List[RawDocument]:
        try:
            logging.info("laoding the data into raw document started")
            docs=[]

            pdf_loader=DirectoryLoader(path=self.data_path,
                        glob="**/*.pdf",
                        loader_cls=PyPDFLoader)
            for loc_docs in pdf_loader.load():
                text=loc_docs.page_content

                source=loc_docs.metadata.get("source")
                if source:
                    if os.path.isabs(source) or os.path.exists(source):
                        pdf_path = source
                    else:
                        pdf_path = os.path.join(self.data_path, source)
                    image_text=self.extract_text_from_pdf_images(pdf_path=pdf_path)
                    text = text + "\n" + image_text
                
                docs.append(
                    RawDocument(
                        id=str(uuid.uuid4()),
                    text=text,
                    metadata=loc_docs.metadata
                    )
                )

            other_loaders=[
                 DirectoryLoader(self.data_path, glob="**/*.txt", loader_cls=TextLoader),
                DirectoryLoader(self.data_path, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader),
                DirectoryLoader(
                    self.data_path,
                    glob="**/*.[pj][pn]g",
                    loader_cls=UnstructuredImageLoader 
            )
            ]

            for loaders in other_loaders:
                for loc_doc in loaders.load():
                    docs.append(
                        RawDocument(
                            id=str(uuid.uuid4()),
                            text=loc_doc.page_content,
                            metadata=loc_doc.metadata
                        )
                    )
            return docs
        except Exception as e:
            logging.error(f"Error in load function with {e}")
        

    def start_data_loading(self):
        try:
            logging.info("Started data loading with data")
            docs=self.load()
            logging.info(f"documents length {len(docs)}")
            return docs
        except Exception as e:
            logging.error(f"Error in start_data_loading function {e}")
        
