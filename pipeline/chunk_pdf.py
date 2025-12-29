from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber

def chunk_pdf_loader(uploaded_file, chunk_size=800, chunk_overlap=200):
    loader = PyPDFLoader(uploaded_file)
    document = loader.load()

    print(document)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(document)
    return chunks



def chunk_pdf(uploaded_file, chunk_size=800, chunk_overlap=200):
    docs = []
    with pdfplumber.open(uploaded_file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = splitter.split_text(text)
            for j, chunk in enumerate(chunks):
                docs.append({"text": chunk, "page": i+1, "chunk_index": j, "filename": uploaded_file.name})
    return docs
    # return chunks



if __name__=="__main__":
    l = chunk_pdf_loader("path_to_pdf")
    print("\n\n\n\n",l,"\n\n\n\n\n\n\n\n")
    p = chunk_pdf("path_to-pdf")
    print("\n\n\n\n",p)

    # print("\n\n\n\n\n",l,"\n\n\n",p)

# print("chunking done..")