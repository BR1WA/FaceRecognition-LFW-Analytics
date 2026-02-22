import sys

def extract_text(pdf_path):
    try:
        import pypdf
        print("Using pypdf")
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except ImportError:
        pass

    try:
        import PyPDF2
        print("Using PyPDF2")
        reader = PyPDF2.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except ImportError:
        pass

    print("Error: Neither pypdf nor PyPDF2 is installed.")
    return None

if __name__ == "__main__":
    pdf_path = "LFW_Facial_Recognition_Architectures.pdf.pdf"
    text = extract_text(pdf_path)
    if text:
        print("--- EXTRACTED TEXT START ---")
        print(text)
        print("--- EXTRACTED TEXT END ---")
