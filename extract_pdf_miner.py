from pdfminer.high_level import extract_text

def read_pdf(pdf_path):
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    pdf_path = "LFW_Facial_Recognition_Architectures.pdf.pdf"
    text = read_pdf(pdf_path)
    if text:
        print("--- START TEXT ---")
        print(text[:2000])  # Print first 2000 chars to verify
        print("--- END TEXT ---")
    else:
        print("No text extracted.")
