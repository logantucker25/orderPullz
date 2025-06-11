import fitz  

def extract_text_from_pdf(path):

    # extract
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)

    # clean
    lines = text.splitlines()
    lines = [line.strip() for line in lines if line.strip()] 
    return "\n".join(lines)