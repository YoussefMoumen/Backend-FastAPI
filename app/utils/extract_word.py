import docx

def extract_text_from_word(file):
    doc = docx.Document(file)
    return [para.text for para in doc.paragraphs if para.text.strip()]
