import requests
from bs4 import BeautifulSoup
from fpdf import FPDF
import re
import unicodedata
import os

# Create a directory to store PDFs
os.makedirs("policy_pdfs", exist_ok=True)

policy_links = {
    "Student Conduct Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-conduct-policy",
    "Academic Schedule Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/academic-schedule-policy",
    "Student Attendance Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-attendance-policy",
    "Student Appeals Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-appeals-policy",
    "Academic Standing Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-standing-policy",
    "Transfer Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/transfer-policy",
    "Admissions Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/admissions-policy",
    "Final Grade Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/final-grade-policy",
    "Registration Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy",
    "Sports and Wellness Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/sport-and-wellness-facilities-and",
    "Scholarship and Financial Assistance Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/scholarship-and-financial-assistance",
    "Student Engagement Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-engagement-policy",
    "International Student Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-policy",
    "Graduation Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduation-policy",
    "Student Counselling Services Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-counselling-services-policy",
    "Graduate Admissions Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-admissions-policy",
    "Graduate Academic Standing Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduate-academic-standing-procedure",
    "Graduate Final Grade Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-final-grade-policy",
    "Library Space Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/use-library-space-policy",
    "Examination Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/examination-policy"
}
# Create directory for PDFs
pdf_dir = "policy_pdfs"
os.makedirs(pdf_dir, exist_ok=True)

def clean_text(text):
    """Cleans and normalizes text."""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fetch_policy_text(url):
    """Fetches policy text with structured content."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        policy_content = []

        # Extract headings (h2, h3) - important for structured answers
        for heading in soup.find_all(['h2', 'h3']):
            policy_content.append("\n📌 " + heading.get_text(strip=True) + "\n")

        # Extract paragraphs
        for p in soup.find_all('p'):
            policy_content.append(p.get_text(strip=True))

        # Extract list items (bullet points)
        for ul in soup.find_all('ul'):
            for li in ul.find_all('li'):
                policy_content.append("• " + li.get_text(strip=True))  # Bullet point formatting

        # Extract table data (if applicable)
        for table in soup.find_all('table'):
            for row in table.find_all('tr'):
                row_data = [td.get_text(strip=True) for td in row.find_all('td')]
                policy_content.append(" | ".join(row_data))

        policy_text = "\n".join(policy_content)
        return clean_text(policy_text)
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def save_text_as_pdf(policy_name, text):
    """Saves extracted text as a PDF, handling Unicode encoding errors."""

    pdf_path = os.path.join(pdf_dir, f"{policy_name.replace(' ', '_')}.pdf")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    # Add policy title
    pdf.cell(200, 10, policy_name.encode("latin-1", "ignore").decode("latin-1"), ln=True, align='C')
    pdf.ln(10)

    def clean_text(text):
        """Removes unsupported Unicode characters before saving to PDF."""
        return unicodedata.normalize("NFKD", text).encode("latin-1", "ignore").decode("latin-1")

    for line in text.split("\n"):
        cleaned_line = clean_text(line)  # Apply Unicode filtering
        pdf.multi_cell(190, 8, cleaned_line)  # Keep structured format
        pdf.ln(2)

    pdf.output(pdf_path)
    return pdf_path


# Fetch and save policies as PDFs
pdf_files = []
for policy_name, url in policy_links.items():
    print(f"Downloading {policy_name}...")
    pdf_path = os.path.join(pdf_dir, f"{policy_name.replace(' ', '_')}.pdf")

    if os.path.exists(pdf_path):
        print(f"✅ {policy_name} PDF already exists. Skipping download.")
        pdf_files.append(pdf_path)
        continue

    policy_text = fetch_policy_text(url)

    if policy_text and len(policy_text) > 100:  # Ensure meaningful content
        pdf_path = save_text_as_pdf(policy_name, policy_text)
        pdf_files.append(pdf_path)
        print(f"✅ Saved: {pdf_path}")
    else:
        print(f"⚠️ Skipping {policy_name} due to insufficient content.")

print(f"\n✅ Successfully saved {len(pdf_files)} policy documents as PDFs in '{pdf_dir}/'")