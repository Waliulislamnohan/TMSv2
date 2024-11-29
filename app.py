import streamlit as st
import cohere
from io import BytesIO
from dotenv import load_dotenv
import os
import pdfplumber
import pandas as pd
import re

# Load environment variables from .env file
load_dotenv()

# Initialize Cohere Client
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
if not COHERE_API_KEY:
    st.error("Cohere API key not found. Please set it in the .env file.")
    st.stop()

co = cohere.Client(COHERE_API_KEY)

# Function to make column names unique
def make_unique(columns):
    counts = {}
    new_columns = []
    for col in columns:
        if col is None:
            col = "Unnamed"
        col = str(col).strip()
        if col in counts:
            counts[col] += 1
            new_col = f"{col}.{counts[col]}"
            new_columns.append(new_col)
        else:
            counts[col] = 0
            new_columns.append(col)
    return new_columns

# Streamlit app title
st.title("Evaluate your document")

# File uploader
uploaded_file = st.file_uploader("", type=['pdf'])

if uploaded_file is not None:
    if uploaded_file.type != 'application/pdf':
        st.error("Please upload a valid PDF file.")
    else:
        try:
            # Extract tables and text from the PDF
            with pdfplumber.open(BytesIO(uploaded_file.read())) as pdf:
                tables = []
                text_content = ''
                for page_number, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + '\n'

                    page_tables = page.extract_tables()
                    for table_number, table in enumerate(page_tables, start=1):
                        if len(table) > 1 and len(table[0]) > 0:
                            df = pd.DataFrame(table[1:], columns=table[0])

                            if df.columns.duplicated().any():
                                df.columns = make_unique(df.columns)

                            tables.append(df)
            
            if not tables:
                st.error("No tables could be extracted from the PDF.")
                st.stop()

            # Combine all tables into one DataFrame
            full_table = pd.concat(tables, ignore_index=True)
            full_table.reset_index(drop=True, inplace=True)

            # Clean and prepare the data
            full_table.columns = [str(col).strip().lower() for col in full_table.columns]

            # Identify columns that might contain amounts
            amount_columns = [col for col in full_table.columns if 'amount' in col or 'price' in col or 'total' in col]

            if amount_columns:
                for col in amount_columns:
                    full_table[col] = full_table[col].astype(str).replace(r'[\$,]', '', regex=True)
                    full_table[col] = full_table[col].str.replace(',', '', regex=False)
                    full_table[col] = pd.to_numeric(full_table[col], errors='coerce')

                total_calculated = full_table[amount_columns].sum(numeric_only=True).sum()

                # Extract the total amount provided in the document using regex
                total_amounts = re.findall(r'Total Amount.*?([\d,\.]+)', text_content, re.IGNORECASE)
                if total_amounts:
                    total_provided = total_amounts[0].replace(',', '')
                    try:
                        total_provided = float(total_provided)
                        if abs(total_calculated - total_provided) < 0.01:
                            st.success("The total amount matches the sum of individual items.")
                        else:
                            st.error(f"Discrepancy found! Calculated total is {total_calculated:.2f}, but the document states {total_provided:.2f}.")
                    except ValueError:
                        st.error("Unable to convert the extracted total amount to a numeric value.")

            else:
                st.error("Could not find amount columns in the extracted data.")

            # Cohere analysis with brief, actionable feedback
            with st.spinner('Analyzing the budget document with Cohere...'):
                response = co.generate(
                    model='command-xlarge-nightly',
                    prompt=(
                        "You are an expert document reviewer. Look at the document and find key issues with pricing and any potential problems that could affect the project. "
                        "For pricing, compare the rates with normal market prices and note if anything is too high or too low. For potential problems, mention any missing details or risks. For market Comparison, Make a comparison between the quoted prices and the typical market rates for similar items or services. Highlight any significant discrepancies and provide reasoning if possible.\n\n"  
                        "Format your output like this:\n"
                        "- **Pricing**: Briefly check if the price is fair compared to the market. Point out if it's too expensive or too cheap.\n"
                        "- **Market Comparison**: Comparison of prices with market norms, including whether any items are priced unusually high or low. Each subsection should be within 1-2 lines.\n\n"
                        "- **Potential Problems**: List any issues like missing information, risks to the project, or things that could cause delays.\n\n"
                        "Example:\n"
                        "- **Pricing**: Lumber is 3% more expensive than usual, and the excavator rental is too cheap by 10%.\n"
                        "- **Market Comparison**: The cost of materials like steel is 10% higher than the market price due to a supplier-specific markup."
                        "- **Potential Problems**: The budget might not include extra fees for special permits. Soil test results are unclear, which could affect the foundation."
                    )
,
                    max_tokens=4096,
                    temperature=0.7,
                    k=0,
                    p=0.75,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop_sequences=["--END--"],
                    return_likelihoods='NONE'
                )
                validation_result = response.generations[0].text.strip()
            st.header("Analysis result: ")
            st.write(validation_result)

        except Exception as e:
            st.error(f"An error occurred while processing the PDF: {str(e)}")
            st.exception(e)


