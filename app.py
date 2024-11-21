# app.py

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
    """
    Make column names unique by appending a suffix to duplicate names.
    Handle None or empty column names gracefully.
    """
    counts = {}
    new_columns = []
    for col in columns:
        # Handle None or empty column names
        if col is None:
            col = "Unnamed"
        col = str(col).strip()  # Convert to string and remove extra spaces
        if col in counts:
            counts[col] += 1
            new_col = f"{col}.{counts[col]}"
            new_columns.append(new_col)
        else:
            counts[col] = 0
            new_columns.append(col)
    return new_columns

# Streamlit app title
st.title("Government Budget Validator")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file containing the budget document", type=['pdf'])

if uploaded_file is not None:
    # Check if the uploaded file is a PDF
    if uploaded_file.type != 'application/pdf':
        st.error("Please upload a valid PDF file.")
    else:
        try:
            # Extract tables and text from the PDF
            with pdfplumber.open(BytesIO(uploaded_file.read())) as pdf:
                tables = []
                text_content = ''
                for page_number, page in enumerate(pdf.pages, start=1):
                    # Extract text from the page
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + '\n'

                    # Extract tables from the page
                    page_tables = page.extract_tables()
                    for table_number, table in enumerate(page_tables, start=1):
                        if len(table) > 1 and len(table[0]) > 0:
                            # Proceed only if table has more than one row and headers are present
                            df = pd.DataFrame(table[1:], columns=table[0])

                            # Handle duplicate column names
                            if df.columns.duplicated().any():
                                st.warning(f"Duplicate column names found in table {table_number} on page {page_number}. Making column names unique.")
                                df.columns = make_unique(df.columns)

                            tables.append(df)
                        else:
                            st.warning(f"Skipping an empty or irregular table {table_number} on page {page_number}.")

            if not tables:
                st.error("No tables could be extracted from the PDF.")
                st.stop()

            # Combine all tables into one DataFrame
            full_table = pd.concat(tables, ignore_index=True)
            full_table.reset_index(drop=True, inplace=True)

            # Display the extracted table
            st.subheader("Extracted Financial Data")
            st.write(full_table)

            # Clean and prepare the data
            # Rename columns to standard names for consistency
            full_table.columns = [str(col).strip().lower() for col in full_table.columns]

            # Display column names for debugging
            st.write("Column names:", full_table.columns.tolist())

            # Identify columns that might contain amounts
            amount_columns = [col for col in full_table.columns if 'amount' in col or 'price' in col or 'total' in col]

            if amount_columns:
                # Process each amount column
                for col in amount_columns:
                    # Remove currency symbols and commas
                    full_table[col] = full_table[col].replace(r'[\$,]', '', regex=True)
                    full_table[col] = full_table[col].str.replace(',', '', regex=False)
                    # Convert to numeric
                    full_table[col] = pd.to_numeric(full_table[col], errors='coerce')

                # Sum the amounts
                total_calculated = full_table[amount_columns].sum(numeric_only=True).sum()

                # Extract the total amount provided in the document using regex
                # This regex looks for lines like "Total Amount: 1,000,000" or similar
                total_amounts = re.findall(r'Total Amount.*?([\d,\.]+)', text_content, re.IGNORECASE)
                if total_amounts:
                    total_provided = total_amounts[0].replace(',', '')
                    try:
                        total_provided = float(total_provided)
                        # Compare calculated total with provided total
                        if abs(total_calculated - total_provided) < 0.01:
                            st.success("The total amount matches the sum of individual items.")
                        else:
                            st.error(f"Discrepancy found! Calculated total is {total_calculated:.2f}, but the document states {total_provided:.2f}.")
                    except ValueError:
                        st.error("Unable to convert the extracted total amount to a numeric value.")
                else:
                    st.warning("Total amount provided in the document could not be identified from text.")

                # Display the total calculated amount
                st.subheader("Calculated Total Amount")
                st.write(f"{total_calculated:.2f}")

            else:
                st.error("Could not find amount columns in the extracted data.")

            # Additional validation using Cohere API if needed
            with st.spinner('Analyzing the budget document with Cohere...'):
                response = co.generate(
                    model='command-xlarge-nightly',
                    prompt=(
                        "You are a financial analyst specializing in government budgets. "
                        "Analyze the following budget document for any inconsistencies in the financial data, especially discrepancies in totals. "
                        "Provide a detailed report highlighting any issues found and recommendations:\n\n"
                        f"{text_content}\n\n"
                        "Provide your analysis below."
                    ),
                    max_tokens=500,
                    temperature=0.7,
                    k=0,
                    p=0.75,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop_sequences=["--END--"],
                    return_likelihoods='NONE'
                )
                validation_result = response.generations[0].text.strip()

            # Display the Cohere analysis
            st.subheader("Cohere Analysis")
            st.write(validation_result)

        except Exception as e:
            st.error(f"An error occurred while processing the PDF: {str(e)}")
            st.exception(e)
