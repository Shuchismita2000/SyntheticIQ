import streamlit as st
import pandas as pd
import json
import re
import subprocess


# Initialize Faker (if needed for synthetic data generation)
from faker import Faker
fake = Faker()


# Function to interact with Ollama via CLI
def get_schema_from_ollama(context):
    """
    Generates a dataset schema using Ollama's CLI.
    Args:
        context (str): Business context input by the user.
    Returns:
        str: JSON-formatted schema or raw response.
    """
    prompt = f"""
    You are a data generation assistant. Based on the following business context:
    {context}


    Generate a dataset schema with fields, data types, and example values.
    The output must be valid JSON format only, without any additional text or explanation.
    """
    try:
        # Call Ollama CLI
        result = subprocess.run(
            ["ollama", "run", "llama3.2", prompt],
            text=True,
            capture_output=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error calling Ollama CLI: {e}"


# Function to extract valid JSON from AI response
def extract_json(response):
    """
    Extracts JSON data from a potentially mixed response using regex.
    Args:
        response (str): The raw response from the AI.
    Returns:
        dict or None: Parsed JSON or None if extraction fails.
    """
    try:
        json_part = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL).group(0)
        data = json.loads(json_part)
        return data
    except (AttributeError, json.JSONDecodeError):
        return None


# Function to convert JSON schema to a flattened table
def json_to_table(schema):
    """
    Converts JSON schema to a tabular format.
    Args:
        schema (JSON): The JSON schema from the raw response.
    Returns:
        pd.DataFrame: A pandas DataFrame in tabular format.
    """
    try:
        # Flatten JSON to handle nested structures
        if isinstance(schema, dict):
            df = pd.json_normalize(schema)
        elif isinstance(schema, list):
            df = pd.json_normalize(schema)
        else:
            raise ValueError("Unexpected JSON structure")
        return df
    except Exception as e:
        st.error(f"Error processing schema: {e}")
        return pd.DataFrame()


# Streamlit UI
st.title("AI-Driven Dataset Generator")
st.markdown("Generate synthetic datasets based on business contexts using AI.")

# Input Section
context = st.text_area("Describe the business context for your dataset:")
num_rows = st.number_input("How many rows of data do you need?", min_value=1, step=1, value=10)

# Button to Generate Schema and Dataset
if st.button("Generate Dataset"):
    if context.strip() == "":
        st.error("Please provide a business context.")
    else:
        # Step 1: Get schema from Ollama
        st.info("Generating schema using AI...")
        schema_response = get_schema_from_ollama(context)

        # Step 2: Attempt to extract and parse JSON
        schema = extract_json(schema_response)
        if schema is None:
            st.error("Failed to parse schema. Ensure the AI response is valid JSON.")
        else:
            # Step 3: Generate dataset
            dataset = json_to_table(schema)

            if dataset.empty:
                st.error("Failed to generate dataset. Please review the AI schema response.")
            else:
                st.success("Dataset generated successfully!")
                st.dataframe(dataset)

                # Step 4: Option to download dataset
                csv = dataset.to_csv(index=False)
                st.download_button(
                    label="Download Dataset as CSV",
                    data=csv,
                    file_name="generated_dataset.csv",
                    mime="text/csv"
                )


