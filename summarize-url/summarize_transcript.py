#!/usr/bin/env python
# coding: utf-8


import requests
from bs4 import BeautifulSoup
from config import Config
from urllib.parse import urlparse, urljoin
from urllib.request import urlopen
import openai
import certifi
import json
import openai
import os

summaries = []

cfg = Config()

FMP_KEY=os.getenv("FMP_KEY")

#The company to summarize earnings
symbol="FLNG"
year="2022"
quarter="4"

# Function to check if the URL is valid
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


# Function to sanitize the URL
def sanitize_url(url):
    return urljoin(url, urlparse(url).path)


# Define and check for local file address prefixes
def check_local_file_access(url):
    local_prefixes = ['file:///', 'file://localhost', 'http://localhost', 'https://localhost']
    return any(url.startswith(prefix) for prefix in local_prefixes)


def get_response(url, headers=cfg.user_agent_header, timeout=10):
    try:
        # Restrict access to local files
        if check_local_file_access(url):
            raise ValueError('Access to local files is restricted')

        # Most basic check if the URL is valid:
        if not url.startswith('http://') and not url.startswith('https://'):
            raise ValueError('Invalid URL format')

        sanitized_url = sanitize_url(url)

        response = requests.get(sanitized_url, headers=headers, timeout=timeout)

        # Check if the response contains an HTTP error
        if response.status_code >= 400:
            return None, "Error: HTTP " + str(response.status_code) + " error"

        return response, None
    except ValueError as ve:
        # Handle invalid URL format
        return None, "Error: " + str(ve)

    except requests.exceptions.RequestException as re:
        # Handle exceptions related to the HTTP request (e.g., connection errors, timeouts, etc.)
        return None, "Error: " + str(re)


def scrape_text(url):
    """Scrape text from a webpage"""
    response, error_message = get_response(url)
    if error_message:
        return error_message

    soup = BeautifulSoup(response.text, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text


def extract_hyperlinks(soup):
    """Extract hyperlinks from a BeautifulSoup object"""
    hyperlinks = []
    for link in soup.find_all('a', href=True):
        hyperlinks.append((link.text, link['href']))
    return hyperlinks


def format_hyperlinks(hyperlinks):
    """Format hyperlinks into a list of strings"""
    formatted_links = []
    for link_text, link_url in hyperlinks:
        formatted_links.append(f"{link_text} ({link_url})")
    return formatted_links


def scrape_links(url):
    """Scrape links from a webpage"""
    response, error_message = get_response(url)
    if error_message:
        return error_message

    soup = BeautifulSoup(response.text, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    hyperlinks = extract_hyperlinks(soup)

    return format_hyperlinks(hyperlinks)



def split_text(text, max_length=3500):
    """Split text into chunks of a maximum length"""
    paragraphs = text.split("\n")
    current_length = 0
    current_chunk = []

    for paragraph in paragraphs:
        # If the paragraph itself is longer than max_length, split it into smaller substrings
        while len(paragraph) > max_length:
            substring = paragraph[:max_length]
            yield substring
            paragraph = paragraph[max_length:]
        
        # Check if adding the paragraph to the current chunk exceeds max_length
        if current_length + len(paragraph) + 1 <= max_length:
            current_chunk.append(paragraph)
            current_length += len(paragraph) + 1
        else:
            yield "\n".join(current_chunk)
            current_chunk = [paragraph]
            current_length = len(paragraph) + 1

    if current_chunk:
        yield "\n".join(current_chunk)


def create_message(chunk, question):
    """Create a message for the user to summarize a chunk of text"""
    return {
        "role": "user",
        "content": f"\"\"\"{chunk}\"\"\" Using the above text, please answer the following question: \"{question}\" -- if the question cannot be answered using the text, please summarize the text. " \
            " Respond starting with a short and creative title that summarizes the text. Insert a new line after the title.  Do not include the word Title in your title.  Do not use bullet points in the summary " \
            "Use a writing style that is fluid, simple, easy to understand, direct and facts oriented.   \n " \
            " And use a writing style that a 5th grade student could understand. Don't use terms like \"The article\" instead say \"The author \"" \
        }

def create_message_final_summary(chunk, question):
    """Create a message for the user to summarize a chunk of text"""
    return {
        "role": "user",
        "content": f"\"\"\"{chunk}\"\"\" Using the above text, please answer the following question: \"{question}\" -- if the question cannot be answered using the text, please summarize the text in full paragraph form " \
        "Please do not use bullet points.  Full and complete paragraphs only\n.  Don't use terms like \"The article \" or \"One article \"" \
        "Use a writing style that is fluid, simple, easy to understand, direct and facts oriented. \n " \
        "Respond in a format like this : \n" \
        "Executive Summary: \n" 
       
            
    }

def summarize_text(text, question):
    """Summarize text using the LLM model"""
    if not text:
        return "Error: No text to summarize"

    text_length = len(text)
    print(f"Text length: {text_length} characters")


    chunks = list(split_text(text))

    for i, chunk in enumerate(chunks):
        messages = [create_message(chunk, question)]

        response = openai.ChatCompletion.create(
                    model=cfg.fast_llm_model,
                    messages=messages,
                    temperature=0,
                    max_tokens=300
                )
            
        summaries.append(response.choices[0].message["content"])
        #print("\n" + str(i) + " " + response.choices[0].message["content"])

    combined_summary = "\n".join(summaries)
    if len(chunks) > 1 :
        messages = [create_message_final_summary(combined_summary, question)]

        response = openai.ChatCompletion.create(
                    model=cfg.fast_llm_model,
                    messages=messages,
                    temperature=0,
                    max_tokens=600
                )
        final_summary =response.choices[0].message["content"]

    else:
        final_summary = combined_summary

    return final_summary


def get_text_summary(url, question):
    """Return the results of a google search"""
    text = scrape_text(url)
    summary = summarize_text(text, question)
    return summary

#Get the text to summarize
url = "https://financialmodelingprep.com/api/v3/earning_call_transcript/"+symbol+"?quarter="+quarter+"&year="+year+"&apikey="+FMP_KEY
response = urlopen(url, cafile=certifi.where())
data = response.read().decode("utf-8")
js = json.loads(data)
fa=js[0]
text=fa["content"]

conclusion=summarize_text(text, "what are the main points")
print("Executive summary: \n" + conclusion)
print("\nDetailed summary :")
for i in range(len(summaries)):
    print(str(i) + " " + summaries[i] + "\n")
