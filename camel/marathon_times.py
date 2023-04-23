#!/usr/bin/env python
# coding: utf-8

# ## AutoGPT example finding Winning Marathon Times
# 
# * Implementation of https://github.com/Significant-Gravitas/Auto-GPT 
# * With LangChain primitives (LLMs, PromptTemplates, VectorStores, Embeddings, Tools)

# In[ ]:


# !pip install bs4
# !pip install nest_asyncio


# In[1]:


# General 
import pandas as pd
from langchain.experimental.autonomous_agents.autogpt.agent import AutoGPT
from langchain.chat_models import ChatOpenAI

from langchain.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
import asyncio
import nest_asyncio


# Needed synce jupyter runs an async eventloop
nest_asyncio.apply()


# In[2]:


llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1.0)


# ### Set up tools
# 
# * We'll set up an AutoGPT with a `search` tool, and `write-file` tool, and a `read-file` tool, and a web browsing tool

# Define any other `tools` you want to use here

# In[3]:


# Tools
from typing import Optional
from langchain.agents import tool
from langchain.tools.file_management.read import ReadFileTool
from langchain.tools.file_management.write import WriteFileTool

@tool
def process_csv(csv_file_path: str, instructions: str, output_path: Optional[str] = None) -> str:
    """Process a CSV by with pandas in a limited REPL. Only use this after writing data to disk as a csv file. Any figures must be saved to disk to be viewed by the human. Instructions should be written in natural language, not code. Assume the dataframe is already loaded."""
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        return f"Error: {e}"
    agent = create_pandas_dataframe_agent(llm, df, max_iterations=30, verbose=True)
    if output_path is not None:
        instructions += f" Save output to disk at {output_path}"
    try:
        return agent.run(instructions)
    except Exception as e:
        return f"Error: {e}"



import json
from duckduckgo_search import ddg


@tool
def web_search(query: str, num_results: int = 8) -> str:
    """Useful for general internet search queries."""
    search_results = []
    if not query:
        return json.dumps(search_results)

    results = ddg(query, max_results=num_results)
    if not results:
        return json.dumps(search_results)

    for j in results:
        search_results.append(j)

    return json.dumps(search_results, ensure_ascii=False, indent=4)


# **Browse a web page with PlayWright**

# In[7]:


# !pip install playwright
# !playwright install


# In[8]:


async def async_load_playwright(url: str) -> str:
    """Load the specified URLs using Playwright and parse using BeautifulSoup."""
    from bs4 import BeautifulSoup
    from playwright.async_api import async_playwright

    results = ""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url)

            page_source = await page.content()
            soup = BeautifulSoup(page_source, "html.parser")

            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            results = "\n".join(chunk for chunk in chunks if chunk)
        except Exception as e:
            results = f"Error: {e}"
        await browser.close()
    return results

def run_async(coro):
    event_loop = asyncio.get_event_loop()
    return event_loop.run_until_complete(coro)

@tool
def browse_web_page(url: str) -> str:
    """Verbose way to scrape a whole webpage. Likely to cause issues parsing."""
    return run_async(async_load_playwright(url))


# **Q&A Over a webpage**
# 
# Help the model ask more directed questions of web pages to avoid cluttering its memory

# In[9]:


from langchain.tools.base import BaseTool
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import WebBaseLoader
from pydantic import Field
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain, BaseCombineDocumentsChain

def _get_text_splitter():
    return RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 500,
        chunk_overlap  = 20,
        length_function = len,
    )


class WebpageQATool(BaseTool):
    name = "query_webpage"
    description = "Browse a webpage and retrieve the information relevant to the question."
    text_splitter: RecursiveCharacterTextSplitter = Field(default_factory=_get_text_splitter)
    qa_chain: BaseCombineDocumentsChain
    
    def _run(self, url: str, question: str) -> str:
        """Useful for browsing websites and scraping the text information."""
        result = browse_web_page.run(url)
        docs = [Document(page_content=result, metadata={"source": url})]
        web_docs = self.text_splitter.split_documents(docs)
        results = []
        # TODO: Handle this with a MapReduceChain
        for i in range(0, len(web_docs), 4):
            input_docs = web_docs[i:i+4]
            window_result = self.qa_chain({"input_documents": input_docs, "question": question}, return_only_outputs=True)
            results.append(f"Response from window {i} - {window_result}")
        results_docs = [Document(page_content="\n".join(results), metadata={"source": url})]
        return self.qa_chain({"input_documents": results_docs, "question": question}, return_only_outputs=True)
    
    async def _arun(self, url: str, question: str) -> str:
        raise NotImplementedError
      


# In[10]:


query_website_tool = WebpageQATool(qa_chain=load_qa_with_sources_chain(llm))


# ### Set up memory
# 
# * The memory here is used for the agents intermediate steps

# In[11]:


# Memory
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools.human.tool import HumanInputRun

embeddings_model = OpenAIEmbeddings()
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


# ### Setup model and AutoGPT
# 
# `Model set-up`

# In[12]:


tools = [
    web_search,
    WriteFileTool(),
    ReadFileTool(),
    process_csv,
    query_website_tool,
    # HumanInputRun(), # Activate if you want the permit asking for help from the human
]


# In[13]:


agent = AutoGPT.from_llm_and_tools(
    ai_name="Tom",
    ai_role="Assistant",
    tools=tools,
    llm=llm,
    memory=vectorstore.as_retriever(search_kwargs={"k": 8}),
    # human_in_the_loop=True, # Set to True if you want to add feedback at each step.
)
# agent.chain.verbose = True


# ### AutoGPT as a research / data munger 
# 
# #### `inflation` and `college tuition`
#  
# Let's use AutoGPT as researcher and data munger / cleaner.
#   
# I spent a lot of time over the years crawling data sources and cleaning data. 
# 
# Let's see if AutoGPT can do all of this for us!
# 
# Here is the prompt comparing `inflation` and `college tuition`.

# In[14]:


agent.run(["What were the winning boston marathon times for the past 5 years? Generate a table of the names, countries of origin, and times."])


# In[ ]:




