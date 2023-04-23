#!/usr/bin/env python
# coding: utf-8

# # AutoGPT
# 
# Implementation of https://github.com/Significant-Gravitas/Auto-GPT but with LangChain primitives (LLMs, PromptTemplates, VectorStores, Embeddings, Tools)

# ## Set up tools
# 
# We'll set up an AutoGPT with a search tool, and write-file tool, and a read-file tool

# In[1]:


from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool

search = SerpAPIWrapper()
tools = [
    Tool(
        name = "search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions"
    ),
    WriteFileTool(),
    ReadFileTool(),
]


# ## Set up memory
# 
# The memory here is used for the agents intermediate steps

# In[ ]:


from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings


# In[ ]:


# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
import faiss

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


# ## Setup model and AutoGPT
# 
# Initialize everything! We will use ChatOpenAI model

# In[ ]:


from langchain.experimental import AutoGPT
from langchain.chat_models import ChatOpenAI


# In[ ]:


agent = AutoGPT.from_llm_and_tools(
    ai_name="Tom",
    ai_role="Assistant",
    tools=tools,
    llm=ChatOpenAI(temperature=0),
    memory=vectorstore.as_retriever()
)
# Set verbose to be true
agent.chain.verbose = True


# ## Run an example
# 
# Here we will make it write a weather report for SF

# In[ ]:


agent.run(["what stock was the biggest gainer today"])

