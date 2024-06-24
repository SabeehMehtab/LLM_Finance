# LLM for financial forecast
This project is about using LLM's for analysing financial reports of companies listed in [Pakistan Stock Exchange (PSX)](https://financials.psx.com.pk/).
They can be useful to draw insights from the reports and summarize them, or provide a judgement on the companies finances. The LLM models solely for this purpose are, yet, limited, but are in the process of being developed. 
Hence, this project utilizes currently developed LLMs (see list below), with the following two approaches, implemented to carry out the same task of fetching and analysing reports from PSX. 

Streamlit was used to host the application and can be accessed here: [https://sms-llmfinance.streamlit.app/](https://sms-llmfinance.streamlit.app/)

## RAG
Retrieval-Augmented Generation (RAG) enhances the capabilities of Large Language Models (LLMs) by integrating references to external authoritative knowledge bases, without the need for retraining.
In our case, the financial reports serve as the knowledge base. However, this external data is stored in a 'Vector Database', a knowledge library that the LLM's can understand. Before storing the data, it is converted into numerical representation, vector, by an 'Embedding Model' and broken down into multiple documents. Once stored, a search query can be performed to fetch relevant document(s) and these are provided as context, along with the user query, for the LLM to generate an accurate response. [Read more](https://aws.amazon.com/what-is/retrieval-augmented-generation/)

However, web scraping, based on user input, is carried out first to partially automate the process of fetching and storing the financial report in the vector database. 
This greatly increases the pace and scope of testing the LLM as otherwise, a report would had to be pre-stored manually before any query could be run and also, restrict the user query to just that report.
[Selenium](https://www.selenium.dev/) is used for web scraping and is running a browser in the background. 

## ReAct Agent
A react agent is a type of LLM agent that combines **re**asoning and **act**ing to make decisions for the LLM, inspired by human's cognition. It enhances LLM capabilities with the use of external tools for more reliable and factual responses. For more info about react agents, [click here](https://www.promptingguide.ai/techniques/react). 

The way that the react agent is applied over here is that custom tools are defined for carrying out the same procedure as in the above RAG method, except for rag chain itself. The following 4 tools were created:
1) set_webpage_year: responsible for setting the correct year in psx webpage for looking up reports.
2) check_company_exists: ensures that a valid company is being searched or is present in the specified year.
3) fetch_financial_docs: stores the financial report of the company as documents in a vector store if it is found.
4) retriever: As the name says, it retrieves relevant documents from the vector store and provides it to the LLM as context.

All these tools are used by the agent once the user has entered a prompt so that the LLM can generate a response based on the context received from the retriever tool and the user prompt. 

## Streamlit Testing
**Note 1 : The appplication does contain errors and if encountered one, please re-run or refresh the page.**

**Note 2 : Many of the reports are big for embeddings to be created with OpenAI Embedding model if you are using the free access and have limited access/request tokens.**

With rag, simply follow the boxes to select and load a financial report, remember to include the api key before clicking the 'store' button. After this, select and adjust a model and its parameters. Then, carry on with any queries you like. Some query examples are:

- Summarize the report
- What are the total assets
- How good is the cash flow

