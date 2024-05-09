# LLM for financial forecast
This project is about using LLM's for analysing financial reports of companies listed in [Pakistan Stock Exchange (PSX)](https://financials.psx.com.pk/).
They can be useful to draw insights from the reports and summarize them, or provide a judgement on the companies finances. The LLM models solely for this purpose are, yet, limited, but are in the process of being developed. 
Hence, this project utilizes currently developed LLMs (see list below), with the following two approaches, implemented to carry out the same task of fetching and analysing reports from PSX. 

## RAG
Retrieval-Augmented Generation (RAG) enhances the capabilities of Large Language Models (LLMs) by integrating references to external authoritative knowledge bases, without the need for retraining.
In our case, the financial reports serve as the knowledge base. However, this external data is stored in a 'Vector Database', a knowledge library that the LLM's can understand. Before storing the data, it is converted into numerical representation, vector, by an 'Embedding Model' and broken down into multiple documents. Once stored, a search query can be performed to fetch relevant document(s) and these are provided as context, along with the user query, for the LLM to generate an accurate response. [Read more](https://aws.amazon.com/what-is/retrieval-augmented-generation/)

However, web scraping, based on user input, is carried out first to partially automate the process of fetching and storing the financial report in the vector database. 
This greatly increases the pace and scope of testing the LLM as otherwise, a report would had to be pre-stored manually before any query could be run and also, restrict the user query to just that report.
[Selenium](https://www.selenium.dev/) is used for web scraping and is running a browser in the background. 

## ReAct Agent
