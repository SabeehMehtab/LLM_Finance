import streamlit as st
import requests

from openai import RateLimitError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
from langchain_core.tools import ToolException
from langchain_core.documents import Document
from typing import List

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException

class AgentTools():
    
    comp_button = None
    vs = None
    
    class YearInput(BaseModel):
        year: int = Field(description="Any year staring from 2020 and onwards")
        
    class CompanyInput(BaseModel):
        company: str = Field(description="Company Name")
        
    class RetrieverInput(BaseModel):
        query: str = Field(description="User's query")
        
    @tool(args_schema=YearInput)
    def set_webpage_year(year: int):
        """Use this tool first to set the year to search for on webpage"""
        header = st.session_state.agent_driver.find_element(By.ID, 'annRpt').text
        table_rows = st.session_state.agent_driver.find_element(By.TAG_NAME, 'tr')
        if str(year) not in header:
            year = st.session_state.agent_driver.find_element(By.ID,  f"{year}")
            year.click()
            WebDriverWait(st.session_state.agent_driver, 15).until(
                lambda d: len(st.session_state.agent_driver.find_elements(By.TAG_NAME, 'tr')) != table_rows)
            header = st.session_state.agent_driver.find_element(By.ID, 'annRpt').text
            table_rows = st.session_state.agent_driver.find_elements(By.TAG_NAME, 'tr')
    
    @tool(args_schema=CompanyInput)
    def check_company_exists(company: str):
        """Checks whether company financial reports exists for the year specified. An exception would be raised if company is not found or more than one company is found."""
        buttons = st.session_state.agent_driver.find_elements(By.CSS_SELECTOR, f"button[onclick*='{company}']")
        if len(buttons) == 1:
            AgentTools.comp_button = buttons[0]
        elif len(buttons) > 1:
            raise ToolException(f"{len(buttons)} companies found with the name '{company}'.")
        else:
            raise ToolException("No such company found in the given year.")
        
    @tool(args_schema=CompanyInput)
    def fetch_financial_docs(company_name: str) -> str:
        """Fetch annual financial report for the company if the company exists. A statement is returned confirming whether the report was found or not."""
        AgentTools.comp_button.click()
        st.session_state.agent_driver.implicitly_wait(5)
        try:
            doc_link = st.session_state.agent_driver.find_element(By.LINK_TEXT, "Annual").get_attribute('href')
            response = requests.get(doc_link)
            filename = company_name + ".pdf"
            pdf = open(filename, 'wb')
            pdf.write(response.content)
            pdf.close()

            loader = PyPDFLoader(filename)
            docs = loader.load()
            documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
            AgentTools.vs.add_documents(documents)
            statement = "Annual report found and stored"
        except NoSuchElementException as e:
            statement = "Error0: No annual report found."
            st.toast(statement)
        except IndexError as ie:
            statement = "Error1: Could not store report. Please select another report!"
            st.toast(statement)
        except ValueError as ve:
            statement = "Error2: Could not store report. Please select another report!"
            st.toast(statement)
        except RateLimitError as re:
            statement = "Error3: RateLimitError. Please increase limit or select another report!"
            st.toast(statement)
            
        finally:
            xpath = '//*[@id="myModal"]/div/div/div[3]/button'
            close_btn = st.session_state.agent_driver.find_element(By.XPATH, xpath)
            close_btn.click()
            if statement != "Annual report found and stored":
                st.stop()
                
        return statement
    
    @tool(args_schema=RetrieverInput)
    def retriever(query : str )-> List[Document]:
        """Useful for retrieving relevant documents from annual report if it was found. Returns a list of relevant documents to be used for answering user's query"""
        retriever = AgentTools.vs.as_retriever(search_kwargs={"k": 4})
        query_docs = retriever.get_relevant_documents(query)
        return query_docs
    
    def return_tools() -> list:
        return [AgentTools.set_webpage_year, AgentTools.check_company_exists,
                AgentTools.fetch_financial_docs, AgentTools.retriever]
        