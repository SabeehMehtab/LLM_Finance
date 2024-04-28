import streamlit as st
import requests

from langchain import hub
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from openai import RateLimitError

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS,  Chroma

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException

from tools import AgentTools

YEAR_OPTIONS = [2020,2021,2022,2023,2024]
EMBEDDING_MODELS = ['OpenAIEmbeddings']#, 'BAAI/bge-base-en-v1.5', 'BAAI/bge-m3']
LLM_MODELS = ["GPT 3", "mistralai/Mixtral-8x7B-Instruct-v0.1", "huggingfaceh4/zephyr-7b-beta", "tiiuae/falcon-7b-instruct"]
prompt = """
      <|system|>
      You are a financial expert and you must answer base on financial report provided.
      </s>
      <|user|>
      {query}
      </s>
      <|assistant|>"""

def set_driver():
  # Set Chrome driver options
  chrome_options = webdriver.ChromeOptions()
  chrome_options.add_argument('--headless')
  chrome_options.add_argument('--no-sandbox')
  chrome_options.add_argument('--disable-dev-shm-usage')

  # Initialize Chrome driver and load the website
  st.session_state.driver = webdriver.Chrome(options=chrome_options)
  st.session_state.driver.get('https://financials.psx.com.pk/')
  st.session_state.driver.implicitly_wait(5)

def set_agent_webdriver():
  # Set LLM Agent Chrome driver options
  chrome_options = webdriver.ChromeOptions()
  chrome_options.add_argument('--headless')
  chrome_options.add_argument('--no-sandbox')
  chrome_options.add_argument('--disable-dev-shm-usage')

  # Initialize LLM Agent Chrome driver and load the website
  st.session_state.agent_driver = webdriver.Chrome(options=chrome_options)
  st.session_state.agent_driver.get('https://financials.psx.com.pk/')
  st.session_state.agent_driver.implicitly_wait(5)


def find_companies():
  # Get table header
  header = st.session_state.driver.find_element(By.ID, 'annRpt').text

  # Check to see if user relevant year table is not loaded.
  # If true, then fetch and load relevant year table
  if str(st.session_state.year) not in header:
    table = st.session_state.driver.find_element(By.ID, "yearlyData")
    table_rows = len(table.find_elements(By.TAG_NAME, 'tr'))
    year = st.session_state.driver.find_element(By.ID, f"{st.session_state.year}")
    year.click()
    WebDriverWait(st.session_state.driver, 15).until(lambda d: len(st.session_state.driver.find_elements(By.TAG_NAME, 'tr')) != table_rows)

  # Get the number of rows in the table
  table = st.session_state.driver.find_element(By.ID, "yearlyData")
  table_rows = len(table.find_elements(By.TAG_NAME, 'tr'))-1

  # Variable to store all the company names
  st.session_state.company_list = []

  # Loop through each row in table and get the company name and store it.
  # Ignore company with no proper name given
  x = 1
  while True:
    company_name = st.session_state.driver.find_element(By.XPATH, f'//*[@id="yearlyData"]/table/tbody/tr[{x}]/td[3]').text
    if company_name != '-':
      st.session_state.company_list.append(company_name)
    if x == table_rows:
      break
    x+=1

def find_reports():
  # Variable to store report type and its online document link
  st.session_state.reports = dict()

  # Search and click download file button for company
  button = st.session_state.driver.find_element(By.CSS_SELECTOR, f"button[onclick*='{st.session_state.company}']")
  button.click()

  # Quarter report(s) fetch
  try:
    q_reports = st.session_state.driver.find_elements(By.LINK_TEXT, "Quarterly")
  except NoSuchElementException as e:
    pass
  else:
    for index, r in enumerate(q_reports):
      st.session_state.reports[f'Quarter{index+1}'] = r.get_attribute("href")

  # Annual report fetch
  try:
    a_report = st.session_state.driver.find_element(By.LINK_TEXT, "Annual")
  except NoSuchElementException as e:
    pass
  else:
    st.session_state.reports['Annual'] = a_report.get_attribute("href")
  finally:
    # Access and click close button (in popup window)
    xpath = '//*[@id="myModal"]/div/div/div[3]/button'
    close_btn = st.session_state.driver.find_element(By.XPATH, xpath)
    close_btn.click()

def fetch_and_store(emb: str, hf_token: str, api_key: str, chunk_size: int, chunk_overlap: int):
  # Get the report document link
  doc_link = st.session_state.reports[st.session_state.user_report]

  # Download the document from the link into the work folder
  response = requests.get(doc_link)
  filename = st.session_state.company+st.session_state.user_report+".pdf"
  pdf = open(filename, 'wb')
  pdf.write(response.content)
  pdf.close()

  # Load and split the document
  loader = PyPDFLoader(filename)
  docs = loader.load()
  documents = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_documents(docs)

  # Choose embedding_model
  if emb is EMBEDDING_MODELS[0]:
    embedding = OpenAIEmbeddings(api_key=api_key)
  else:
    embedding = HuggingFaceInferenceAPIEmbeddings(api_key = hf_token, model_name=emb)

  # Try to store the document(s) in a vectorestore. Catch any error(s) produced and display approriate action message
  # Create a retriever variable to fetch 2 relevant documents
  try:
    vs = FAISS.from_documents(documents = documents, embedding = embedding)
  except RateLimitError as rle:
    st.toast("File exceeds the limit for creating embeddings. Please increase limit or select another report to proceed")
  except IndexError as e:
    st.toast("Cannot store this file. Please select another report.")
  else:
    st.session_state.retriever = vs.as_retriever(search_kwargs={'k': 4})
    st.session_state.REPORT_STORED = True

def delete_report():
  del st.session_state.REPORT_STORED


def setup_chain(llm: str, hf_token: str, api_key: str, temperature: float) -> BaseRetrievalQA:
  # Assign LLM based on user choice and create the retrieval chain. 
  if llm is LLM_MODELS[0]:
    model = OpenAI(temperature=temperature, api_key=api_key)
  else:
    model = HuggingFaceHub(
      repo_id=llm,
      model_kwargs={"temperature": temperature, "max_length": 64,"max_new_tokens":512},
      huggingfacehub_api_token = hf_token)
    
  return RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=st.session_state.retriever)

def llm_inference_rag(tabRAG, llm, hf_token, api_key, temp):
  # Text for user about report being successfully stored.
  try:
    tabRAG.text(
      f"{st.session_state.user_report} report stored successfully of {st.session_state.company} for {st.session_state.year}")
  except AttributeError as e:
    pass
  
  # Subheading 2 and info for user
  tabRAG.subheader('LLM Inference')
  tabRAG.write("Now that the report is stored successfully, LLM can be tested. So, please a LLM and set its temperature from the left sidebar. Check the box to proceed!")
  
  # Give an option to user for changing the report via button. Once clicked, it should clear previously stored report.
  tabRAG.button("Change report", on_click=delete_report)
  
  # Before testing and running the llm, take user's agreement for selecting an LLM model and its temperature
  llm_checkbox = tabRAG.checkbox("I have set the LLM and its temperature")
  
  # After agreement, setup RAG chain and ask user for query. 
  if llm_checkbox:
    ragchain = setup_chain(llm=llm, hf_token=hf_token, api_key=api_key,temperature=temp)
    tabRAG.write("Next, enter a query to run the LLM against and wait for its output below.")
    user_query = tabRAG.text_input("Enter query here: ")
    
    # Use a run button to run the LLM RAG chain and display the output. Make sure user has entered a query
    tabRAG.button('Run', key = 'run_btn')
    if st.session_state.run_btn and user_query == "":
      st.toast("Please enter a query!")
    elif st.session_state.run_btn and user_query != "":
      if llm == LLM_MODELS[0]: 
        tabRAG.write(ragchain.invoke(prompt.format(query = user_query))["result"])
      else:
        tabRAG.write(ragchain.invoke(prompt.format(query = user_query))["result"].split("Helpful Answer:")[-1])

def web_scraping_rag(tabRAG, emb, hf_token, api_key, chunk_size, chunk_overlap):
  # Subheading 1 and info for user
  tabRAG.subheader('Web scraping', divider ='rainbow')
  tabRAG.write("This is the first approach which uses the 'Retrieval-Augmented Generation' (RAG) method. FIrst, to store the financial report in a vector store, web scraping is done to fetch the relevant report from the https://financials.psx.com.pk/. So, follow & fill in the the boxes below to extract any report from the website. ")
  
  # Set Chrome driver if not previously set.
  if 'driver' not in st.session_state:
    set_driver()
    
  # Ask the user to select year, company and report type to fetch and store the financial report from the website.
  col1, col2 = tabRAG.columns([0.4, 0.6])
  col1.selectbox("Select year", YEAR_OPTIONS, index=None, key = 'year', on_change=find_companies)
  if st.session_state.year is not None:
      col2.selectbox("Select company", st.session_state.company_list, index=None, key='company', on_change=find_reports)
  col3, col4 = tabRAG.columns([0.8, 0.2])
  try:
    if st.session_state.company is not None:
      col3.selectbox("Select financial report to examine", st.session_state.reports.keys(),
                      index=None, key='user_report',
                      on_change = lambda: st.toast(
                        "Please make sure api key/token is present with appropriate embedding model to store the report"))
    if st.session_state.user_report is not None:
      col4.text('')
      col4.text('')
      col4.button("Store report", key='store_btn', use_container_width=True,
                  on_click=fetch_and_store, args=(emb,hf_token,api_key, chunk_size, chunk_overlap))
  except AttributeError as e:
    pass
  
def set_llm_agent(api_key, temp) -> AgentExecutor:
  # Assign GPT 3.5 model as LLM for react agent
  llm = OpenAI(temperature=temp, api_key=api_key)
  
  # Assign the react agent tools in tools.py
  tools = AgentTools.return_tools()
  
  # Get the react agent template
  base_prompt = hub.pull("langchain-ai/react-agent-template")
  
  # Give appropriate instructions to the react agent
  prompt = base_prompt.partial(instructions = 
    """
    You are an agent designed to provide annual financial forecast or information about companies in Pakistan.
    You can extract and access company relevant report for any year with the tools provided.
    """)
  
  # Create the react agent and it executor. 
  agent = create_react_agent(llm=llm, tools = tools, prompt=prompt)
  agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
  return agent_executor

def main():
  # Webpage settings
  st.set_page_config(page_title="Fin-est Gen App", layout="wide")
  
  # Webpage title & intro
  st.title("Financial Forecast Using Open Source LLM")
  st.write("This page contains two ways to test out LLM's on producing financial insights through the use of fiancial reports. ")
  
  # Webpage sidebar elements
  hf_token = st.sidebar.text_input("Enter hugging face token", type="password", autocomplete='off')
  openai_apikey = st.sidebar.text_input("Enter OpenAI API Key", type="password", autocomplete='off')
  emb = st.sidebar.selectbox("Embeddings", EMBEDDING_MODELS)
  chunk_size = st.sidebar.slider("Chunk Size", min_value=20, max_value = 5000,
                                 step=10, value=1000)
  chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=10, max_value = 2000,
                                    step=10, value=200)
  llm = st.sidebar.selectbox("LLM",LLM_MODELS)
  temp = st.sidebar.number_input(f"Set the {llm} temperature", min_value = 0.0, max_value = 1.0,
                                 step = 0.1, value=0.4)
  
  # Webpage tabs: RAG and Agent
  tabRAG, tabAgent = st.tabs(["RAG", "Agent"])
  
  # RAG tab execution 
  with tabRAG:
    # Check to see whether report is already stored. If stored, then proceed with LLM inference.
    # If not stored, then proceed with web scraping.
    if 'REPORT_STORED' in st.session_state and st.session_state.REPORT_STORED:
      llm_inference_rag(tabRAG=tabRAG, llm=llm, hf_token=hf_token,
                        api_key=openai_apikey, temp=temp)
    else:
      web_scraping_rag(tabRAG=tabRAG, emb=emb, hf_token=hf_token,
                       api_key=openai_apikey, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  
  # Agent tab execution
  with tabAgent:
    # Set agent Chrome driver if previously not set
    if 'agent_driver' not in st.session_state:
      set_agent_webdriver()
      
    # Subheadings and info for user
    tabAgent.subheader('ReAct Agent with LLM', divider='rainbow')
    tabAgent.write("In the second approach, a ReAct agent is used with LLM to make use of different tools, some of them responsible for web scraping to fetch and store the financial report from https://financials.psx.com.pk/ into a vector store")
    tabAgent.write("**However, the agent can only get annual financial reports in a given year and works with just OpenAI GPT 3 model.**")
    tabAgent.subheader('Test the Agent', divider='green')
    tabAgent.write("Before testing the agent, please set the model to GPT 3  and set its temperature. Make sure the api key is present. Check the box to proceed!")
    
    # Before moving on with the agent, take user's agreement on api key and LLM temperature
    llm_checkbox = tabAgent.checkbox("I have entered the api key and set the llm temperature")
    
    # After agreement, set the agent and ask user for prompt
    if llm_checkbox:
      agent = set_llm_agent(openai_apikey, temp)
      tabAgent.write("Now, enter a prompt to run the LLM against and wait for its output below. ")
      user_prompt = tabAgent.text_input(label='Enter prompt here: ')
      
      # Use a run button to run the agent and display the output. Make sure user has entered a prompt
      tabAgent.button('Run', key='run_agent')
      if st.session_state.run_agent and user_prompt == "":
        st.toast("Please enter a prompt")
      elif st.session_state.run_agent and user_prompt != "":
        AgentTools.vs = Chroma(embedding_function=OpenAIEmbeddings(api_key=openai_apikey))
        tabAgent.write(agent.invoke({'input' : user_prompt}))
    
if __name__=="__main__":
  main()