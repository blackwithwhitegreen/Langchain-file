from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

search_tool = DuckDuckGoSearchRun()

# Tool creation
@tool
def get_weather_data(city: str) -> str:
    """
    This function featches the current weather data for a given city.
    """
    url = f'https://api.weatherstack.com/current?access_key=52c71ef4903e51a99e1d6c2b856bbcdb&query={city}'
    response = requests.get(url)

    return response.json()


llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    temperature=0.5
)

# step 2 pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")

# step 3 Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm = llm,
    tools =[search_tool, get_weather_data],
    prompt= prompt
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True
)

# Step 5: Invoke
response = agent_executor.invoke({"input": "Find the capital of Madhya Pradesh, then find it's current weather condition"})
print(response)
print(response['output'])

