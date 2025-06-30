from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import requests
import json
load_dotenv(find_dotenv())

# tool Creation
#------------------First tool(conversion factor)-------------------
@tool
def get_conversion_factor(base_currency:str, target_currency: str ) -> float:
    """
    This function featches the currency conversion factor between base currecny and target currency.
    """
    url = f"https://v6.exchangerate-api.com/v6/e1b060f352265ad55846a5b8/pair/{base_currency}/{target_currency}"

    response = requests.get(url)

    return response.json()

@tool
def convert(base_currency_value: int, convert_rate: Annotated[float, InjectedToolArg]) -> float:
    """
    This function featches the currency conversion factor between a given base currency and a target currency.
    """
    return base_currency_value * convert_rate


get_conversion_factor.invoke({'base_currency':'USD', 'target_currency': 'INR'})
convert.invoke({'base_currency_value': 10 , 'convert_rate': 85.5496})


# LLM Bindeing 
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    temperature=0.5
)

# tools Bindeing 
llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

# tool calling

messages = [HumanMessage('What is the conversion factor betweeen USD and INR and based on that can you convert 10 USD to inr and present of mulitiple tools so use all the tools as per reuiremnts.')]

ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)

for tool_call in ai_message.tool_calls:
    # Execute the 1st tool and get the value of conversion rate
    if tool_call['name'] == 'get_conversion_factor':
        tool_message1 = get_conversion_factor.invoke(tool_call)
        # Featch this conversion rate
        conversion_rate = json.loads(tool_message1.content)['conversion_rate']  
        # append this tool meaagae to messages list.
        messages.append(tool_message1)
    # EXecute the 2nd tool using the conversion rate from tool 1
    if tool_call['name'] == 'convert':
        # Featch the current args
        tool_call['args']['convert_rate'] = conversion_rate
        tool_message2 = convert.invoke(tool_call)
        messages.append(tool_message2)


result = llm.invoke(messages)
print(result.content)