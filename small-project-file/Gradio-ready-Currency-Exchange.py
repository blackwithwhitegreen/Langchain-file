import gradio as gr
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
from typing import Annotated
import requests
import json

# Load .env
load_dotenv(find_dotenv())

# Tool 1: Fetch conversion rate
@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """
    Get the conversion factor (exchange rate) from base_currency to target_currency using the ExchangeRate API.
    """
    url = f"https://v6.exchangerate-api.com/v6/e1b060f352265ad55846a5b8/pair/{base_currency}/{target_currency}"
    response = requests.get(url)
    return response.json()

# Tool 2: Convert using the rate
@tool
def convert(base_currency_value: int, convert_rate: Annotated[float, InjectedToolArg]) -> float:
    """
    Multiply the base_currency_value by the convert_rate to get the final converted amount.
    """
    return base_currency_value * convert_rate


# LLM with tools
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.5)
llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

# List of major currency codes for dropdown
# currency_codes = [
#     "USD", "EUR", "GBP", "INR", "JPY", "CAD", "AUD", "CHF", "CNY", "SEK",
#     "NZD", "SGD", "HKD", "KRW", "ZAR", "THB", "AED", "MYR", "BRL", "RUB",
#     "MXN", "DKK", "PLN", "NOK", "IDR", "SAR", "TRY", "TWD", "PKR", "EGP"
# ]

currency_list = [
    ("USD - United States Dollar"),("EUR - Euro"),("GBP - British Pound"),("INR - Indian Rupee"),
    ("JPY - Japanese Yen"),("CAD - Canadian Dollar"),("AUD - Australian Dollar"),
    ("CHF - Swiss Franc"),("CNY - Chinese Yuan"),("SEK - Swedish Krona"),("NZD - New Zealand Dollar"),
    ("SGD - Singapore Dollar"),("HKD - Hong Kong Dollar"),("KRW - South Korean Won"),("ZAR - South African Rand"),
    ("THB - Thai Baht"),("AED - UAE Dirham"),("MYR - Malaysian Ringgit"),("BRL - Brazilian Real"),("RUB - Russian Ruble"),
    ("MXN - Mexican Peso"),("DKK - Danish Krone"),("PLN - Polish Zloty"),("NOK - Norwegian Krone"),
    ("IDR - Indonesian Rupiah"),("SAR - Saudi Riyal"),("TRY - Turkish Lira"),("TWD - Taiwan Dollar"),
    ("PKR - Pakistani Rupee"),("EGP - Egyptian Pound")
]


# Function logic
def currency_conversion(base_currency, target_currency, amount):
    try:
        messages = [HumanMessage(
            f"What is the conversion factor between {base_currency} and {target_currency}? "
            f"Based on that, convert {amount} {base_currency} to {target_currency}."
        )]

        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)

        for tool_call in ai_message.tool_calls:
            if tool_call['name'] == 'get_conversion_factor':
                tool_message1 = get_conversion_factor.invoke(tool_call)
                conversion_rate = json.loads(tool_message1.content)['conversion_rate']
                messages.append(tool_message1)

            if tool_call['name'] == 'convert':
                tool_call['args']['convert_rate'] = conversion_rate
                tool_message2 = convert.invoke(tool_call)
                messages.append(tool_message2)

        final_result = llm.invoke(messages)
        return final_result.content

    except Exception as e:
        return f" Error: {str(e)}"

# Gradio Interface
gr.Interface(
    fn=currency_conversion,
    inputs=[
        gr.Dropdown(choices=currency_list, label="Base Currency", value="USD"),
        gr.Dropdown(choices=currency_list, label="Target Currency", value="INR"),
        gr.Number(label="Amount", value=10)
    ],
    outputs=gr.Textbox(label="Converted Result"),
    title="Currency Converter",
    description="Uses Gemini + LangChain Tools to fetch live exchange rates and convert currency."
).launch()
