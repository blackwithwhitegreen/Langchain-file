from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv,find_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv(find_dotenv())

model = ChatGoogleGenerativeAI(
    model = "models/gemini-1.5-flash-latest",
    temperature = 1.5
)

#Schema
class Review(TypedDict):

    # for Guiding to the llm we are adding one line discription in each, for better understand for the llm, that's why we are using Annotated here.
    summary : Annotated[str, "A breif summary of the review"]
    sentiment : Annotated[str,"Return Sentiment of the review either negative, Positive or Nutral"]
    key_themes : Annotated[list[str], "Write down all the key themes discussed in the review"]
    pro : Annotated[Optional[list[str]],"Write down all the pros inside a list"]# TypedDict is not allow the Optional key, instead, we 
    econ : Annotated[Optional[list[str]],"Write down all the cons inside a list"]# can use the Pydantic where we can use Optinal key.


structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
        """After using the PulseX Smartwatch Series 5 for the past month, I can confidently say it strikes a solid balance between functionality and affordability. Right out of the box, the watch feels premium — the metal frame is sturdy yet lightweight, and the silicone strap doesn’t irritate even during long hours of wear. The 1.6-inch AMOLED screen is one of its standout features. It’s crisp, colorful, and easily visible even in direct sunlight. Whether I’m checking messages, viewing fitness stats, or just using it as a regular watch, the display quality makes a huge difference.

        The step counter, workout modes, and hydration reminders help me stay consistent with my fitness goals. I’ve tested several workout options, and the watch adjusts the metrics accordingly — which is something many budget watches don’t do well. The battery easily lasts me 6–7 days per charge, even with GPS and regular usage, which is a big plus.
        
        Notifications from WhatsApp, calls, texts, and even emails come through in real time, and the vibration alert is strong enough to catch my attention without being annoying. The Bluetooth calling feature works great indoors and in quiet spaces, but it does struggle in noisy environments. The UI on the watch is responsive, and swiping between widgets feels snappy.
        Pros:
        The AMOLED display is bright and sharp. Battery life is excellent. Fitness tracking is accurate and detailed.
        Cons:
        No third-party app support. Voice assistant feels basic and sometimes doesn’t respond. Watch face customization is limited"""
)

print(result)
# print(result['summary'])
# print(result['sentiment'])



# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.utils.function_calling import convert_to_openai_function
# from typing import TypedDict, Annotated, Optional
# from dotenv import load_dotenv, find_dotenv

# load_dotenv(find_dotenv())

# # Step 1: Create your model
# model = ChatGoogleGenerativeAI(
#     model="models/gemini-1.5-flash-latest",
#     temperature=0.7
# )

# # Step 2: Define TypedDict schema
# class Review(TypedDict):
#     key_themes: Annotated[list[str], "Write down all the key themes discussed in the review"]
#     summary: Annotated[str, "A brief summary of the review"]
#     sentiment: Annotated[str, "Return Sentiment of the review either negative, positive or neutral"]
#     pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
#     cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]

# # Step 3: Convert schema using `convert_to_openai_function`
# structured_model = model.with_structured_output(Review)

# # Step 4: Use long review text
# review_text = """
# After using the PulseX Smartwatch Series 5 for the past month, I can confidently say it strikes a solid balance between functionality and affordability. Right out of the box, the watch feels premium — the metal frame is sturdy yet lightweight, and the silicone strap doesn’t irritate even during long hours of wear. The 1.6-inch AMOLED screen is one of its standout features. It’s crisp, colorful, and easily visible even in direct sunlight. Whether I’m checking messages, viewing fitness stats, or just using it as a regular watch, the display quality makes a huge difference.

# The step counter, workout modes, and hydration reminders help me stay consistent with my fitness goals. I’ve tested several workout options, and the watch adjusts the metrics accordingly — which is something many budget watches don’t do well. The battery easily lasts me 6–7 days per charge, even with GPS and regular usage, which is a big plus.

# Notifications from WhatsApp, calls, texts, and even emails come through in real time, and the vibration alert is strong enough to catch my attention without being annoying. The Bluetooth calling feature works great indoors and in quiet spaces, but it does struggle in noisy environments. The UI on the watch is responsive, and swiping between widgets feels snappy.

# Pros:
# The AMOLED display is bright and sharp. Battery life is excellent. Fitness tracking is accurate and detailed.

# Cons:
# No third-party app support. Voice assistant feels basic and sometimes doesn’t respond. Watch face customization is limited.
# """

# # Step 5: Invoke the structured output
# result = structured_model.invoke(review_text)

# # Step 6: Output
# print(result)
# if result:
#     print("Summary:", result["summary"])
#     print("Sentiment:", result["sentiment"])
# else:
#     print("No result returned. Make sure your schema is valid.")





# # from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain_core.pydantic_v1 import BaseModel, Field
# # from dotenv import load_dotenv, find_dotenv
# # from typing import Optional
# # import os

# # load_dotenv(find_dotenv())

# # model = ChatGoogleGenerativeAI(
# #     model="models/gemini-1.5-flash-latest",
# #     temperature=1.0
# # )

# # class Review(BaseModel):
# #     key_themes: list[str] = Field(..., description="Write down all the key themes discussed in the review")
# #     summary: str = Field(..., description="A brief summary of the review")
# #     sentiment: str = Field(..., description="Return Sentiment of the review either negative, positive or neutral")
# #     pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
# #     cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")

# # structured_model = model.with_structured_output(Review)

# # review_text = """After using the PulseX Smartwatch Series 5 for the past month... [your full review here]"""

# # result = structured_model.invoke(review_text)

# # if result:
# #     print(result)
# #     print(result.summary)
# #     print(result.sentiment)
# # else:
# #     print("No result returned. Check API key, model ID, or request content.")
