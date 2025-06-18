from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv,find_dotenv
from typing import TypedDict, Annotated, Optional,Literal
from pydantic import BaseModel,Field

load_dotenv(find_dotenv())

model = ChatGoogleGenerativeAI(
    model = "models/gemini-1.5-flash-latest",
    temperature = 1.5
)

#Schema
class Review(BaseModel):

    summary : str = Field(discription="A breif summary of the review") 
    sentiment : Literal["pso","neg"] = Field(discription = "Return Sentiment of the review either negative, Positive or Nutral")
    key_themes :list[str] = Field(description = "Write down all the key themes discussed in the review")
    pros : Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    con : Optional[list[str]] = Field(default=None,description="Write down all the cons inside a list")


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