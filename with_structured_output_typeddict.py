from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Optional, Literal
import os

# Load environment variables
load_dotenv()

# Initialize gemini-2.0-pro model via OpenRouter
model = ChatOpenAI(
    model="google/gemini-2.0-pro-exp-02-05:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# Define the schema
class Review(TypedDict):
    key_themes: list[str]  # Key themes discussed in the review
    summary: str  # A brief summary of the review
    sentiment: Literal["pos", "neg"]  # Sentiment of the review (pos or neg)
    pros: Optional[list[str]]  # List of pros mentioned in the review
    cons: Optional[list[str]]  # List of cons mentioned in the review
    name: Optional[str]  # The name of the reviewer

# Use structured output
structured_model = model.with_structured_output(Review)

# Invoke the model with structured output

result = structured_model.invoke(
        """
        Review: I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, its an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast-whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45w fast charging is a lifesaver.
        The S-pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera-the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.
        However, the weight and size make it a bit uncomfortable for one-handed use.Also, Samsung's One UI still comes with bloatware-why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

        Pros:
        Insanely powerful processor (great for gaming and productivity)
        Stunning 200MP camera with incredible zoom capabilities
        Long battery life with fast charging
        S-Pen support is unique and useful

        Reviewer Name: Suraj Jagtap
        """
    )

# Print the result
print(result["name"])
