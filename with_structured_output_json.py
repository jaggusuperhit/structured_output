from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Initialize OpenRouter Gemini Model
model = ChatOpenAI(
    model="google/gemini-2.0-pro-exp-02-05:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# Define the JSON schema
json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "description": "Write down all the Key themes discussed in the review in a list",
            "items": {
                "type": "string"
            }
        },
        "summary": {
            "type": "string",
            "description": "Write a brief summary of the review"
        },
        "sentiment": {
            "type": "string",
            "enum": ["pos", "neg"],
            "description": "Sentiment of the review (pos or neg)"
        },
        "pros": {
            "type": "array",
            "description": "List of pros mentioned in the review",
            "items": {
                "type": "string"
            },
            "nullable": True
        },
        "cons": {
            "type": "array",
            "description": "List of cons mentioned in the review",
            "items": {
                "type": "string"
            },
            "nullable": True
        },
        "name": {
            "type": "string",
            "description": "The name of the reviewer",
            "nullable": True
        }
    },
    "required": ["key_themes", "summary", "sentiment"]
}

# Convert JSON schema to a string for the prompt
structured_model = model.with_structured_output(json_schema)

# Define the input review text
result = structured_model.invoke( """
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
print(result)