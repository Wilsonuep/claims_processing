import os
from dotenv import load_dotenv
from google_adk import Agent

load_dotenv()

ga1 = Agent.create(
    model = "gemini-2.0-flash-exp",
    api_key = os.getenv("gemini_api"),
    name = "ga1"
    description = "Basic agent serving as a baseline for comparison in effectiveness"
    instructions = """
    Input: Claim to be fact checked and 4 options to choose from
    Instructions: fact check the claim and choose the most appropriate option
    Output: 0, 1, 2 or 3
    """
)