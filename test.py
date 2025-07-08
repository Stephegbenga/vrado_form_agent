# Google Form Filler with browser-use and OpenAI
# This script uses the browser-use library to automate filling a Google Form.
# It leverages an AI agent to understand a high-level task and execute it.

# -----------------
# --- LIBRARIES ---
# -----------------

# You'll need to install a few libraries for this to work.
# It's recommended to use a virtual environment.
#
# pip install browser-use langchain-openai python-dotenv
#
# After installing, you also need to install the browser playwright needs:
# playwright install chromium

import os
import asyncio
from dotenv import load_dotenv
from browser_use import Agent, Browser, BrowserConfig # Import Browser and BrowserConfig
from browser_use.llm import ChatOpenAI

# -----------------
# --- SETUP ---
# -----------------

# Load environment variables from a .env file
load_dotenv()

# Check if the OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Please create a .env file and add your key, or set the environment variable directly.")
    exit()

# -------------------
# --- MAIN SCRIPT ---
# -------------------

async def fill_form_with_agent(form_url, form_data):
    """
    Uses a browser-use agent to fill out a Google Form with specific data.

    Args:
        form_url (str): The URL of the Google Form.
        form_data (dict): A dictionary where keys are the visible question labels
                          and values are the exact answers to be entered.
    """
    print("Constructing the task for the AI agent...")

    # Construct a detailed, single-string task for the agent.
    # This task tells the agent where to go and what exact text to enter for each question.
    task_description = f"First, go to the following URL: {form_url}. "
    task_description += "Then, carefully fill out the form with the exact information provided: "

    for question, answer in form_data.items():
        task_description += f"For the question labeled '{question}', enter the exact text: '{answer}'. "

    task_description += "After filling out all the fields accurately, find and click the 'Submit' button to submit the form."

    print("\n--- Agent Task ---")
    print(task_description)
    print("--------------------\n")

    try:
        # Initialize the LLM that will power the agent.
        # A powerful model like "gpt-4o" is recommended for understanding complex tasks.
        llm = ChatOpenAI(model="gpt-4o")

        # --- FIX FOR SANDBOX ISSUE (Alternative Method) ---
        # Configure the browser to run without a sandbox using the specific 'chromium_sandbox' flag.
        browser = Browser(
            config=BrowserConfig(
                chromium_sandbox=False
            )
        )

        # Create the agent with the specified task, LLM, and custom browser configuration
        agent = Agent(task=task_description, llm=llm, browser=browser)

        # Run the agent and wait for it to complete the task
        print("Agent is starting... It will now open a browser and perform the task.")
        result = await agent.run()

        print("\n--- Agent Result ---")
        print(result)
        print("--------------------\n")
        print("Task completed successfully!")

    except Exception as e:
        print(f"An error occurred while running the agent: {e}")

# -----------------
# --- EXECUTION ---
# -----------------

if __name__ == "__main__":
    # --- USER INPUT ---
    # Replace with your Google Form URL
    google_form_url = "https://docs.google.com/forms/d/e/1FAIpQLSeIMJVgS4za5CPuW8JlJMreenM6XMez3dkIsFuVWCkjKLn4ow/viewform"

    # Define the data to be filled into the form.
    # The keys should be the exact visible text label of the question on the form.
    # The values are the exact answers to be entered.
    form_data_to_fill = {
        "Role": "business owner",
        "First Name": "steve",
        "Middle Name": "Ola",
        "Surname": "James",
        "Date of Birth": "07/07/2002", # Assuming MM/DD/YYYY or similar format
        "Phone Number": "7867898789",
        "Email Address": "test@gg.com",
        "Gender": "Male",
        "Residential Address": "A31, orisun",
        "City": "ilesa",
        "Local Government": "ilesa west",
        "State": "osun",
        "Identification Number": "1234",
        "Number of Shares": "33"
    }


    if google_form_url == "YOUR_GOOGLE_FORM_URL":
        print("Please replace 'YOUR_GOOGLE_FORM_URL' with the actual URL of your Google Form in the script.")
    else:
        # Run the asynchronous function
        asyncio.run(fill_form_with_agent(google_form_url, form_data_to_fill))
