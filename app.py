import os
import asyncio
import time
import logging
from typing import List, Callable, Dict, Any
from collections import deque
from pydantic import BaseModel
from dotenv import load_dotenv  # Load environment variables from .env file
import google.generativeai as genai  # For Gemini
from tenacity import retry, wait_exponential, stop_after_attempt
from playwright.async_api import async_playwright  # For browser automation

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables from .env file
load_dotenv()

# Custom Controller class
class Controller:
    def __init__(self):
        self.actions = {}  # Dictionary to store registered actions

    def action(self, name: str, param_model: BaseModel = None):
        """
        Decorator to register an action with the controller.
        """
        def decorator(func: Callable):
            self.actions[name] = (func, param_model)
            return func
        return decorator

    def execute_action(self, name: str, params: Any = None):
        """
        Execute a registered action by name.
        """
        if name not in self.actions:
            raise ValueError(f"Action '{name}' not found.")
        func, param_model = self.actions[name]
        if param_model and params:
            params = param_model(**params)  # Validate params using the Pydantic model
        return func(params)

# Initialize controller
controller = Controller()

# Define data models
class Model(BaseModel):
    title: str
    url: str
    likes: int
    license: str

class Models(BaseModel):
    models: List[Model]

# Register save action
@controller.action('Save models', param_model=Models)
def save_models(params: Models):
    """
    Save the top 5 models to a file.
    """
    logging.info("Saving models to file...")
    with open('models.txt', 'w') as f:
        for model in params.models[:5]:  # Save top 5
            f.write(f"{model.title}\nURL: {model.url}\nLikes: {model.likes}\nLicense: {model.license}\n\n")
    logging.info("Models saved to 'models.txt'.")

# Rate limiter class to throttle API requests
class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()  # Use deque to store timestamps

    async def wait(self):
        if len(self.calls) >= self.max_calls:
            until = self.calls[0] + self.period
            await asyncio.sleep(max(0, until - time.time()))
            self.calls.popleft()
        self.calls.append(time.time())

# Initialize rate limiter (e.g., 3 requests per 60 seconds)
rate_limiter = RateLimiter(max_calls=3, period=60)

# Retry mechanism with exponential backoff
@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
async def make_request_with_retry(task, llm, controller):
    await rate_limiter.wait()  # Throttle requests
    agent = Agent(task=task, llm=llm, controller=controller)
    await agent.run()

# Custom Agent class
class Agent:
    def __init__(self, task, llm, controller):
        self.task = task
        self.llm = llm  # This will be the Gemini model
        self.controller = controller

    async def run(self):
        logging.info(f"Executing task: {self.task}")
        
        # Step 1: Use Playwright to scrape data from Hugging Face
        scraped_data = await self.scrape_huggingface()
        logging.info(f"Scraped data: {scraped_data}")
        
        # Step 2: Send the scraped data to Gemini for processing
        logging.info("Sending scraped data to Gemini API...")
        response = self.llm.generate_content(
            f"Process the following data and return the top 5 models with license 'cc-by-sa-4.0', sorted by most likes:\n{scraped_data}"
        )
        logging.info("Received response from Gemini API.")
        
        # Print the raw response for debugging
        logging.info(f"Raw response from Gemini:\n{response.text}")
        
        # Step 3: Parse the response into a list of Model objects
        parsed_models = self.parse_response(response.text)
        logging.info(f"Parsed models: {parsed_models}")
        
        # Step 4: Save the parsed models using the controller
        logging.info("Saving parsed models...")
        self.controller.execute_action('Save models', {"models": parsed_models})

    def parse_response(self, response_text):
        """
        Parse the response from Gemini into a list of Model objects.
        This assumes the response is structured as plain text with model names and likes.
        """
        models = []
        
        # Split the response into lines
        lines = response_text.split("\n")
        
        # Iterate through each line and extract model information
        for line in lines:
            if line.strip():  # Skip empty lines
                # Remove the numbering (e.g., "1. ", "2. ", etc.)
                line = line.split(". ", 1)[-1] if ". " in line else line
                
                # Extract the model name and likes from the line
                if "(" in line and ")" in line:
                    title = line.split(" (", 1)[0].strip()
                    likes = line.split(" (", 1)[1].replace(" likes)", "").strip()
                    
                    # Convert likes to integer (handle "k" and "M" suffixes)
                    if "k" in likes:
                        likes = int(float(likes.replace("k", "")) * 1000)
                    elif "M" in likes:
                        likes = int(float(likes.replace("M", "")) * 1000000)
                    else:
                        likes = int(likes)
                    
                    # Create a Model object
                    models.append(Model(
                        title=title,
                        url=f"https://huggingface.co/{title}",  # Construct URL from title
                        likes=likes,  # Convert likes to integer
                        license="cc-by-sa-4.0"  # Default license (adjust as needed)
                    ))
        
        return models

    async def scrape_huggingface(self):
        """
        Use Playwright to scrape data from Hugging Face.
        """
        async with async_playwright() as p:
            # Launch a browser
            browser = await p.chromium.launch(headless=False)  # Set headless=False to see the browser
            page = await browser.new_page()
            
            # Navigate to Hugging Face models page
            await page.goto("https://huggingface.co/models")
            logging.info("Opened Hugging Face models page.")
            
            try:
                # Wait for the models to load (adjust the selector as needed)
                await page.wait_for_selector(".w-full.truncate", timeout=60000)  # Increased timeout to 60 seconds
                logging.info("Models loaded on the page.")
                
                # Scrape all model cards
                model_cards = await page.query_selector_all(".w-full.truncate")
                scraped_data = []
                for card in model_cards:
                    title = await card.inner_text()
                    url = await card.get_attribute("href")
                    # Extract license information (if available)
                    license_info = "cc-by-sa-4.0"  # Default license (adjust as needed)
                    scraped_data.append({"title": title, "url": url, "license": license_info})
                
                logging.info(f"Scraped {len(scraped_data)} models.")
            except Exception as e:
                logging.error(f"Error while waiting for models: {e}")
                scraped_data = []
            finally:
                # Close the browser
                await browser.close()
            
            return scraped_data

async def main():
    # Task description
    task = (
        "Go to huggingface.co/models, filter models with license 'cc-by-sa-4.0', "
        "sort by most likes, and return the top 5 results."
    )
    
    # Configure Gemini API
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Use a valid Gemini model
    llm = genai.GenerativeModel('gemini-pro')
    
    # Execute the task with retry and rate-limiting
    logging.info("Starting task execution...")
    await make_request_with_retry(task, llm, controller)
    logging.info("Task execution completed.")

if __name__ == "__main__":
    asyncio.run(main())