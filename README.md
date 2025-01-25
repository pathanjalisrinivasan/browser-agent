# Browser-Agent: Automating Web Tasks with Browser-Use and Gemini

This project demonstrates how to automate web tasks using **Browser-Use**, an open-source framework for browser automation, and **Gemini**, an AI tool for intelligent data processing. The goal is to scrape model data from Hugging Face’s repository, filter and sort it using Gemini, and save the results to a file.

---

## Features

- **Automated Data Scraping**: Uses Browser-Use’s Playwright integration to scrape model details from Hugging Face.
- **AI-Driven Processing**: Leverages Gemini to filter models by license type (`cc-by-sa-4.0`) and sort them by popularity (based on likes).
- **Custom Controller**: Saves the top 5 models to a file for easy access and reference.

---

## Installation

Follow these steps to set up and run the project locally.

### Prerequisites

- Python 3.8 or higher
- A Gemini API key (you can get one from [Google’s Gemini API](https://ai.google.dev/))

---

### Step 1: Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/pathanjalisrinivasan/browser-agent.git
cd browser-agent
```
### Step 2: Install Dependencies

Install the required Python packages using pip:

```bash
pip install playwright python-dotenv google-generativeai pydantic
```

After installing the dependencies, set up Playwright’s browser binaries:

```bash
python -m playwright install
```

### Step 3: Add Your Gemini API Key

Create a .env file in the root directory of the project and add your Gemini API key:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

Replace your_gemini_api_key_here with your actual Gemini API key.

### Step 4: Run the Script
Once everything is set up, you can run the script:

```bash
python app.py
```

This will:

Scrape model data from Hugging Face.

Filter and sort the models using Gemini.

Save the top 5 models to a file.

### Project Structure

Here’s an overview of the project structure:

Copy
browser-agent/
├── app.py                # Main script for scraping and processing data
├── .env                  # Environment variables (e.g., Gemini API key)
├── requirements.txt      # List of dependencies
├── README.md             # This file
└── output/               # Directory where results are saved

### Contributing

If you’d like to contribute to this project, feel free to open an issue or submit a pull request. Your feedback and contributions are welcome!

### License

This project is licensed under the MIT License. See the LICENSE file for details.

