# Wikipedia Project

Welcome to the Wikipedia Project! This project is designed to interact with Wikipedia's API to fetch and display information.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/wikipedia-project.git
    ```
2. **Navigate to the project directory:**
    ```bash
    cd wikipedia-project
    ```
3. **Create a virtual environment:**
    - Using `venv`:
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows use `venv\Scripts\activate`
        ```
4. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use the project, run the following command:

```bash
streamlit run app.py
```


## Features

- **Search Wikipedia Articles:** Enter a keyword or phrase to search for related Wikipedia articles.
- **Display Summaries:** View the introduction sections of the top search results.
- **View Article Images:** Display the main image (thumbnail) associated with each article, if available.
- **Article Categorization:** Uses zero-shot classification to categorize articles into predefined categories with associated probabilities.
- **Interactive Web Interface:** User-friendly interface built with Streamlit for easy interaction.
