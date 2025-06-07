# Excel Insights Chatbot

This project is a Streamlit-based chatbot designed to provide insights from Excel data. It leverages various Python libraries for data manipulation, analysis, and visualization, enhanced with AI capabilities.

## Features

*   **Smart Data Processing**: Automatic column normalization and type detection for clean data.
*   **AI-Powered Analysis**: Get deep insights and actionable recommendations using advanced language models (Groq API).
*   **Interactive Visualizations**: Generate various chart types (Bar, Histogram, Scatter, Line, Box, Pie, Heatmap) with Plotly for interactive data exploration.
*   **Data Quality Reports**: Identify and highlight common data issues like missing values and duplicate rows.
*   **Natural Language Queries**: Ask questions about your data in plain English and receive intelligent responses.
*   **Comprehensive Data Overview**: Quick statistics, column details, and data previews.

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/hs-chaitra/excel_insights_chatbot.git
    cd excel_insights_chatbot
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    source venv/bin/activate  # On macOS/Linux
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment (Optional - for AI features):**
    Create a `.env` file in your project directory. You can use the provided `.env.example` as a template.

    ```
    GROQ_API_KEY=your_groq_api_key_here
    ```

    *Get a Groq API key:*
    - Visit [console.groq.com](https://console.groq.com)
    - Sign up for a free account
    - Generate an API key

    *Note: The application works perfectly without the API key; you'll still get all visualizations and basic insights!*

## Usage

To run the Streamlit application, execute the following command in your terminal:

```bash
streamlit run app.py
```

This will open the application in your web browser.

Once the application is running, you can upload your Excel file using the "Choose your Excel file" button in the sidebar.

## Project Structure

*   `app.py`: The main Streamlit application script containing the chatbot logic, data processing, visualization functions, and AI integration.
*   `requirements.txt`: Lists all the Python dependencies required to run the project.
*   `.env.example`: An example file for setting up environment variables, specifically for the Groq API key.
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore.

## Dependencies

The project relies on the following Python libraries:

*   `streamlit`: For building interactive web applications.
*   `pandas`: For data manipulation and analysis.
*   `numpy`: For numerical operations.
*   `plotly`: For creating interactive visualizations.
*   `openpyxl`: For reading and writing Excel files.
*   `groq`: For integrating with the Groq AI API.
*   `python-dotenv`: For managing environment variables.
*   `matplotlib`: For basic plotting.
*   `seaborn`: For statistical data visualization.
*   `requests`: For making HTTP requests to the Groq API.

## Deployment on Streamlit Cloud

If you plan to deploy this application on Streamlit Cloud, you'll need to configure your `GROQ_API_KEY` as a secret.

1.  Go to your Streamlit Cloud application dashboard.
2.  Click on the three dots (`...`) next to your app and select "Settings".
3.  Navigate to the "Secrets" section.
4.  Add your `GROQ_API_KEY` in TOML format:

    ```toml
    GROQ_API_KEY="your_groq_api_key_here"
    ```

    Replace `"your_groq_api_key_here"` with your actual Groq API key. Changes typically propagate within a minute.

## Contributing

Contributions are welcome! Please feel free to submit issues, pull requests, or suggest improvements.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details. (Note: A `LICENSE.md` file is not included in this repository yet, but this is a common practice.)
