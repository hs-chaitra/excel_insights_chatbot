import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import os
from dotenv import load_dotenv
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env
load_dotenv()

# --- Configuration ---
st.set_page_config(
    page_title="Excel Insight Chatbot", 
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
def normalize_column_names(df):
    """Normalize column names for consistency"""
    df.columns = [col.strip().lower().replace(' ', '').replace("-", "").replace("(", "").replace(")", "").replace("/", "_") for col in df.columns]
    return df

def infer_column_types(df):
    """Get detailed information about column types"""
    type_info = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_count = df[col].nunique()
        null_count = df[col].isnull().sum()
        type_info[col] = {
            'dtype': dtype,
            'unique_values': unique_count,
            'null_values': null_count,
            'null_percentage': round((null_count / len(df)) * 100, 2)
        }
    return type_info

def get_data_summary(df):
    """Generate comprehensive data summary"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    summary = {
        "shape": f"{df.shape[0]} rows × {df.shape[1]} columns",
        "columns": list(df.columns),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "missing_values": df.isnull().sum().to_dict(),
        "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    }
    
    if len(numeric_cols) > 0:
        summary["numeric_summary"] = df[numeric_cols].describe().to_dict()
    
    return summary

def ask_groq(query, df_info, df_sample=None):
    """Enhanced function to query Groq API with better context"""
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    if not GROQ_API_KEY:
        return """
        ⚠ *API Configuration Required*
        
        To enable AI analysis, please set up your Groq API key:
        
        1. Get a free API key from [console.groq.com](https://console.groq.com)
        2. Create a .env file in your project directory
        3. Add: GROQ_API_KEY=your_api_key_here
        4. Restart the application
        
        *Meanwhile, you can still use all the visualization features!*
        """
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Enhanced context with sample data
    context = f"""
    Dataset Information:
    {df_info}
    
    Sample Data (first 3 rows):
    {df_sample if df_sample else "No sample data available"}
    
    Please provide detailed insights and analysis based on this data structure.
    """
    
    payload = {
        "model": "gemma2-9b-it",
        "messages": [
            {
                "role": "system", 
                "content": """You are an expert data analyst specializing in Excel data analysis. 
                Provide clear, actionable insights with specific recommendations. 
                When possible, suggest visualizations or further analysis steps.
                Be concise but thorough in your explanations."""
            },
            {
                "role": "user", 
                "content": f"{context}\n\nUser Question: {query}"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        elif response.status_code == 401:
            return """
            🔑 *Invalid API Key*
            
            Your Groq API key appears to be invalid. Please:
            
            1. Check your .env file for typos
            2. Verify your API key at [console.groq.com](https://console.groq.com)
            3. Make sure the key has proper permissions
            4. Restart the application after updating
            
            *You can still use visualizations without the AI features!*
            """
        elif response.status_code == 429:
            return """
            ⏱ *Rate Limit Exceeded*
            
            You've hit the API rate limit. Please wait a moment and try again.
            
            *Tip:* Consider upgrading your Groq plan for higher limits.
            """
        else:
            return f"""
            ❌ *API Error ({response.status_code})*
            
            There was an issue with the API request. This could be temporary.
            
            *You can still use all visualization features!*
            
            Error details: {response.text[:200]}...
            """
    except requests.exceptions.Timeout:
        return """
        ⏱ *Request Timeout*
        
        The API request took too long. Please try again with a simpler question.
        """
    except Exception as e:
        return f"""
        ❌ *Connection Error*
        
        Unable to connect to the AI service: {str(e)[:100]}...
        
        *You can still use all visualization features!*
        """

def get_basic_insights(df, query):
    """Provide basic insights without API when Groq is unavailable"""
    insights = []
    query_lower = query.lower()
    
    # Basic dataset info
    insights.append(f"📊 *Dataset Overview:*")
    insights.append(f"- {df.shape[0]:,} rows and {df.shape[1]} columns")
    insights.append(f"- {df.isnull().sum().sum():,} missing values total")
    
    # Column analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if numeric_cols:
        insights.append(f"\n📈 *Numeric Columns ({len(numeric_cols)}):* {', '.join(numeric_cols)}")
        for col in numeric_cols[:3]:  # Top 3 numeric columns
            insights.append(f"- {col}: Mean = {df[col].mean():.2f}, Std = {df[col].std():.2f}")
    
    if categorical_cols:
        insights.append(f"\n📝 *Categorical Columns ({len(categorical_cols)}):* {', '.join(categorical_cols)}")
        for col in categorical_cols[:3]:  # Top 3 categorical columns
            top_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'
            insights.append(f"- {col}: {df[col].nunique()} unique values, most common: '{top_value}'")
    
    # Query-specific insights
    if any(word in query_lower for word in ['trend', 'pattern', 'correlation']):
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            # Find highest correlation
            corr_values = corr.abs().unstack().sort_values(ascending=False)
            corr_values = corr_values[corr_values < 1.0]  # Remove self-correlations
            if not corr_values.empty:
                col1, col2 = corr_values.index[0]
                corr_val = corr_values.iloc[0]
                insights.append(f"\n🔗 *Strongest Correlation:* {col1} & {col2} (r = {corr_val:.3f})")
    
    if any(word in query_lower for word in ['outlier', 'anomaly']):
        for col in numeric_cols[:2]:  # Check first 2 numeric columns
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            if len(outliers) > 0:
                insights.append(f"\n⚠ *Outliers in {col}:* {len(outliers)} potential outliers detected")
    
    insights.append(f"\n💡 *Suggestions:*")
    insights.append("- Try asking for specific visualizations (bar chart, histogram, scatter plot)")
    insights.append("- Look for patterns in the Data Quality tab")
    insights.append("- Use the Quick Stats tab for detailed statistics")
    
    return "\n".join(insights)

def create_visualization(df, query, chart_type=None):
    """Enhanced visualization function with multiple chart types"""
    query_lower = query.lower()
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    try:
        # Bar Chart
        if any(word in query_lower for word in ["bar", "count", "frequency"]) or chart_type == "bar":
            if len(categorical_cols) > 0:
                col_to_plot = categorical_cols[0]
                # Find the categorical column mentioned in query
                for col in categorical_cols:
                    if col in query_lower:
                        col_to_plot = col
                        break
                
                # Create value counts and reset index properly
                value_counts = df[col_to_plot].value_counts().reset_index()
                value_counts.columns = [col_to_plot, 'count']  # Rename columns explicitly
                
                fig = px.bar(value_counts, 
                           x=col_to_plot, y='count',
                           title=f'Count of {col_to_plot.replace("_", " ").title()}')
                st.plotly_chart(fig, use_container_width=True)
                return True
        
        # Histogram
        elif any(word in query_lower for word in ["histogram", "distribution"]) or chart_type == "histogram":
            if len(numeric_cols) > 0:
                col_to_plot = numeric_cols[0]
                for col in numeric_cols:
                    if col in query_lower:
                        col_to_plot = col
                        break
                
                fig = px.histogram(df, x=col_to_plot, 
                                 title=f'Distribution of {col_to_plot.replace("_", " ").title()}')
                st.plotly_chart(fig, use_container_width=True)
                return True
        
        # Scatter Plot
        elif any(word in query_lower for word in ["scatter", "relationship", "correlation"]) or chart_type == "scatter":
            if len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                
                # Try to find specific columns mentioned in query
                mentioned_cols = [col for col in numeric_cols if col in query_lower]
                if len(mentioned_cols) >= 2:
                    x_col, y_col = mentioned_cols[0], mentioned_cols[1]
                
                fig = px.scatter(df, x=x_col, y=y_col,
                               title=f'{y_col.replace("", " ").title()} vs {x_col.replace("", " ").title()}')
                st.plotly_chart(fig, use_container_width=True)
                return True
        
        # Line Chart
        elif any(word in query_lower for word in ["line", "trend", "time", "over time"]) or chart_type == "line":
            if len(numeric_cols) >= 1:
                # For line charts, try to use index or a sequence as x-axis
                if len(numeric_cols) >= 2:
                    x_col, y_col = numeric_cols[0], numeric_cols[1]
                    # Try to find specific columns mentioned in query
                    mentioned_cols = [col for col in numeric_cols if col in query_lower]
                    if len(mentioned_cols) >= 2:
                        x_col, y_col = mentioned_cols[0], mentioned_cols[1]
                    
                    fig = px.line(df.reset_index(), x='index', y=y_col,
                                title=f'{y_col.replace("_", " ").title()} Trend')
                else:
                    # Single numeric column - plot against index
                    y_col = numeric_cols[0]
                    for col in numeric_cols:
                        if col in query_lower:
                            y_col = col
                            break
                    
                    fig = px.line(df.reset_index(), x='index', y=y_col,
                                title=f'{y_col.replace("_", " ").title()} Trend')
                
                st.plotly_chart(fig, use_container_width=True)
                return True
        
        # Box Plot
        elif any(word in query_lower for word in ["box", "outlier"]) or chart_type == "box":
            if len(numeric_cols) > 0:
                col_to_plot = numeric_cols[0]
                for col in numeric_cols:
                    if col in query_lower:
                        col_to_plot = col
                        break
                
                fig = px.box(df, y=col_to_plot,
                           title=f'Box Plot of {col_to_plot.replace("_", " ").title()}')
                st.plotly_chart(fig, use_container_width=True)
                return True
        
        # Pie Chart
        elif any(word in query_lower for word in ["pie", "proportion", "percentage"]) or chart_type == "pie":
            if len(categorical_cols) > 0:
                col_to_plot = categorical_cols[0]
                for col in categorical_cols:
                    if col in query_lower:
                        col_to_plot = col
                        break
                
                value_counts = df[col_to_plot].value_counts()
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f'Distribution of {col_to_plot.replace("_", " ").title()}')
                st.plotly_chart(fig, use_container_width=True)
                return True
        
        # Heatmap (correlation matrix)
        elif any(word in query_lower for word in ["heatmap", "correlation matrix"]) or chart_type == "heatmap":
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                              title="Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)
                return True
        
        return False
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return False

def display_data_overview(df):
    """Display comprehensive data overview"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    with col4:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

def show_data_quality_report(df):
    """Show data quality issues and suggestions"""
    st.subheader("📋 Data Quality Report")
    
    issues = []
    
    # Check for missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        issues.append(f"*Missing Values:* {len(missing_cols)} columns have missing values: {', '.join(missing_cols)}")
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"*Duplicate Rows:* {duplicates} duplicate rows found")
    
    # Check for potential data type issues
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        try:
            if df[col].str.isnumeric().any():
                issues.append(f"*Data Type Issue:* Column '{col}' contains numeric values but is stored as text")
        except:
            pass
    
    if issues:
        for issue in issues:
            st.warning(issue)
    else:
        st.success("✅ No major data quality issues detected!")

# --- Main Streamlit App ---
def main():
    st.title("📊 Excel Insight Chatbot")
    st.markdown("Upload your Excel file and get AI-powered insights with interactive visualizations!")
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'original_columns' not in st.session_state:
        st.session_state.original_columns = None
    
    # Sidebar
    with st.sidebar:
        st.header("📁 File Upload")
        uploaded_file = st.file_uploader(
            "Choose your Excel file",
            type=["xlsx", "xls"],
            help="Upload an Excel file (.xlsx or .xls format)"
        )
        
        if uploaded_file:
            st.success("✅ File uploaded successfully!")
            
            # File info
            st.info(f"*Filename:* {uploaded_file.name}")
            st.info(f"*Size:* {uploaded_file.size / 1024:.2f} KB")
        
        st.header("🎨 Quick Visualizations")
        if uploaded_file and st.session_state.df is not None:
            chart_types = ["Auto-detect", "Bar Chart", "Histogram", "Scatter Plot", 
                          "Line Chart", "Box Plot", "Pie Chart", "Heatmap"]
            selected_chart = st.selectbox("Select chart type:", chart_types)
            
            if st.button("Create Visualization"):
                if selected_chart != "Auto-detect":
                    chart_type_map = {
                        "Bar Chart": "bar",
                        "Histogram": "histogram", 
                        "Scatter Plot": "scatter",
                        "Line Chart": "line",
                        "Box Plot": "box",
                        "Pie Chart": "pie",
                        "Heatmap": "heatmap"
                    }
                    create_visualization(st.session_state.df, "", chart_type_map.get(selected_chart, "bar"))
    
    # Main content area
    if uploaded_file is not None:
        try:
            # Load and process data
            with st.spinner("🔄 Processing your Excel file..."):
                df = pd.read_excel(uploaded_file)
                original_columns = df.columns.tolist()
                df = normalize_column_names(df)
                
                # Store in session state
                st.session_state.df = df
                st.session_state.original_columns = original_columns
            
            # Display data overview
            display_data_overview(df)
            
            # Tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Preview", "📈 Quick Stats", "🔍 Data Quality", "💬 AI Chat"])
            
            with tab1:
                st.subheader("📊 Data Preview")
                
                # Show original vs normalized column names
                with st.expander("🔄 Column Name Mapping"):
                    col_mapping = pd.DataFrame({
                        'Original': original_columns,
                        'Normalized': df.columns
                    })
                    st.dataframe(col_mapping, use_container_width=True)
                
                # Data preview with pagination
                rows_per_page = st.selectbox("Rows per page:", [10, 25, 50, 100], index=1)
                total_rows = len(df)
                total_pages = (total_rows - 1) // rows_per_page + 1
                
                if total_pages > 1:
                    page = st.number_input("Page:", min_value=1, max_value=total_pages, value=1)
                    start_idx = (page - 1) * rows_per_page
                    end_idx = min(start_idx + rows_per_page, total_rows)
                    st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)
                    st.caption(f"Showing rows {start_idx + 1}-{end_idx} of {total_rows}")
                else:
                    st.dataframe(df, use_container_width=True)
            
            with tab2:
                st.subheader("📈 Quick Statistics")
                
                # Numeric columns statistics
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    st.write("*Numeric Columns Summary:*")
                    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                
                # Categorical columns info
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                if categorical_cols:
                    st.write("*Categorical Columns Info:*")
                    cat_info = []
                    for col in categorical_cols:
                        cat_info.append({
                            'Column': col,
                            'Unique Values': df[col].nunique(),
                            'Most Frequent': df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A',
                            'Frequency': df[col].value_counts().iloc[0] if not df[col].empty else 0
                        })
                    st.dataframe(pd.DataFrame(cat_info), use_container_width=True)
            
            with tab3:
                show_data_quality_report(df)
                
                # Detailed column information
                st.subheader("📋 Column Details")
                col_details = []
                for col in df.columns:
                    col_details.append({
                        'Column': col,
                        'Data Type': str(df[col].dtype),
                        'Non-Null Count': df[col].count(),
                        'Null Count': df[col].isnull().sum(),
                        'Unique Values': df[col].nunique(),
                        'Memory Usage (bytes)': df[col].memory_usage(deep=True)
                    })
                st.dataframe(pd.DataFrame(col_details), use_container_width=True)
            
            with tab4:
                st.subheader("💬 AI-Powered Analysis")
                
                # Prepare data info for AI
                data_summary = get_data_summary(df)
                sample_data = df.head(3).to_string() if len(df) > 0 else "No data available"
                
                # Example questions
                st.write("💡 Example Questions:")
                example_questions = [
                    "What are the key insights from this data?",
                    "Show me a bar chart of the most common values",
                    "What patterns do you see in the numeric columns?",
                    "Are there any outliers or anomalies?",
                    "Create a correlation analysis between numeric variables",
                    "What's the distribution of the categorical variables?"
                ]
                
                cols = st.columns(2)
                for i, question in enumerate(example_questions):
                    col = cols[i % 2]
                    if col.button(question, key=f"example_{i}"):
                        st.session_state.user_query = question
                
                # User input
                user_query = st.text_input(
                    "Ask a question about your data:",
                    value=st.session_state.get('user_query', ''),
                    placeholder="e.g., 'Show me a histogram of sales data' or 'What are the main trends?'"
                )
                
                if user_query:
                    with st.spinner("🤖 Analyzing your data..."):
                        # Try to create visualization first
                        viz_created = create_visualization(df, user_query)
                        
                        # Get AI response or fallback to basic insights
                        ai_response = ask_groq(user_query, str(data_summary), sample_data)
                        
                        # If API failed and no helpful response, provide basic insights
                        if any(phrase in ai_response for phrase in ["API Configuration Required", "Invalid API Key", "Connection Error"]):
                            basic_insights = get_basic_insights(df, user_query)
                            st.markdown("### 🤖 AI Analysis")
                            st.markdown(ai_response)
                            st.markdown("### 📊 Basic Data Insights")
                            st.markdown(basic_insights)
                        else:
                            st.markdown("### 🤖 AI Analysis")
                            st.markdown(ai_response)
                        
                        if viz_created:
                            st.success("📊 Visualization created above!")
                        
                        # Clear the query from session state
                        if 'user_query' in st.session_state:
                            del st.session_state.user_query
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.error("Please make sure you've uploaded a valid Excel file (.xlsx or .xls)")
    
    else:
        # Welcome message
        st.markdown("""
        ### Welcome to Excel Insight Chatbot! 🎉
        
        This powerful tool helps you:
        - 📊 *Analyze* your Excel data with AI
        - 📈 *Visualize* patterns and trends
        - 🔍 *Discover* insights automatically
        - 💬 *Chat* with your data using natural language
        
        *Get started by uploading an Excel file using the sidebar!*
        
        ---
        
        #### 🚀 Features:
        - *Smart Data Processing*: Automatic column normalization and type detection
        - *AI-Powered Analysis*: Get insights using advanced language models
        - *Interactive Visualizations*: Multiple chart types with Plotly
        - *Data Quality Reports*: Identify and fix data issues
        - *Natural Language Queries*: Ask questions in plain English
        
        #### 💡 Example Use Cases:
        - Sales data analysis and forecasting
        - Customer behavior insights
        - Financial data exploration
        - Survey response analysis
        - Inventory management insights
        """)
        
        # Setup instructions
        with st.expander("🔧 Setup Instructions"):
            st.markdown("""
            1. *Install required packages:*
            bash
            pip install streamlit pandas matplotlib seaborn plotly requests python-dotenv numpy openpyxl
            
            
            2. *Set up your environment (Optional - for AI features):*
            Create a .env file in your project directory with:
            
            GROQ_API_KEY=your_groq_api_key_here
            
            
            3. *Get a Groq API key (Optional):*
            - Visit [console.groq.com](https://console.groq.com)
            - Sign up for a free account
            - Generate an API key
            
            4. *Run the app:*
            bash
            streamlit run app.py
            
            
            *Note:* The app works perfectly without the API key - you'll still get all visualizations and basic insights!
            """)

if __name__ == "__main__":
    main()
