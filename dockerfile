# Use an official lightweight Python image.
FROM python:3.10-slim

# Set the working directory.
WORKDIR /app

# Copy the requirements file.
COPY requirements.txt .

# Upgrade pip and install dependencies.
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Expose the port Streamlit uses.
EXPOSE 8501

# Run the Streamlit app.
CMD ["streamlit", "run", "master_client_chatbot.py", "--server.enableCORS", "false"]
