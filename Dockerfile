# Stage 1: Use an official Python runtime as a parent image
FROM python:3.9-slim

# Stage 2: Set the working directory inside the container
WORKDIR /app

# Stage 3: Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 4: Copy the rest of your application code and data
COPY . .

# Stage 5: Expose the port that Streamlit runs on
EXPOSE 8501

# Stage 6: Define the command to run your application
CMD ["streamlit", "run", "app.py"]