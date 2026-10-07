# Use a lightweight Python image.
FROM python:3.12-slim

# Create the non-root user expected by Hugging Face Spaces.
RUN useradd -m -u 1000 user

# Use the non-root user.
USER user

# Configure the user's home and local Python package path.
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory inside the container.
WORKDIR $HOME/app

# Copy the project into the container.
COPY --chown=user . $HOME/app

# Install the Python dependencies required by the app.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Streamlit uses port 8501.
EXPOSE 8501

# Start the Streamlit application.
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]