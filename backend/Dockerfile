# 1. Start with a stable Python 3.11 environment
FROM python:3.11

# 2. Set the working directory inside the container
WORKDIR /code

# 3. Copy just the requirements first and install them
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 4. Copy the rest of your backend code and the heavy AI model
COPY . .

# 5. Create a temporary folder for audio uploads and give it permissions
RUN mkdir -p /code/temp_uploads && chmod 777 /code/temp_uploads

# 6. Hugging Face REQUIRES your server to run on port 7860!
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]