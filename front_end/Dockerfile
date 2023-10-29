FROM python:3.9
# Install Java - necessary for Tika
RUN apt update && apt install -y default-jre tesseract-ocr default-jre wkhtmltopdf


# Create a directory where the code is to be hosted
RUN mkdir /app
# Define the working directory in the container
WORKDIR /app 
# Copy and install the requirements.
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Make sure Tika Jar file is downloaded
RUN python -c 'import io; from tika import parser; parser.from_buffer(io.BytesIO(b""), xmlContent=True)'
# Make sure Spacy language model is downloaded
RUN python -m spacy download en_core_web_sm
# Make sure NLTK stopwords are downloaded
RUN python -c 'import nltk; nltk.download("stopwords")'


# Copy application code to the image
COPY . /app/

# Define environment variables
ENV dash_port=80
ENV dash_debug=False

ARG COMMIT_ID="No commit ID specified"
ENV COMMIT_ID=$COMMIT_ID

CMD ["python", "application.py"]
