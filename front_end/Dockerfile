FROM python:3.9
# Install Java - necessary for Tika
RUN apt-get update
#RUN apt-get install default-jre -y
RUN apt-get install wkhtmltopdf -y
# RUN wget https://dlcdn.apache.org/tika/2.4.0/tika-server-standard-2.4.0.jar

#ENV TIKA_SERVER_JAR=https://dlcdn.apache.org/tika/2.4.0/tika-server-standard-2.4.0.jar
# Special treatment for Tika and Spacy at beginning of building container
#RUN pip install tika==1.24
RUN pip install spacy==3.2.3
# Make sure Tika Jar file is downloaded
#RUN python -c 'import tika; tika.initVM(); from tika import parser; import io; parser.from_buffer(io.BytesIO(b""), xmlContent=True)'
# Make sure Spacy language model is downloaded
RUN python -m spacy download en_core_web_sm

# Create a directory where the code is to be hosted
RUN mkdir /app
# Define the working directory in the container
WORKDIR /app 
# Copy and install the requirements.
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

RUN python -c 'import nltk; nltk.download("stopwords")'

# Copy application code to the image
COPY . /app/

# Define environment variables
ENV dash_port=80
ENV dash_debug=False

ARG COMMIT_ID="No commit ID specified"
ENV COMMIT_ID=$COMMIT_ID

CMD ["python", "application.py"]
