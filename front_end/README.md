# Front end code for the Clinical Trial Risk Tool

This folder contains all the code for running the front end in isolation, but not for training machine learning models.

If you only want to run the tool, but not develop it further, then everything you need is in this folder.

## Dependencies for PDF analysis

In order to run the tool you need to also have installed Apache Tika (https://tika.apache.org/) and run it (`java -jar tika[VERSION].jar`.

## Quick start guide (Python only, no Docker)

You need to install the requirements in `requirements.txt`.

```
pip install -r requirements.txt
```

Then you can run the tool using Python from the command line:

```
python application.py
```

## Developing the tool

It is recommended to open this folder in an IDE such as Pycharm, since there are many Python files that interact and import one another.

## Running with Docker/Docker-compose

The tool is architected as two separate Docker containers: the front end, which does most of the logic, and a Tika server which is Java-based and is for PDF extraction.

They can be run as Docker Compose using the `docker-compose.yml`, or individually.

If you run both containers individually, it is recommended to run the script in `build_deploy.sh`, or you can run more simply the following commands:

```
docker build -t protocols-front .
docker run protocols-front
```

To run the Tika container, cd into `tika` and run


```
docker build -t protocols-tika .
docker run protocols-tika
```

You will need to set environment variables on the front end container `COMMIT_ID` and `TIKA_SERVER_ENDPOINT` so that the front end knows where to find Tika.

If you have any problems, check the ports that both servers are running on and check that the ports are all correctly exposed and mapped.
