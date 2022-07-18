# Protocol Analysis Tool for Clinical Trial Risk

Live demo available at: https://protocols.fastdatascience.com/

Based on the Dash Natural Gas demo: https://github.com/plotly/dash-sample-apps

Runs on Dash interactive Python framework developed by [Plotly](https://plot.ly/). 

Developed by Thomas Wood / Fast Data Science
thomas@fastdatascience.com

This tool is written in Python using the Dash front end library and the Java library Tika for reading PDFs, and runs on Linux, Mac, and Windows, and can be deployed as a web app using Docker.

## Very quick guide to running the tool on your computer

1. Install Git and [Git LFS](https://git-lfs.github.com/) and clone this repository with the following command:
```
git clone git@github.com:fastdatascience/clinical_trial_risk.git
```
2. Install [Docker](https://docs.docker.com/get-docker/).
3. Install [Docker Compose](https://docs.docker.com/compose/gettingstarted/).
4. Open a command line or Terminal window. Change folder to where you downloaded and unzipped the repository, and go to the folder `front_end`.  Run the following command:
```
docker-compose up
```
5. Open your browser at `https://localhost:80`
![The web app running](screenshots/applocal.png)

The username is `admin` and the password is `DsRNJmZ`.

## Developer's guide: Running the tool on your computer in Python and without using Docker

### Architecture

![Tool architecture](screenshots/protocol_risk_tool_architecture.png)

### Installing requirements

Download and install Java if you don't have it already. Download and install Apache Tika and run it on your computer https://tika.apache.org/download.html

```
java -jar tika-server-standard-2.3.0.jar
```

(the version number of your Jar file name may differ.)

Install everything in `requirements.txt`:

```
pip install -r requirements.txt
```

Install the Spacy model

```
python -m spacy download en_core_web_sm
```

Download the NLTK dependencies:

```
python -c 'import nltk; nltk.download("stopwords")'
```


### Running the front end app locally

Go into `front_end` and run

```
python application.py
```

You can then open your browser at `localhost:8050` and you will see the tool.

### Working with the training data and re-training the machine learning models

If you don't have the training data, go into `data/raw_protocols` folder and run

```
./download_protocols.sh
```

This will download a subset of the raw PDF training data from the internet.

Next, run the following command to preprocess the data:

```
python preprocess.py
```

This will populate the `data/preprocessed_text` and `data/preprocessed_tika` folders with the preprocessed data.

### Training the classifier model

Go into the `train` folder and run

```
python train_condition_classifier.py
python train_effect_estimate_classifier.py
python train_num_subjects_classifier.py
python train_sap_classifier.py
python train_simulation_classifier.py
python train_word_cloud.py
```

This will write to the following files:

* `app/models/condition_classifier.pkl.bz2` - the three-way Naive Bayes classifier model that classifies protocols into HIV, TB or other.
* `app/models/effect_estimate_classifier.pkl.bz2` - the two-way Naive Bayes classifier model that classifies individual tokens of a protocol into effect estimate or not effect estimate.
* `app/models/num_subjects_classifier.pkl.bz2` - the Random Forest regressor model to identify sample sizes.
* `app/models/sap_classifier.pkl.bz2` - the two-way Naive Bayes classifier model that classifies individual pages of a protocol into SAP or not SAP.
* `app/models/simulation_classifier.pkl.bz2` - the two-way Naive Bayes classifier model that classifies individual pages of a protocol into simulation or not simulation.
* `app/models/idfs_for_word_cloud.pkl.bz2` - the Inverse Document Frequencies of words in the training dataset, used to select words for the word cloud.

## Developer's guide: Deploying the tool as a web app to Microsoft Azure App Service

The Azure deployment consists of two separate web apps: a front end, and a Tika web app for PDF parsing.

### Creating a Tika Web App

The app depends on Tika as its PDF extraction library, which is written in Java. If you are running on the web instead of your computer, you need a remote Tika instance.

I deployed a Tika Docker instance on Azure Webapp.

This is from DockerHub, ID `apache/tika:2.3.0`.

So I went into Azure, selected New Webapp, and followed the options to create one from DockerHub.

Documentation is at https://github.com/apache/tika-docker

Now go into the Azure web portal at `portal.azure.com` and find the web app you created.

Also go into Settings -> Configuration -> Application settings and create an environment variable `PORT` and set it to `9998`. 

Finally, make sure to go into the Settings -> Configuration -> General settings and turn off "Always On"!

Note the URL of your Tika instance, e.g. `https://protocols-tika-webapp2.azurewebsites.net`.

### Deploying the Front End app as a Docker container to Microsoft Azure App Service via Azure Container Registry

Create a container registry in Microsoft Azure:

![alt text](screenshots/create_container_registry.png)

You need to enable the admin user in the Azure portal. Navigate to your registry, select Access keys under SETTINGS, then Enable under Admin user.

See here for more details: https://docs.microsoft.com/en-us/azure/container-registry/container-registry-authentication?tabs=azure-cli#admin-account

In command line, if you have installed Azure CLI, log into both the Azure Portal and Azure Container Registry:

```
az login
az acr login --name regprotocolsfds
```

If the admin user is not yet enabled, you can use the command:
```
az acr update -n regprotocolsfds --admin-enabled true
```

You can use Docker to deploy the front end app. In folder `front_end`, run command:

```
docker build -t protocols-front --build-arg COMMIT_ID=`git log | head -1 | awk '{print $2}'` .
```

(You can also run `./build_deploy.sh` as a shortcut)

To test the front end, you can run (with the appropriate value for your Tika endpoint URL):

```
docker run -p 80:80 -e TIKA_SERVER_ENDPOINT=https://protocols-tika-webapp2.azurewebsites.net protocols-front
```

Now to deploy it, you can run:

```
docker tag protocols-front regprotocolsfds.azurecr.io/protocols-front
docker push regprotocolsfds.azurecr.io/protocols-front
```


Now go into the Azure portal and create a web app:

![alt text](screenshots/create_webapp.png)

Choose the Docker options and select Azure Container Registry, and select protocols-front as the container.

Make sure to go into the Settings -> Configuration in Azure and turn off "Always On"!

![alt text](screenshots/turn_off_always_on.png)

Now tell the front end app where to find the Tika app. You can do this either by command line or in the web portal. To use command line, do:

```
az webapp config appsettings set -g protocol -n protocols-webapp --settings TIKA_SERVER_ENDPOINT=https://protocols-tika-webapp2.azurewebsites.net
```

Alternatively, find the web app in the Azure web portal and go into Configuration and set environment variable in the Azure Dash web app to point to your Tika instance, e.g.

```
TIKA_SERVER_ENDPOINT=[URL of your Tika instance]
```

It's also a good idea to turn on continuous deployment, then you can redeploy the app from the command line by pushing to the Docker Container Registry.

![alt text](screenshots/tika_env_var_config.png)

Also for the front end web app, make sure to go into the Settings -> Configuration in Azure and turn off "Always On"!

## Alternative deployment (which hasn't worked as well): Deploying both Docker Containers via Docker Compose to Microsoft Azure App Service via Azure Container Registry

Create a container registry in Microsoft Azure with a name according to your choice, for example `regprotocolsfds`.

If your Azure container registry is named `regprotocolsfds` then its address will be `regprotocolsfds.azurecr.io`.

Make sure that the address of the container registry is in the `app/docker-compose.yml` file.

![alt text](screenshots//create_container_registry.png)

You need to enable the admin user in the Azure portal. Navigate to your registry, select Access keys under SETTINGS, then Enable under Admin user.

See here for more details: https://docs.microsoft.com/en-us/azure/container-registry/container-registry-authentication?tabs=azure-cli#admin-account

In command line, if you have installed Azure CLI, log into both the Azure Portal and Azure Container Registry:

```
az login
az acr login --name regprotocolsfds
```

If the admin user is not yet enabled, you can use the command:
```
az acr update -n regprotocolsfds --admin-enabled true
```

Build the Docker Compose containers:

```
docker-compose build
```

Push the Docker container to the Azure container registry:

```
docker-compose push
```

![alt text](screenshots//create_webapp.png)

Choose the Docker options and select Container Type - Docker Compose and Registry Source - Azure Container Registry, and select protocols_prod as the container.

Set the Configuration File to `app/docker-compose.yml`.

Make sure to go into the Settings -> Configuration in Azure and turn off "Always On"!

![alt text](screenshots//turn_off_always_on.png)

You can configure the Azure web app so that every time you push a new container to the Azure container registry, the app is redeployed.

## Restricting access to the Tika webapp

You might want to set up an IP rule to make sure that only the front end app can access the Tika web app.

You can do this by going into the Tika webapp, and selecting Networking and Access Restrictions. You can add whitelist rules for the IP addresses of the front end app.

## Built With

- [Dash](https://dash.plot.ly/) - Main server and interactive components
- [Plotly Python](https://plot.ly/python/) - Used to create the interactive plots
- [Docker](https://docs.docker.com/) - Used for deployment to the web
- [Apache Tika](https://tika.apache.org/) - Used for parsing PDFs to text
- [spaCy](https://spacy.io/) - Used for NLP analysis
- [NLTK](https://www.nltk.org/) - Used for NLP analysis
- [Scikit-Learn](https://scikit-learn.org/) - Used for machine learning

## Licences of Third Party Software

- Apache Tika: [Apache 2.0 License](https://tika.apache.org/license.html)
- spaCy: [MIT License](https://github.com/explosion/spaCy/blob/master/LICENSE)
- NLTK: [Apache 2.0 License](https://github.com/nltk/nltk/blob/develop/LICENSE.txt)
- Scikit-Learn: [BSD 3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING)

## References

* Deploying a Dash webapp via Docker to Azure: https://medium.com/swlh/deploy-a-dash-application-in-azure-using-docker-ed46c4b9d2b2
