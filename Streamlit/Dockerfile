FROM continuumio/miniconda3
# miniconda3 une version light

WORKDIR /home/app

RUN apt-get update
#pour faire les mise à jour
RUN apt-get install nano unzip
RUN apt install curl -y

RUN curl -fsSL https://get.deta.dev/cli.sh | sh

COPY requirements.txt /dependencies/requirements.txt
# Il faut créer le folder dans le container
RUN pip install -r /dependencies/requirements.txt
#Intall toutes les librairies qui sont dans le fichier requirements

COPY . . 

#CMD streamlit run --server.port '80' --server.address "0.0.0.0" app.py

CMD streamlit run --server.port $PORT app.py
