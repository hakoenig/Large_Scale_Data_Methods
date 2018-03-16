# Flower classification with Spark and Docker
----
For this project, I built a simple web application with flask that enables a user to input values for features that are then passed to a Random Forest Classifier implemented in pyspark. The classifier was trained on the Iris dataset and makes predictions based on 4 flower attributes entered by the user. The estimated probabilities are then displayed on another page.
The web application and machine learning model are hosted in separate docker containers, with both containers being launch-able by the single docker compose file docker_compose.yml.
----
## How to get it up and running on EC2 instance
We install Docker on an EC2 instance according to the AWS instructions:
```
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user
sudo curl -L https://github.com/docker/compose/releases/download/1.19.0/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

Next, log out and log back in again to pick up the new docker group permissions. Verify that the ec2-user can run Docker commands without sudo:
```
docker info
```

Get the base image:

`docker pull jupyter/pyspark-notebook`

Get git and clone the repo:
```
sudo yum install git -y
git clone https://github.com/hakoenig/Large_Scale_Data_Methods.git
cd Large_Scale_Data_Methods/Docker_and_Spark
```

Get the docker containers up and running:

`docker-compose up`

In order to connect to the web-app, we need to get the IP of the EC2 instance. Once you have the public DNS of your EC2 instance, point your browser to (make sure to open port 5000 in your security group):
<EC2 instance Public DNS>:5000

----
If you want to recreate the Random Forest, run
`python spark_rf.py` in the *api* dir.
----
Shout-out to https://github.com/mdagost/pug_classifier - the video and repo were very helpful!
