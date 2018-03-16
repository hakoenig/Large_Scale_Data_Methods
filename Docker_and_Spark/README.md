# Flower classification with Spark and Docker

Installing Docker on EC2 instance according to AWS instructions:
```
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user
sudo curl -L https://github.com/docker/compose/releases/download/1.19.0/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose
```

Log out and log back in again to pick up the new docker group permissions. Verify that the ec2-user can run Docker commands without sudo:

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

Get docker containers:
`docker-compose up`

In order to connect to the web-app, we need to get the IP of the EC2 instance. Once you have the public DNS of your EC2 instance, point your browser to:
<EC2 instance Public DNS>:5000

----
If you want to recreate the Random Forest, run
`python spark_rf.py` in the *api* dir.
