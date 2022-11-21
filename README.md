# auto-ml-comparison

Criar VM do Ubuntu (22.04.1 LTS) no VirtualBox
(usuário e senha automl; nome da máquina automl-vbox)
(alocar pelo menos 8GB de RAM e 2 núcleos de CPU)
(instalar VBox Guest Additions já durante o setup)

sudo add-apt-repository ppa:deadsnakes/ppa -y

sudo apt-get update -y

sudo apt-get upgrade -y

sudo apt-get dist-upgrade -y

sudo apt-get autoclean -y

sudo apt-get autoremove -y

reboot

sudo apt-get install git build-essential software-properties-common swig python3.8 python3.8-dev python3.8-venv python3.8-distutils default-jre -y

snap install code --classic

git clone https://github.com/marcelovca90/auto-ml-comparison.git

cd auto-ml-comparison

chmod +x run.sh

./run.sh

