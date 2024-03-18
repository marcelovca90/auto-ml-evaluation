# A practical evaluation of AutoML tools for binary, multiclass, and multilabel classification

Authors: *Marcelo Aragão, Augusto Afonso, Rafaela Ferraz, Rairon Ferreira, Sávio Leite, Felipe A. P. de Figueiredo, and Samuel B. Mafra.*

## Abstract:
	Automated Machine Learning (AutoML) is a recent technology that provides speed to machine learning iterations and allows individuals with less experience to take advantage of existing tools. Due to several frameworks with different features, deciding the best option to solve each classification problem becomes difficult. It is necessary to consider aspects such as performance metrics and time when choosing the algorithm to reduce the demand for highly technical, specific knowledge in the subject. There are some comparisons of AutoML tools and approaches that perform tests in the area of data preprocessing, model selection, and hyperparameter optimization. However, most of these studies focus on binary and multiclass classification, not covering multilabel classifications and consequently not exploiting the full potential of the tools. In this paper, a comparative study between multiple AutoML tools is performed related to the features, architecture, capabilities, and results achieved on binary, multiclass, and multilabel classification problems from experimentation on various data sets.

## How to run our tests:
### To run our tests we need a Linux OS (bare-metal or virtualized).
#### If you are using Windows OS, you will need to make a few configurations to avoid future problems with some frameworks:
- Enable virtualization at your mainboard BIOS ([example](https://www.youtube.com/watch?v=GK0DOfdLCa8))

- Open the command prompt as Admin and type:
	- `bcdedit /set hypervisorlaunchtype off`
	- `DISM /Online /Disable-Feature:Microsoft-Hyper-V`

- At Windows menu, search for "Turn Windows features on or off" and:
	- Turn off Hyper-V
	- Turn off Virtual Machine Platform
	- Turn off Windows Hypervisor Platform

- Restart the computer

#### How to download and run the Virtual Machine:

- Download the [Oracle VM VirtualBox](https://download.virtualbox.org/virtualbox/7.0.4/VirtualBox-7.0.4-154605-Win.exe)

- Download the Ubuntu [22.04.1 LTS](https://releases.ubuntu.com/22.04/ubuntu-22.04.1-desktop-amd64.iso)

- Create the Ubuntu VM at VirtualBox:
	- Choose the downloaded ISO
	- Select the option `Skip Unattended Installation`
	- Define user and password as automl and machine name as automl-VirtualBox
	- allocate at least 40GB disk, 8GB RAM and 2 CPU cores

- Now execute the Ubunto VM and proceed with installation

#### After you have finished Ubuntu installation, follow these steps:
- Check if the AVX instruction is in the list of instructions supported by the CPU
	- `more /proc/- cpuinfo | grep flags | grep avx`
- Open the terminal and type:
	- `sudo apt-get update -y`
	- `sudo apt-get install gcc make perl -y`
- Install the VBox Guest Additions:
	- Click the "Devices" menu and select "Insert Guest Additions CD image"
	- Copy the image content to the user folder (`cp -r /media/automl/VBox_GAs_7.0.2 ~/`)
	- Change to the user folder (`cd ~/VBox_GAs_7.0.2/`)
	- Run the installer with admin privileges (`sudo ./VBoxLinusAdditions.run`)
	- `reboot`
	
- Then, execute the following commands at the terminal:
	- `sudo add-apt-repository ppa:deadsnakes/ppa -y`
	- `sudo apt-get update -y`
	- `sudo apt-get upgrade -y`
	- `sudo apt-get dist-upgrade -y`
	- `sudo apt-get autoclean -y`
	- `sudo apt-get autoremove -y`
	- `reboot`
	- `sudo apt-get install git build-essential software-properties-common htop swig python3.8 python3.8-dev python3.8-venv python3.8-distutils default-jre -y`
	- `snap install code --classic`
	- `wget "https://github.com/GitCredentialManager/git-credential-manager/releases/download/v2.0.696/gcmcore-linux_amd64.2.0.696.deb" -O /tmp/gcmcore.deb`
	- `sudo dpkg -i /tmp/gcmcore.deb`
	- `git-credential-manager-core configure`
	- Add `export GCM_CREDENTIAL_STORE=secretservice` at the end of the file ~/.bashrc (`gedit ~/.bashrc`)
		- If you have further problems, execute `export GCM_CREDENTIAL_STORE=cache` at the terminal.
	- `cd ~/ && mkdir git && cd ~/git/`
	- `git clone https://github.com/marcelovca90/auto-ml-evaluation.git`
	- `cd auto-ml-evaluation`
	- `chmod +x run.sh`
	- `./run.sh | tee run.log`
