# auto-ml-comparison

- Habilitar virtualização na BIOS da placa-mãe ([exemplo](https://www.youtube.com/watch?v=GK0DOfdLCa8))

- Abrir Prompt de Comando como Administrador
	- `bcdedit /set hypervisorlaunchtype off`
	- `DISM /Online /Disable-Feature:Microsoft-Hyper-V`

- No menu Iniciar, digitar "Ativar ou desativar recursos do Windows"
	- Desabilitar Hyper-V
	- Desabilitar Plataforma de Máquina Virtual
	- Desabilitar Plataforma do Hipervisor do Windows

- Reiniciar o computador

- Baixar o [Oracle VM VirtualBox](https://download.virtualbox.org/virtualbox/7.0.4/VirtualBox-7.0.4-154605-Win.exe)

- Baixar o Ubuntu [22.04.1 LTS](https://releases.ubuntu.com/22.04/ubuntu-22.04.1-desktop-amd64.iso)

- Criar VM do Ubuntu no VirtualBox
	- Usuário e senha automl; nome da máquina automl-vbox
	- Alocar pelo menos 40GB de disco, 8GB de RAM e 2 núcleos de CPU

- Executar a VM do Ubuntu e proceder com a instalação (desmarcar a opção Unattended installation)

- Assim que o Ubuntu estiver instalado, abrir o terminal e digitar:
	- sudo apt-get update
	- sudo apt-get install gcc make perl
	- Instalar VBox Guest Additions
		- Clicar em "Dispositivos" > Inserir imagem para convidados
		- Copiar o CD montado para a pasta do usuário (`cp -r /media/automl/VBox_GAs_7.0.2 ~/`)
		- Alterar para a pasta do usuário (`cd ~/VBox_GAs_7.0.2/`)
		- Executar o instalador com privilégio de administrador (`sudo ./VBoxLinusAdditions.run`)

- Verificar se a "tartaruga verde" não está aparecendo no canto inferior direito da VM

- Verificar se a instrução AVX está na lista de instruções suportadas pelo CPU
	- `more /proc/- cpuinfo | grep flags | grep avx`
	
- Executar os seguintes comandos no Terminal:
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
	- Adicionar `export GCM_CREDENTIAL_STORE=secretservice` ao final do arquivo ~/.bashrc
	- `cd ~/ && mkdir git && cd ~/git/`
	- `git clone https://github.com/marcelovca90/auto-ml-comparison.git`
	- `cd auto-ml-comparison`
	- `chmod +x run.sh`
	- `./run.sh`
