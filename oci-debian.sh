#!/bin/sh
echo ">>>[biniou oci 🧠 ]: biniou one-click installer for Debian based-distributions"
echo ">>>[biniou oci 🧠 ]: Installing prerequisites"
if [ "$(lsb_release -si|grep Debian)" != "" ]
  then
    su root -c "apt -y install git pip python3 python3-venv gcc perl make ffmpeg openssl libtcmalloc-minimal4"
  else
    sudo apt -y install git pip python3 python3-venv gcc perl make ffmpeg openssl libtcmalloc-minimal4
fi
echo ">>>[biniou oci 🧠 ]: Cloning repository"
git clone --branch main https://github.com/Woolverine94/biniou.git
echo ">>>[biniou oci 🧠 ]: Installing Virtual environment"
cd ./biniou
./install.sh
echo ">>>[biniou oci 🧠 ]: Installation finished. Use cd biniou && ./webui.sh to launch biniou. Press enter to exit"
read dummy
exit 0

