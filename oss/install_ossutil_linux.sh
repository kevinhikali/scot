rm -f ossutil64
wget http://antsys-oss-tools-dev.cn-hangzhou-alipay-b.oss-cdn.aliyun-inc.com/ossutil64      
chmod 755 ossutil64
current_dir=$(pwd)
echo export PATH='$PATH':$current_dir >> ~/.bashrc
source ~/.bashrc