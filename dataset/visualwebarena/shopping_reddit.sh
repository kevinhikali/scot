docker stop $(docker ps -q)
bash dataset/visualwebarena/scripts/reset_reddit.sh
bash dataset/visualwebarena/scripts/reset_shopping.sh
docker run -d --name=wikipedia --volume=/Users/kevin/data/vwa/:/data -p 8888:80 ghcr.io/kiwix/kiwix-serve:3.3.0 wikipedia_en_all_maxi_2022-05.zim
cd dataset/visualwebarena/environment_docker/webarena-homepage/
flask run --host=0.0.0.0 --port=4399
