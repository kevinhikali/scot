docker stop $(docker ps -q)
docker rm $(docker ps -aq)

bash dataset/visualwebarena/scripts/reset_reddit.sh
bash dataset/visualwebarena/scripts/reset_shopping.sh

cd /Users/kevin_air/data/vwa/classifieds_docker_compose
docker compose up --build -d
cd -

curl -X POST http://localhost:9980/index.php?page=reset -d "token=4b61655535e7ed388f0d40a93600254c"

docker run -d --name=wikipedia --volume=/Users/kevin_air/data/vwa/:/data -p 8888:80 ghcr.io/kiwix/kiwix-serve:3.3.0 wikipedia_en_all_maxi_2022-05.zim

cd dataset/visualwebarena/environment_docker/webarena-homepage/
flask run --host=0.0.0.0 --port=4399