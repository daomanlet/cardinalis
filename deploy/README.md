1. Install docker
2. Start docker daemon
3. sudo docker images 
4. sudo docker build . -t rouynxia/cardinalis
5. sudo docker images
6. delete images in local sudo docker images rm -f {image_id}
7. push to dockerhub
   7.1 sudo docker login -u rouynxia
   7.2 sudo docker push rouynxia/cardinalis:latest