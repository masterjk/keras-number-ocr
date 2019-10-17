NAME="keras-number-ocr"

DOCKER_IMAGE="hub.docker.com/josephkiok/keras-number-ocr:latest"

default: build

build:
	@docker build -t "${DOCKER_IMAGE}" .

publish:
	@docker push ${IMAGE_NAME}

run-local: stop-local
	@docker run -p 5000:5000 --name ${NAME} --detach ${DOCKER_IMAGE}

stop-local:
	@docker stop ${NAME} || true
	@docker rm ${NAME} || true
