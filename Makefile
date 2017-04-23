
IMAGE_NAME = "captcha-solver"

build:
	docker build -t $(IMAGE_NAME) .

shell: build
	docker run --rm -it --net=host -v `pwd`:/data/ $(IMAGE_NAME) bash

run: build
	docker run --rm -it --net=host -v `pwd`:/data/ $(IMAGE_NAME)
