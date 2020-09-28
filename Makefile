.PHONY := all

.DEFAULT_GOAL := init

all: help build init run

DEPLOY_DIR := deploy

SVC_DIR := $(HOME)/bentoml/repository/ProfanityFilterService/*

# ls -ltr $(SVC_DIR) | grep '^d' | tail -1 | awk '{print $(NF)}'
LATEST := $(shell ls -td -- $(SVC_DIR) | head -n1)

help: ## List of defined target
	@grep -E '^[a-zA-Z_-]+:.*?##.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'	

train:
	cd src && python train.py

init: ## package trained model to bento
	cd src && python packer.py

build: ## build bento docker images then deploy on 5000
	rm -rf $(DEPLOY_DIR) && cp -r $(LATEST) $(DEPLOY_DIR)
	cp requirements.txt $(DEPLOY_DIR) && cp src/config.yml $(DEPLOY_DIR)/ProfanityFilterService
	cp -r src/.data/aclImdb $(DEPLOY_DIR)/ProfanityFilterService/.data 
	cp -r src/.data/{aclImdb_v1.tar.gz,glove.6B.50d.txt} $(DEPLOY_DIR)/ProfanityFilterService/.data
	cd $(DEPLOY_DIR) && docker build -t aar0npham/profanity-filter:latest .
	
run: ## run docker images
	docker run -p 5000:5000 aar0npham/profanity-filter:latest

push: ## push docker images to repository
	docker push -t aar0npham/profanity-filter:latest
