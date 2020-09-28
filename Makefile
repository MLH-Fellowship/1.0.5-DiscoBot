.PHONY := all

.DEFAULT_GOAL := init

all: help build

DEPLOY_DIR := deploy

SVC_DIR := $(HOME)/bentoml/repository/ProfanityFilterService/*

# ls -ltr $(SVC_DIR) | grep '^d' | tail -1 | awk '{print $(NF)}'
LATEST := $(shell ls -td -- $(SVC_DIR) | head -n1)

help: ## List of defined target
	@grep -E '^[a-zA-Z_-]+:.*?##.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'	

init: ## package trained model to bento
	cd src && python packer.py

build: ## build bento docker images then deploy on 5000
	rm -rf $(DEPLOY_DIR) && cp -r $(LATEST) $(DEPLOY_DIR)
	cp src/config.yml $(DEPLOY_DIR)/ProfanityFilterService
	cp -r src/.data/aclImdb $(DEPLOY_DIR)/ProfanityFilterService/.data
	cp src/.data/glove.6B.50d.txt $(DEPLOY_DIR)/ProfanityFilterService/.data
	cp requirements.txt $(DEPLOY_DIR)
	cd $(DEPLOY_DIR) && docker build -t profanity-filter:latest .
	docker run -p 5000:5000 profanity-filter:latest
