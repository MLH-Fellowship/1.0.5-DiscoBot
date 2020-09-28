DiscoBot
========

*sentiment analysis discord bot*

- `biLSTM` trained on `iMDB` for sentiment analysis
- `torchtext` was built from `master`, which can be found here_

instructions
~~~~~~~~~~~~

`make help` for further instructions

- To train the model do `make train` 
- To package the model to bento do `make && make build`
- to run the docker container do `make run`
- endpoint can be accessed through `POST` request via `localhost:5000`

.. code-block:: shell

	curl -X POST 0.0.0.0:5000/predict -H "accept: */*" -H "Content-Type: application/json" \
	-d "{\"text\":\"I hate you this is the worst experience I have ever seen\"}"
	# > 0.2898484170436859

building `torchtext` from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `git clone --recurse-submodules https://github.com/MLH-Fellowship/1.0.5-DiscoBot && cd torchtext`
- `git submodules update --init --recursive && python setup.py clean install`
- to prep the pretrained embedding run `make prep`, otherwise if you want to train the model parses `ARGS=--train`, like so:

.. code-block:: shell

    # This will train the model
    make ARGS=--train prep 

.. _here: https://github.com/pytorch/text
