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


.. _here: https://github.com/pytorch/text
