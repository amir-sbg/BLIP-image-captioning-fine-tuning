.PHONY: install test train clean

install:
	python -m pip install -r requirements.txt
	python -m pip install -e .

test:
	python -m pytest -q

train:
	PYTHONPATH=src python -m vlm_finetune.train

clean:
	rm -rf artifacts reports build dist src/*.egg-info
