.PHONY: train run test docker-build docker-run

train:
\tpython scripts/train_model.py

run:
\tFLASK_APP=app.app:app flask run --host=0.0.0.0 --port=5000

test:
\tpytest -q

docker-build:
\tdocker build -t breast-cancer-api:latest .

docker-run:
\tdocker run --rm -p 5000:5000 --name bc-api breast-cancer-api:latest