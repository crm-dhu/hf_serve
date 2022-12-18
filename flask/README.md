# huggingmask
containerized a hugging face unmasker example in Flask.  A REST endpoint to fill in a MASK word in a string of words.

## reference

huggingface https://huggingface.co/bert-base-uncased/blob/main/README.md

## build docker 

```shell
% echo "${GITLAB_ACCESS_TOKEN}" | docker login -u ${GITLAB_USERNAME} --password-stdin git.infra.igodigital.com:4567
% docker build . --tag hugmask:latest
```

## run container

### set ENV variables

```shell
export FLASK_ENV=development
export NUM_WORKERS=5
export HTTP_PORT=8081
```
FLASK_ENV defaults to production if it is not set. 
Production runs gunicorn, rather than the Flask development WSGI HTTP Server.

NUM_WORKERS, recommends 1 + 2 * NUM_CORES.  Suppose 2 cores, NUM_WORKERS=5. 
This setting is relevant for gunicorn only.

HTTP_PORT defaults to 8080 if it is not set.  For development, it may give port conflict.

### run docker-compose
```shell
% docker-compose -f docker-compose.yml up
```

## REST endpoint 

Please note that it could take more than 1 minute to load the bert model.  The docker-compose step initializes the model loading.  

### request 
POST http://localhost:8081/unmask 

json body 
```json
{
    "input": "[MASK] is a music instrument."
}
```

### response 
```json
{
    "input": "[MASK] is a music instrument.",
    "output": [
        {
            "score": 0.7996816039085388,
            "sequence": "it is a music instrument.",
            "token": 2009,
            "token_str": "it"
        },
        {
            "score": 0.01909128949046135,
            "sequence": "this is a music instrument.",
            "token": 2023,
            "token_str": "this"
        },
        {
            "score": 0.014181786216795444,
            "sequence": "percussion is a music instrument.",
            "token": 6333,
            "token_str": "percussion"
        },
        {
            "score": 0.010806316509842873,
            "sequence": "flute is a music instrument.",
            "token": 8928,
            "token_str": "flute"
        },
        {
            "score": 0.010750767774879932,
            "sequence": "violin is a music instrument.",
            "token": 6710,
            "token_str": "violin"
        }
    ]
}
```
