#!/bin/sh
mc alias set minio http://minio:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}
if mc ls minio | grep -q 'mlflow'; then
    echo 'Bucket already exists'
else
    mc mb minio/mlflow && echo 'Bucket created successfully'
fi
