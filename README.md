# tek-ingineer-pfe

## Vérifiez les variables d'environnement dans le conteneur MLflow

`docker exec -it mlflow_server env | grep AWS`

## vider les expereinces et artefacts supprimer de la bd

## gc
Supprimez définitivement les exécutions de l' étape de cycle de vie supprimée du magasin principal spécifié. Cette \
commande supprime tous les artefacts et métadonnées associés aux exécutions spécifiées. Si l'URL d'artefact fournie \
n'est pas valide, la suppression de l'artefact sera ignorée et le processus gc se poursuivra.

`aws configure --profile minio`
`export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000`

`mlflow gc --backend-store-uri postgresql://mlflowAdmin:Admin123@localhost:5432/pfe-db-monitoring --artifacts-destination s3://mlflow/ --experiment-ids 13`

## Vérifier l'Existence du Bucket 
`aws --profile minio s3 ls --endpoint-url http://localhost:9000`
