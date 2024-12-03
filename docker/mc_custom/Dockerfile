FROM alpine:latest

# Installer bash, grep, et minio-client
RUN apk --no-cache add bash grep curl && \
    wget https://dl.min.io/client/mc/release/linux-amd64/mc && \
    chmod +x mc && \
    mv mc /usr/local/bin/

# Ajouter le script pour créer un bucket
COPY create_bucket.sh /usr/local/bin/create_bucket.sh
RUN chmod +x /usr/local/bin/create_bucket.sh

ENTRYPOINT ["/usr/local/bin/create_bucket.sh"]

