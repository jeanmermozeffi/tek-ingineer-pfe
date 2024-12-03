FROM postgres:latest
# use the Debian variant (not the Alpine) to "fake" /etc/passwd contents for initdb
# see: https://github.com/docker-library/docs/tree/master/postgres#arbitrary---user-notes

COPY init_db.sql /docker-entrypoint-initdb.d/