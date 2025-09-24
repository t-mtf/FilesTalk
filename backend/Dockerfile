FROM dockerproxy.repos.tech.orange/python:3.12-slim

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Installation des dépendances système
# hadolint ignore=DL3008
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    telnet \
    curl \
    gpg-agent \
    gnupg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Installation des certificats Orange
# No fixed version for orange-certs to ensure latest certificates
# hadolint ignore=DL3008
RUN curl -OsSL 'https://repo.yourdev.tech.orange/yourdev-install.sh' && \
    curl -sSL https://repo.yourdev.tech.orange/key.asc | gpg --dearmor -o /usr/share/keyrings/yourdev.gpg && \
    bash yourdev-install.sh && \
    apt-get update && \
    apt-get install -y --no-install-recommends orange-certs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* yourdev-install.sh

WORKDIR /tmp

COPY pyproject.toml poetry.lock /tmp/

RUN pip install --no-cache-dir poetry==1.8.3 && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction

COPY . /tmp/

RUN chmod +x /tmp/create_superuser.sh && \
    python manage.py collectstatic --noinput

EXPOSE 8000

WORKDIR /app

CMD ["sh", "-c", "cp -r /tmp/* /app/ && ./create_superuser.sh && gunicorn --log-level debug --access-logfile access.log --error-logfile error.log --capture-output --reload --bind 0.0.0.0:8000 config.wsgi:application"]
