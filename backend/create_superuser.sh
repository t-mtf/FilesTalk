#!/bin/sh

# Wait for db
until telnet postgres-service.$NAMESPACE.svc.cluster.local 5432; do
  echo "Waiting for the database to start on port 5432..."
  sleep 2
done
echo "Database is up!"

# Make migrations
python manage.py makemigrations api

# Run migrations
python manage.py migrate

# Create partner users and dev team users
python manage.py create_user 
