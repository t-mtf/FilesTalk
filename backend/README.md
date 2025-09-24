# How to push on the branch
- Create your own local branch from main
- Do commits on your local branch (you can disable jobs by having a commit message that starts with "no-test" when you're not on main branch)
- Push your branch on GitLab
- Create a tag of your branch (format expected: X.Y.Z-dev)
- Test your image in the dev cluster
- If your tag is not valide: modify your code and push it on your branch
- Test your image in the dev cluster again
- If your tag is valide: Merge your branch into main
- Create a tag from main (format expected: X.Y.Z)
- You can now use your image in UAT then Prod
# Prerequisites :
 
- python 
- poetry 

## Installing Poetry:

Poetry is recommended to be installed via pipx. Follow these steps:

1. Update your system and install pipx:

```bash
# On Ubuntu
sudo apt update
sudo apt install pipx
pipx ensurepath
```

2. Once pipx is installed, you can install Poetry:

```bash
pipx install poetry
```


# Project Setup:
From the backend directory, execute the following commands:

1. Install the project dependencies:
```sh
poetry install
```

2. Activate the virtual environment created by Poetry:

```bash
poetry shell
```
3. Set up the pre-commit package:

The pre-commit package is used to ensure consistent code quality and style. Each time you commit, a series of tests will be run based on the pre-commit configurations. You will be able to commit the changes only if these tests pass.

Note: The Pylint package included in pre-commit needs to be run locally. When adding new hooks, it's recommended to run them against all files to ensure consistency:

```bash
pre-commit install
pre-commit run --all-files
```

4. Committing Changes to Git:

After making changes to your files, you can add them to your Git repository, commit, and push your changes to the remote repository with the following commands:

```bash
git add filename
git commit -m "commit message"
# push the files if all tests pass
git push
```

# Starting the Development Server:

Run the following command to start the development server:

```sh
python manage.py runserver
```

# Runnig backend locally

Some files need to be modified to run the backend locally:

### create_superuser.sh

```bash
# remove this line
until telnet postgres-service.$NAMESPACE.svc.cluster.local 5432; do

# add this line instead
until telnet postgres 5432; do
``` 

### config/settings.py

```bash
# remove these lines
# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False
ALLOWED_HOSTS = [os.getenv("ALLOWED_HOST")]

# replace them with
DEBUG = True
# ALLOWED_HOSTS = [os.getenv("ALLOWED_HOST")]


# Change HOST
# Remove
"HOST": f"redis-service.{os.getenv('NAMESPACE')}.svc.cluster.local",

# Replace with
"HOST": "redis_server",
```

### Dockerfile

Use the provided dockerfile instead of the one from the gitlab repo. 

## Running the Application


- To launch the application, build and start Docker compose using the following command:

```bash
docker compose up --build
```

The backend endpoints documentation can be found [here](https://espace.agir.orange.com/display/GENAC/API). When running the application locally, use `http://localhost:8000` as the `BASE URL`.

Before proceeding, obtain the API-key from the development team. You can use Postman to make API calls.
### Testing the API

To test the API, call the hello endpoint using the GET method as shown below:
```bash
http://localhost:8000/api/hello
``` 
If the API is working correctly, you should receive the following response:

```json
{
    "message": "Hello, World!"
}
```

### Generating a Token


For batch processing requests, you first need to obtain a token using the POST method.
```bash
http://localhost:8000/api/token
```
Body content:

```json
{"cuid":"ABCD1234"}
``` 

Example response:

```json
{
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzM5OTYwOTc5LCJpYXQiOjE3Mzk5NTczNzksImp0aSI6ImI3MmRkNThiM2UyMzQ4NThhYTA2OTE0YjUwZGM5MDQ2IiwidXNlcl9pZCI6NH0.zApPBGRDyuogj2Oi2K8VApRjcdMlqFqMbyZD7hlAmUg",
    "expires_in": 3600
}
``` 

### Creating a Batch Job

Use this endpoint with a POST method to make batch processing requests.

```bash
http://localhost:8000/api/batch
```
Body content:
```json
{
  "cuid": "ABCD1234",
  "scope": {
       "ic01_list": ["ic01_1", "ic01_2"]
        },
  "prompts": [
        {"name": "prompt_1", "value": "Give me a summary of the document"}
        ]
}
``` 

In the Authorization tab, select Bearer Token as the Auth Type and use the token you retrieved earlier. 

Example response:
```json
{
    "job_id": "d9cb6f56-08ca-4659-976f-b2f98b90d087",
    "creation_date": "2025-02-19T10:33:57.668328Z",
    "status": "queued",
    "queue_rank": -1
}
``` 

### Checking Job Status

You can check the status of a job using the `job_id` and the following endpoint with the GET method:

```bash
http://localhost:8000/api/batch/d9cb6f56-08ca-4659-976f-b2f98b90d087
``` 
Example response:
```json
{
    "status": "started",
    "queue_rank": -1
}
```

### Retrieving Results

Once the job is finished, you can retrieve the results using the `job_id` and the following endpoint with the GET method:

```bash
http://localhost:8000/api/batch/d9cb6f56-08ca-4659-976f-b2f98b90d087/file
``` 
Example response:
```json
{
    "file_name": "d9cb6f56-08ca-4659-976f-b2f98b90d087.xlsx",
    "file_content": "UEsDBBQAAAAIAAAAPwDV8zb9WAEAAJ8FAAATAAAAW0NvbnRlbnRfVHlwZXNdLnhtbM2Uy27CMBBF9/i2efb98Pk5LkzK0TRhc0vT9QwnVtGqG7gv45He8+UeKA6YZJo3lBZ+7oY/khdzBL/="
}
```
Note: The `file_content` is encoded with the base64 package. You can decode it using this Python script.

```python
import base64
import json
from pathlib import Path
# Assume the results file is located in the current directory
results_directory = Path(".") 
file_path = results_directory / "results.json" 
with open(file_path, "r") as f:
    data = json.load(f)

decoded_content = base64.b64decode(data["file_content"])

output_file_path = results_directory / data["file_name"]
with open(output_file_path, "wb") as f:
    f.write(decoded_content)
```

## Cleaning Up

After you're done testing the API and retrieving results, you might want to clean up the Docker environment. You can do this by running the following commands:

```bash
docker compose down
docker volume prune
docker volume ls
docker volume rm volume_name 
```
