PROJECT_ID='model-training-491400'
REPO_NAME='noisy'
IMAGE_URI=us-central1-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/noisy-image:latest

gcloud auth configure-docker us-central1-docker.pkg.dev

docker build ./ -t $IMAGE_URI
docker push $IMAGE_URI
