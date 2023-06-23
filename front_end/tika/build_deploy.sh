#az login
#az acr login --name regprotocolsfds
export COMMIT_ID=`git show -s --format=%ci_%h | sed s/[^_a-z0-9]//g | sed s/0[012]00_/_/g`
docker build -t protocols-tika --build-arg COMMIT_ID=$COMMIT_ID .
docker tag protocols-tika regprotocolsfds.azurecr.io/protocols-tika:$COMMIT_ID
docker push regprotocolsfds.azurecr.io/protocols-tika:$COMMIT_ID
echo "The container version is $COMMIT_ID"
