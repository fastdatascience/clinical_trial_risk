export COMMIT_ID=`git show -s --format=%ci_%h | sed s/[^_a-z0-9]//g | sed s/0[012]00_/_/g`
docker build -t protocols-front --build-arg COMMIT_ID=$COMMIT_ID .
docker tag protocols-front regprotocolsfds.azurecr.io/protocols-front:$COMMIT_ID
docker push regprotocolsfds.azurecr.io/protocols-front:$COMMIT_ID
echo "The container version is $COMMIT_ID"
