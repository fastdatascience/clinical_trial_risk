docker login -u fastdatascience
export COMMIT_ID=`git show -s --format=%ci_%h | sed s/[^_a-z0-9]//g | sed s/0[012]00_/_/g`
docker build -t protocols-front --build-arg COMMIT_ID=$COMMIT_ID .
docker tag protocols-front fastdatascience/clinical_trial_risk:$COMMIT_ID
docker push fastdatascience/clinical_trial_risk:$COMMIT_ID
