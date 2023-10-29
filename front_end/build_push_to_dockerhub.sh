docker login -u fastdatascience
export COMMIT_ID=`git show -s --format=%ci_%h | sed s/[^_a-z0-9]//g | sed s/0[012]00_/_/g` && docker build -t clinical_trial_risk -t clinical_trial_risk:latest -t clinical_trial_risk:$COMMIT_ID -t fastdatascience/clinical_trial_risk:latest -t fastdatascience/clinical_trial_risk:$COMMIT_ID --build-arg COMMIT_ID=$COMMIT_ID . && docker push fastdatascience/clinical_trial_risk --all-tags && echo "The container version is $COMMIT_ID"