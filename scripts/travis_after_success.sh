#!/usr/bin/env bash
echo "This script is run automatically after_success."

## The actions below tries to mirror the repo if this is run on the master branch of the main repoitory.
mirror_from="https://github.com/asyml/texar.git"
mirror_to="git@github.com:petuum/texar.git"

if [[ ${TRAVIS_BRANCH} == "master" ]] && [[ ${TRAVIS_PULL_REQUEST} == "false" ]] && [[ ${TRAVIS_REPO_SLUG} == "asyml/texar" ]]; then
  echo "Mirroring from "${mirror_from} " to "${mirror_to}

  openssl version
  # Decrypt the private key.
  openssl aes-256-cbc -k "$travis_key_password" -md sha256 -d -a -in travis_key.enc -out ./travis_key

  if [[ -s ./travis_key && $? == 0 ]]
  then
	chmod 400 ./travis_key
  	echo "Host github.com" > ~/.ssh/config
  	echo "  IdentityFile $(pwd)/travis_key" >> ~/.ssh/config

  	git clone --bare ${mirror_from} bare
  	if cd bare; then
    	git push --mirror ${mirror_to}
  	else
   		echo "Cannot cd into forte_bare, clone may be unsuccessful."
  	fi
  else
	echo "Decrypted key is empty, will not try to push."
  fi
   
else
  echo "Not on the master of the main repo, will not do mirror."
  echo "Travis Branch: ""${TRAVIS_BRANCH}"
  echo "Travis PR: ""${TRAVIS_PULL_REQUEST}"
  echo "Travis REPO: ""${TRAVIS_REPO_SLUG}"
fi

# More after_success commands can be added below.
