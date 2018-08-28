#!/usr/bin/env bash

# This code was adapted from Tensorflow NMT toolkit on 03/24/2018.
# URL: https://raw.githubusercontent.com/tensorflow/nmt/master/nmt/scripts/wmt16_en_de.sh

# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
OUTPUT_DIR="data/en_de/"
DOWNLOADED_DATA_DIR="data/en_de_temp/"
OUTPUT_DIR_CACHE="${DOWNLOADED_DATA_DIR}/cache"
echo "Writing to ${OUTPUT_DIR_CACHE}. To change this, set the OUTPUT_DIR_CACHE environment variable."
mkdir -p $DOWNLOADED_DATA_DIR
mkdir -p ${OUTPUT_DIR}
if [ ! -f ${DOWNLOADED_DATA_DIR}/europarl-v7-de-en.tgz ]; then
    echo "Downloading Europarl v7. This may take a while..."
    curl -o ${DOWNLOADED_DATA_DIR}/europarl-v7-de-en.tgz \
        http://www.statmt.org/europarl/v7/de-en.tgz
else
    echo "${DOWNLOADED_DATA_DIR}/europarl-v7-de-en.tgz already exists."
fi

if [ ! -f ${DOWNLOADED_DATA_DIR}/common-crawl.tgz ]; then
    echo "Downloading Common Crawl corpus. This may take a while..."
    curl -o ${DOWNLOADED_DATA_DIR}/common-crawl.tgz \
    http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
else
    echo "${DOWNLOADED_DATA_DIR}/common-crawl.tgz already exists."
fi

if [ ! -f ${DOWNLOADED_DATA_DIR}/nc-v11.tgz ]; then
    echo "Downloading News Commentary v11. This may take a while..."
    curl -o ${DOWNLOADED_DATA_DIR}/nc-v11.tgz \
        http://data.statmt.org/wmt16/translation-task/training-parallel-nc-v11.tgz
else
    echo "${DOWNLOADED_DATA_DIR}/nc-v11.tgz already exists"
fi

if [ ! -f ${DOWNLOADED_DATA_DIR}/dev.tgz ]; then
    echo "Downloading dev/test sets"
    curl -o ${DOWNLOADED_DATA_DIR}/dev.tgz \
        http://data.statmt.org/wmt16/translation-task/dev.tgz
else
    echo "${DOWNLOADED_DATA_DIR}/dev.tgz already exists"
fi

if [ ! -f ${DOWNLOADED_DATA_DIR}/test.tgz ]; then
    curl -o ${DOWNLOADED_DATA_DIR}/test.tgz \
        http://data.statmt.org/wmt16/translation-task/test.tgz
else
    echo "${DOWNLOADED_DATA_DIR}/test.tgz already exists"
fi

# Extract everything
echo "Extracting all files..."
if  [ ! -d ${DOWNLOADED_DATA_DIR}/europarl-v7-de-en ]; then
    mkdir -p "${DOWNLOADED_DATA_DIR}/europarl-v7-de-en"
    tar -xvzf "${DOWNLOADED_DATA_DIR}/europarl-v7-de-en.tgz" -C "${DOWNLOADED_DATA_DIR}/europarl-v7-de-en"
    mkdir -p "${DOWNLOADED_DATA_DIR}/common-crawl"
    tar -xvzf "${DOWNLOADED_DATA_DIR}/common-crawl.tgz" -C "${DOWNLOADED_DATA_DIR}/common-crawl"
    mkdir -p "${DOWNLOADED_DATA_DIR}/nc-v11"
    tar -xvzf "${DOWNLOADED_DATA_DIR}/nc-v11.tgz" -C "${DOWNLOADED_DATA_DIR}/nc-v11"
    mkdir -p "${DOWNLOADED_DATA_DIR}/dev"
    tar -xvzf "${DOWNLOADED_DATA_DIR}/dev.tgz" -C "${DOWNLOADED_DATA_DIR}/dev"
    mkdir -p "${DOWNLOADED_DATA_DIR}/test"
    tar -xvzf "${DOWNLOADED_DATA_DIR}/test.tgz" -C "${DOWNLOADED_DATA_DIR}/test"
else
    echo "the tar files have been unzipped"
fi

# Concatenate Training data
wc -l ${DOWNLOADED_DATA_DIR}/europarl-v7-de-en/europarl-v7.de-en.en
wc -l ${DOWNLOADED_DATA_DIR}/common-crawl/commoncrawl.de-en.en
wc -l ${DOWNLOADED_DATA_DIR}/nc-v11/training-parallel-nc-v11/news-commentary-v11.de-en.en

cat "${DOWNLOADED_DATA_DIR}/europarl-v7-de-en/europarl-v7.de-en.en" \
  "${DOWNLOADED_DATA_DIR}/common-crawl/commoncrawl.de-en.en" \
  "${DOWNLOADED_DATA_DIR}/nc-v11/training-parallel-nc-v11/news-commentary-v11.de-en.en" \
  > "${OUTPUT_DIR_CACHE}/train.en" &&\
wc -l "${OUTPUT_DIR_CACHE}/train.en"

cat "${DOWNLOADED_DATA_DIR}/europarl-v7-de-en/europarl-v7.de-en.de" \
  "${DOWNLOADED_DATA_DIR}/common-crawl/commoncrawl.de-en.de" \
  "${DOWNLOADED_DATA_DIR}/nc-v11/training-parallel-nc-v11/news-commentary-v11.de-en.de" \
  > "${OUTPUT_DIR_CACHE}/train.de" &&\
wc -l "${OUTPUT_DIR_CACHE}/train.de"

# Clone Moses
if [ ! -d "${OUTPUT_DIR_CACHE}/mosesdecoder" ]; then
  echo "Cloning moses for data processing"
  git clone https://github.com/moses-smt/mosesdecoder.git "${OUTPUT_DIR_CACHE}/mosesdecoder"
fi

${OUTPUT_DIR_CACHE}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${DOWNLOADED_DATA_DIR}/dev/dev/newstest2014-deen-src.de.sgm \
  > ${DOWNLOADED_DATA_DIR}/dev/dev/newstest2014.de
${OUTPUT_DIR_CACHE}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${DOWNLOADED_DATA_DIR}/dev/dev/newstest2014-deen-ref.en.sgm \
  > ${DOWNLOADED_DATA_DIR}/dev/dev/newstest2014.en

# Copy dev/test data to output dir
cp ${DOWNLOADED_DATA_DIR}/dev/dev/newstest20*.de ${OUTPUT_DIR_CACHE}
cp ${DOWNLOADED_DATA_DIR}/dev/dev/newstest20*.en ${OUTPUT_DIR_CACHE}

# Tokenize data
for f in ${OUTPUT_DIR_CACHE}/*.de; do
  echo "Tokenizing $f..."
  ${OUTPUT_DIR_CACHE}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l de -threads 8 < $f > ${f%.*}.tok.de
done

for f in ${OUTPUT_DIR_CACHE}/*.en; do
  echo "Tokenizing $f..."
  ${OUTPUT_DIR_CACHE}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l en -threads 8 < $f > ${f%.*}.tok.en
done

# Clean train corpora
for f in ${OUTPUT_DIR_CACHE}/train.tok.en; do
  fbase=${f%.*}
  echo "Cleaning ${fbase}..."
  ${OUTPUT_DIR_CACHE}/mosesdecoder/scripts/training/clean-corpus-n.perl $fbase de en "${fbase}.clean" 1 80
done

cp ${OUTPUT_DIR_CACHE}/train.tok.clean.en ${OUTPUT_DIR}/train.en
cp ${OUTPUT_DIR_CACHE}/train.tok.clean.de ${OUTPUT_DIR}/train.de
cp ${OUTPUT_DIR_CACHE}/newstest2013.tok.en ${OUTPUT_DIR}/dev.en
cp ${OUTPUT_DIR_CACHE}/newstest2013.tok.de ${OUTPUT_DIR}/dev.de
cp ${OUTPUT_DIR_CACHE}/newstest2014.tok.en ${OUTPUT_DIR}/test.en
cp ${OUTPUT_DIR_CACHE}/newstest2014.tok.de ${OUTPUT_DIR}/test.de
