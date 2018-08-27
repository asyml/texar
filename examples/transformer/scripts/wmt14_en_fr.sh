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

OUTPUT_DIR="/home/hzt/shr/txtgen/examples/transformer/data/en_fr/"
OUTPUT_DIR_CACHE="${OUTPUT_DIR}/cache"
echo "Writing to ${OUTPUT_DIR_CACHE}. To change this, set the OUTPUT_DIR environment variable."

OUTPUT_DIR_DATA="${OUTPUT_DIR}/data"
mkdir -p $OUTPUT_DIR_DATA

if [ ! -f ${OUTPUT_DIR_DATA}/europarl-v7-fr-en.tgz ]; then
    echo "Downloading Europarl v7. This may take a while..."
    curl -o ${OUTPUT_DIR_DATA}/europarl-v7-fr-en.tgz \
        http://www.statmt.org/europarl/v7/fr-en.tgz
else:
    echo "${OUTPUT_DIR_DATA}/europarl-v7-fr-en.tgz already exists."
fi

if [ ! -f ${OUTPUT_DIR_DATA}/common-crawl.tgz]; then
    echo "Downloading Common Crawl corpus. This may take a while..."
    curl -o ${OUTPUT_DIR_DATA}/common-crawl.tgz \
    http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
else:
    echo "${OUTPUT_DIR_DATA}/common-crawl.tgz already exists."
fi

if [ ! -f ${OUTPUT_DIR_DATA}/nc-v9.tgz]; then
    echo "Downloading News Commentary v11. This may take a while..."
    curl -o ${OUTPUT_DIR_DATA}/nc-v9.tgz \
        http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz
else:
    echo "${OUTPUT_DIR_DATA}/nc-v9.tgz already exists"
fi

if [ ! -f ${OUTPUT_DIR_DATA}/giga-fren.tar]; then
    echo "Downloading giga-fren. This may take a while"
    curl -o ${OUTPUT_DIR_DATA}/giga-fren.tar \
        http://www.statmt.org/wmt10/training-giga-fren.tar
else:
    echo "${OUTPUT_DIR_DATA}/giga-fren.tar already exists"
fi

if [ ! -f ${OUTPUT_DIR_DATA}/un.tgz]; then
    echo "Downloading trainning_parallel_un data. This may take a while"
    curl -o ${OUTPUT_DIR_DATA}/un.tgz \
        http://www.statmt.org/wmt13/training-parallel-un.tgz
else:
    echo "${OUTPUT_DIR_DATA}/un.tgz already exists"


if [ ! -f ${OUTPUT_DIR_DATA}/dev.tgz]; then
    echo "Downloading dev/test sets"
    curl -o ${OUTPUT_DIR_DATA}/dev.tgz \
        http://data.statmt.org/wmt16/translation-task/dev.tgz
else:
    echo "${OUTPUT_DIR_DATA}/dev.tgz already exists"
fi

if [ ! -f ${OUTPUT_DIR_DATA}/test.tgz]; then
    curl -o ${OUTPUT_DIR_DATA}/test.tgz \
        http://data.statmt.org/wmt16/translation-task/test.tgz
else:
    echo "${OUTPUT_DIR_DATA}/test.tgz already exists"
fi

# Extract everything
echo "Extracting all files..."
mkdir -p "${OUTPUT_DIR_DATA}/europarl-v7-de-en"
tar -xvzf "${OUTPUT_DIR_DATA}/europarl-v7-de-en.tgz" -C "${OUTPUT_DIR_DATA}/europarl-v7-de-en"
mkdir -p "${OUTPUT_DIR_DATA}/common-crawl"
tar -xvzf "${OUTPUT_DIR_DATA}/common-crawl.tgz" -C "${OUTPUT_DIR_DATA}/common-crawl"
mkdir -p "${OUTPUT_DIR_DATA}/nc-v11"
tar -xvzf "${OUTPUT_DIR_DATA}/nc-v11.tgz" -C "${OUTPUT_DIR_DATA}/nc-v11"
mkdir -p "${OUTPUT_DIR_DATA}/dev"
tar -xvzf "${OUTPUT_DIR_DATA}/dev.tgz" -C "${OUTPUT_DIR_DATA}/dev"
mkdir -p "${OUTPUT_DIR_DATA}/test"
tar -xvzf "${OUTPUT_DIR_DATA}/test.tgz" -C "${OUTPUT_DIR_DATA}/test"

# Concatenate Training data
cat "${OUTPUT_DIR_DATA}/europarl-v7-de-en/europarl-v7.de-en.en" \
  "${OUTPUT_DIR_DATA}/common-crawl/commoncrawl.de-en.en" \
  "${OUTPUT_DIR_DATA}/nc-v11/training-parallel-nc-v11/news-commentary-v11.de-en.en" \
  > "${OUTPUT_DIR_CACHE}/train.en"
wc -l "${OUTPUT_DIR_CACHE}/train.en"

cat "${OUTPUT_DIR_DATA}/europarl-v7-de-en/europarl-v7.de-en.de" \
  "${OUTPUT_DIR_DATA}/common-crawl/commoncrawl.de-en.de" \
  "${OUTPUT_DIR_DATA}/nc-v11/training-parallel-nc-v11/news-commentary-v11.de-en.de" \
  > "${OUTPUT_DIR_CACHE}/train.de"
wc -l "${OUTPUT_DIR_CACHE}/train.de"

# Clone Moses
if [ ! -d "${OUTPUT_DIR_CACHE}/mosesdecoder" ]; then
  echo "Cloning moses for data processing"
  git clone https://github.com/moses-smt/mosesdecoder.git "${OUTPUT_DIR_CACHE}/mosesdecoder"
fi

${OUTPUT_DIR_CACHE}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newstest2013-deen-src.de.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/newstest2013.de
${OUTPUT_DIR_CACHE}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newstest2013-deen-ref.en.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/newstest2013.en

# Convert SGM files
# Convert newstest2014 data into raw text format
${OUTPUT_DIR_CACHE}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newstest2014-deen-src.de.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/newstest2014.de
${OUTPUT_DIR_CACHE}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newstest2014-deen-ref.en.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/newstest2014.en

# Copy dev/test data to output dir
cp ${OUTPUT_DIR_DATA}/dev/dev/newstest20*.de ${OUTPUT_DIR_CACHE}
cp ${OUTPUT_DIR_DATA}/dev/dev/newstest20*.en ${OUTPUT_DIR_CACHE}
cp ${OUTPUT_DIR_DATA}/test/test/newstest20*.de ${OUTPUT_DIR_CACHE}
cp ${OUTPUT_DIR_DATA}/test/test/newstest20*.en ${OUTPUT_DIR_CACHE}

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

cp ${OUTPUT_DIR_CACHE}/train.tok.en.clean ${OUTPUT_DIR}/train.en
cp ${OUTPUT_DIR_CACHE}/train.tok.de.clean ${OUTPUT_DIR}/train.de
cp ${OUTPUT_DIR_CACHE}/newstest2013.tok.en ${OUTPUT_DIR}/dev.en
cp ${OUTPUT_DIR_CACHE}/newstest2013.tok.de ${OUTPUT_DIR}/dev.de
cp ${OUTPUT_DIR_cACHE}/newstest2014.tok.en ${OUTPUT_DIR}/test.en
cp ${OUTPUT_DIR_CACHE}/newstest2014.tok.de ${OUTPUT_DIR}/test.de
