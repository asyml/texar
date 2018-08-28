#!/bin/sh
# Copied from https://github.com/tensorflow/nmt/blob/master/nmt/scripts/download_iwslt15.sh
#
# Download small-scale IWSLT15 Vietnames to English translation data for NMT
# model training.
#
# Usage:
#   ./download_iwslt15.sh path-to-output-dir
#
# If output directory is not specified, "./iwslt15" will be used as the default
# output directory.

OUT_DIR="${1:-data/en_vi}"
SITE_PREFIX="https://nlp.stanford.edu/projects/nmt/data"

mkdir -v -p $OUT_DIR

# Download iwslt15 small dataset from standford website.
echo "Download training dataset train.en and train.vi."
curl -o "$OUT_DIR/train.en" "$SITE_PREFIX/iwslt15.en-vi/train.en"
curl -o "$OUT_DIR/train.vi" "$SITE_PREFIX/iwslt15.en-vi/train.vi"

echo "Download dev dataset tst2012.en and tst2012.vi."
curl -o "$OUT_DIR/dev.en" "$SITE_PREFIX/iwslt15.en-vi/tst2012.en"
curl -o "$OUT_DIR/dev.vi" "$SITE_PREFIX/iwslt15.en-vi/tst2012.vi"

echo "Download test dataset tst2013.en and tst2013.vi."
curl -o "$OUT_DIR/test.en" "$SITE_PREFIX/iwslt15.en-vi/tst2013.en"
curl -o "$OUT_DIR/test.vi" "$SITE_PREFIX/iwslt15.en-vi/tst2013.vi"
