#!/bin/bash

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
mkdir -p $SITE_PACKAGES/hrca
cp $SRC_DIR/hrca/__init__.py $SITE_PACKAGES/hrca
cp $SRC_DIR/hrca/hrca.py $SITE_PACKAGES/hrca
cp $SRC_DIR/hrca/dataload.py $SITE_PACKAGES/hrca
cp $SRC_DIR/hrca/utils.py $SITE_PACKAGES/hrca
cp $SRC_DIR/hrca/model.py $SITE_PACKAGES/hrca