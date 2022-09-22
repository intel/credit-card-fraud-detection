# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

#!/bin/bash

# Technology distributions
PKG_ARRAY=(stock intel)
PKG_ELEMENTS=${#PKG_ARRAY[@]}

function success {
echo "Environment $ENV_NAME created"
echo "Activate:"
echo -e "       \$ conda activate $ENV_NAME"
exit 0
}

function failed {
echo "Environment $ENV_NAME creation failed "
exit 1
}

if [ $# -ne 0 ]; then
  PACKAGE=${PKG_ARRAY[0]}
  echo "Warning: No parameters expected. Applying default options..."
else

  echo -e "Select technology distribution: "
  select PACKAGE in "${PKG_ARRAY[@]}"; do
    [ -n "${PACKAGE}" ] && break
  done
fi

ENV_NAME="FraudDetection_"$PACKAGE
echo Creating conda environment $ENV_NAME...
echo Setting up environment with packages : $PACKAGE...

# using stock package
if eval "[[ $PACKAGE = ${PKG_ARRAY[0]} ]]"; then

	conda env create -n $ENV_NAME -f ./env/stock/FraudDetection_stock.yml
	if [[ $? -ne 0 ]] ; then
	  failed
	else
	  success
	fi

fi

# using intel technologies
if eval "[[ $PACKAGE = ${PKG_ARRAY[1]} ]]"; then

	conda env create -n $ENV_NAME -f ./env/intel/FraudDetection_intel.yml
	if [[ $? -ne 0 ]] ; then
	  failed
	else
	  success
	fi

fi

