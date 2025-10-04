#!/bin/bash

test_spec="conv_stride_test -H 6 -W 6 -kH 3 -kW 3 -sW 1 -sH 1"
echo "Test spec: $test_spec"

sH=$(echo "$test_spec" | grep -o '\-sH [0-9]\+' | awk '{print $2}')
sW=$(echo "$test_spec" | grep -o '\-sW [0-9]\+' | awk '{print $2}')

echo "Parsed sH: [$sH]"
echo "Parsed sW: [$sW]"

if [ -z "$sH" ]; then
    echo "ERROR: sH is empty!"
fi

if [ -z "$sW" ]; then
    echo "ERROR: sW is empty!"
fi
