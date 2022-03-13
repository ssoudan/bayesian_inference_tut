#!/bin/sh

for d in $(ls -d */);
do
	pushd $d
	convert -delay 100  -dispose Background -background White -alpha remove -alpha off *.png animated.gif
#  	convert -delay 100  -dispose Background -background White *.png animated.gif
	popd
done
