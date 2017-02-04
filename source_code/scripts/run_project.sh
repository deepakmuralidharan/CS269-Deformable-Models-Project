#!/bin/bash

echo "exporting path"
export PATH=$PATH:/Applications/MATLAB_R2016b.app/bin/

# get initial contour outside the script

for i in {1..5}
do
	# save the patches
	echo "saving patches"
	matlab -r -nodesktop -nodisplay -nosplash 'savePatchesfromContourPoints';

	# get the normal vectors
	echo "getting normal vectors"
	matlab -r -nodesktop -nodisplay -nosplash 'getContourNormals';

	# run testing script to get the direction vectors
	echo "getting direction vectors"
	python convolutional_network_testing.py

	# extend the contour
	echo "extending the contour"
	matlab -r -nodesktop -nodisplay -nosplash 'extendContour';
	
	#rm ../tmp/patches/*
done
