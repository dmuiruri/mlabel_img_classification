#!/bin/bash

# This script is used to evaluate the number of items in a given class
# as listed in the annotations folder. The annotations folder
# container a file named by class and then file contains the id of the
# file.

# This script should be used as follows: ./count_lines.sh annotations

# Check if the directory argument is supplied
if [ $# -eq 0 ]
then
    echo "No directory name is supplied"
    exit 1
fi

# Store the directory name
dir=$1

# Get teh list of files in the current dir
files=$(ls)

# Loop through and count the lines in each file
for file in "$dir"/*
do
    if [ -f "$file" ] 
    then
	line_count=$(wc -l < "$file")
	echo "File $file has $line_count lines"
    fi
done
