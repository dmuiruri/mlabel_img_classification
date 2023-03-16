#!/bin/bash

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
