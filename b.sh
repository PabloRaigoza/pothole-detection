#!/bin/bash

# Specify the directory containing your images
directory="FilterImages"

# Change to the directory
cd "$directory" || exit

# Initialize a counter
counter=1

# Rename the files
for filename in *.JPG; do
  new_filename="$counter.jpg"
  mv "$filename" "$new_filename"
  ((counter++))
done

# Check if no .jpg files were found
if [ $counter -eq 1 ]; then
  echo "No .jpg files found in the specified directory."
fi
