#!/bin/bash

# Script to extract all .tar.xz files in the current directory

# Usage: Run this script in the directory containing the .tar.xz files

# It will extract each file to the current directory

for file in *.tar.xz; do
    if [ -f "$file" ]; then
        echo "Extracting: $file"
        tar -xf "$file"
        if [ $? -eq 0 ]; then
            echo "Extracted successfully: $file"
        else
            echo "Failed to extract: $file"
        fi
    fi
done

echo "Extraction complete."