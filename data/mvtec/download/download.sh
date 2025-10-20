#!/bin/bash

# Script to download MVTec dataset files from the URLs listed in text file

# Usage: Run this script in the same directory as the 'text' file

# It will download all .tar.xz files to the current directory

while IFS= read -r url; do
    if [ -n "$url" ]; then
        echo "Downloading: $url"
        wget -c "$url"  # -c for resume if interrupted
        if [ $? -ne 0 ]; then
            echo "Failed to download: $url"
        fi
    fi
done < text

echo "Download complete."