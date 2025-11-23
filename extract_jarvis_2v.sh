#!/bin/bash
# Script to extract Jarvis-2v-main.zip into a dedicated folder

# Set variables
ZIP_FILE="Jarvis-2v-main.zip"
TARGET_DIR="Jarvis-2v-main"

# Check if zip file exists
if [ ! -f "$ZIP_FILE" ]; then
    echo "Error: $ZIP_FILE not found in current directory"
    echo "Please ensure the zip file is present before running this script"
    exit 1
fi

# Check if unzip is installed
if ! command -v unzip &> /dev/null; then
    echo "Error: unzip command not found"
    echo "Please install unzip: sudo apt-get install unzip"
    exit 1
fi

# Create target directory if it already exists
if [ -d "$TARGET_DIR" ]; then
    echo "Warning: Directory $TARGET_DIR already exists"
    echo "Overwriting existing directory..."
    rm -rf "$TARGET_DIR"
fi

# Extract the zip file
echo "Extracting $ZIP_FILE to $TARGET_DIR..."
unzip -q "$ZIP_FILE" -d "$TARGET_DIR"

if [ $? -eq 0 ]; then
    echo "Successfully extracted $ZIP_FILE to $TARGET_DIR/"
    echo "Contents:"
    ls -lh "$TARGET_DIR/"
else
    echo "Error: Failed to extract $ZIP_FILE"
    exit 1
fi
