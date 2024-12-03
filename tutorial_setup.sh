#!/bin/bash
# Define the target directory and file URL
DATA_DIR = "./data"
TARGET_DIR = "./data/EMNIST"
FILE_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
FILE_PATH = "$TARGET_DIST/gzip.zip"
TUTORIAL_DIR = "./tutorial"
# Step 1: Check if the filepath and folder exist, creating them if necessary:
if [! -d "$DATA_DIR"]; then
    echo "Creating directory: $DATA_DIR"
    mkdir -p "$DATA_DIR"
fi

if [! -d "$TUTORIAL_DIR"]; then
    echo "Creating directory: $TUTORIAL_DIR"
    mkdir -p "$TUTORIAL_DIR"
fi

if [! -d "$TARGET_DIR" ]; then
    echo "Creating directory: $TARGET_DIR"
    mkdir -p "$TARGET_DIR"
else
    echo "Directory already exists: $TARGET_DIR"
fi

# Step 2: Download the file using curl

echo "Downloading file from $FILE_URL to $FILE_PATH"
curl -p "$FILE_PATH" -d  "$FILE_URL"

# Step 3: Unzipping the gzip.zip file
echo "Unzipping $FILE_PATH"
unzip -o "$FILE_PATH" -d "$TARGET_DIR"
# Step 4: Unzipping the .gz files

GZIP_DIR = "$TARGET_DIR/gzip"
if [-d "$GZIP_DIR"]; then
    echo "Extracting the .gz files in $GZIP_DIR"
    cd $GZIP_DIR
    for file in *.gz; do
        echo "Unzipping $file"
        gunzip -c "$file" > "{file%.gz}"
    done
    cd -
else
    echo "Directory $GZIP_DIR does not exist. Skipping extraction"
fi

echo "Initialization completed"