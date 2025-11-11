#!/bin/bash

BASE_DIR="your/desired/path"  # Set your desired base directory
C4_DIR="$BASE_DIR/c4"
CACHE_DIR="$BASE_DIR/cache"

mkdir -p "$C4_DIR" "$CACHE_DIR"

DOWNLOAD_COMMAND="export HF_ENDPOINT=https://hf-mirror.com && \
                  huggingface-cli download \
                    --repo-type dataset \
                    --resume-download \
                    allenai/c4 \
                    --include 'en/*' \
                    --local-dir '$C4_DIR' \
                    --cache-dir '$CACHE_DIR' \
                    --local-dir-use-symlinks False"

check_download_files_count() {
    local required=1032
    local count

    count=$(find "$C4_DIR/en" -type f | wc -l)

    if [ "$count" -eq "$required" ]; then
        return 0
    else
        echo "Current file count: $count (expected: $required)"
        return 1
    fi
}

while true; do
    echo "Starting download attempt..."
    eval $DOWNLOAD_COMMAND
    result=$?

    if [ $result -eq 0 ]; then
        if check_download_files_count; then
            echo "✅ Download completed successfully with all 1032 files."
            break
        else
            echo "⚠️ Download finished but file count is incorrect. Retrying in 10 seconds..."
            sleep 10
        fi
    else
        echo "❌ Download failed with error code $result. Retrying in 10 seconds..."
        sleep 10
    fi
done

echo "✅ Script finished."