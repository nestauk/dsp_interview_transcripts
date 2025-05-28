#!/bin/bash

# Get the folder where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define outputs directory relative to the script location
OUTPUTS_DIR="$SCRIPT_DIR/../outputs"

# Delete all contents of the outputs folder
rm -rf "$OUTPUTS_DIR"/*

# Optional: log cleanup time
echo "$(date): Cleaned outputs folder" >> "$SCRIPT_DIR/cleanup.log"
