#!/usr/bin/env bash

set -euo pipefail  # safer bash: stop on error, unset vars, etc.

# --- Function to download and extract ---
download_and_extract() {
  local folder_name="$1"
  shift
  local download_links=("$@")

  mkdir -p "$folder_name"

  # Download all files
  for link in "${download_links[@]}"; do
    local filename
    filename=$(basename "$link")
    echo "Downloading $filename..."
    wget -q --show-progress -O "$filename" "$link"
  done

  # Check if any .001 split files exist
  if compgen -G "*.001" > /dev/null; then
    echo "Extracting split archive into $folder_name..."
    first_part=$(ls *.001 | head -n 1)
    7z x "$first_part" -o"$folder_name"
    rm -f *.00?
  else
    # Extract regular .zip and .7z files
    for archive in *.7z *.zip; do
      [[ -e "$archive" ]] || continue  # skip if no match
      echo "Extracting $archive into $folder_name..."
      case "$archive" in
        *.7z) 7z x "$archive" -o"$folder_name" ;;
        *.zip) unzip -q "$archive" -d "$folder_name" ;;
      esac
      rm -f "$archive"
    done
  fi
}

# --- Main Downloads ---

echo "Downloading and extracting giantinplayground..."
download_and_extract "giantinplayground" \
  "https://files.catbox.moe/emijbl.7z" \
  "https://files.catbox.moe/v16bul.7z" \
  "https://files.catbox.moe/0cszzp.7z" \
  "https://files.catbox.moe/yv3c7g.7z" \
  "https://files.catbox.moe/mdcno7.7z" \
  "https://files.catbox.moe/8em6zp.7z"

echo "Downloading and extracting elliquiy..."
download_and_extract "elliquiy" \
  "https://files.catbox.moe/qyhzbc.zip"

echo "Downloading and extracting roleplayerguild..."
download_and_extract "roleplayerguild" \
  "https://files.catbox.moe/uuub3g.7z" \
  "https://files.catbox.moe/luxmdg.7z" \
  "https://files.catbox.moe/o4bo59.7z" \
  "https://files.catbox.moe/wpceng.7z" \
  "https://files.catbox.moe/exl3ia.7z" \
  "https://files.catbox.moe/q9mys3.7z" \
  "https://files.catbox.moe/nhhnj0.7z" \
  "https://files.catbox.moe/gqilvh.7z"

echo "Downloading and extracting roleplay-by-post..."
download_and_extract "roleplay-by-post" \
  "https://files.catbox.moe/xvgmvr.001" \
  "https://files.catbox.moe/sxf7nd.002"

echo "Downloading FIREBALL from Hugging Face..."
git clone https://huggingface.co/datasets/lara-martin/FIREBALL
echo "All done!"

