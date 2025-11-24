# Jarvis-2v-main Extraction Guide

This guide explains how to extract the `Jarvis-2v-main.zip` archive into the appropriate folder structure.

## Quick Start

### Option 1: Using Python Script (Recommended)

```bash
# Make sure you have Jarvis-2v-main.zip in the project root
python extract_jarvis_2v.py
```

This script will:
- ✅ Verify the zip file exists
- ✅ Display file size and extraction progress
- ✅ Extract all contents to `Jarvis-2v-main/` folder
- ✅ Show a summary of extracted files

### Option 2: Using Bash Script

```bash
# Make sure you have Jarvis-2v-main.zip in the project root
./extract_jarvis_2v.sh
```

### Option 3: Manual Extraction

```bash
# Using unzip command
unzip Jarvis-2v-main.zip -d Jarvis-2v-main/

# Or using Python's zipfile module
python -m zipfile -e Jarvis-2v-main.zip Jarvis-2v-main/
```

## Prerequisites

Before extracting, ensure:

1. **The zip file exists**: Place `Jarvis-2v-main.zip` in the project root directory
   ```
   /home/engine/project/Jarvis-2v-main.zip
   ```

2. **Sufficient disk space**: Check available space before extraction
   ```bash
   df -h .
   ```

3. **Required tools**: 
   - For bash script: `unzip` command
   - For Python script: Python 3.6+ (already included in project)

## Directory Structure

After extraction, the structure will look like:

```
project/
├── Jarvis-2v-main.zip          # Original archive (optional to keep)
├── Jarvis-2v-main/             # Extracted contents
│   ├── (archive contents)
│   └── ...
├── extract_jarvis_2v.py        # Python extraction script
└── extract_jarvis_2v.sh        # Bash extraction script
```

## Troubleshooting

### Zip file not found

**Problem**: `Error: Jarvis-2v-main.zip not found`

**Solution**: Make sure the zip file is in the project root directory:
```bash
ls -lh Jarvis-2v-main.zip
```

### Permission denied

**Problem**: Permission error when running scripts

**Solution**: Make the scripts executable:
```bash
chmod +x extract_jarvis_2v.py
chmod +x extract_jarvis_2v.sh
```

### Corrupted archive

**Problem**: `Error: not a valid zip file or is corrupted`

**Solution**: Verify the zip file integrity:
```bash
unzip -t Jarvis-2v-main.zip
```

If corrupted, re-download the archive.

### Disk space

**Problem**: No space left on device

**Solution**: Check and free up disk space:
```bash
# Check available space
df -h .

# Clean up unnecessary files
# (use with caution)
```

## Cleanup

After verifying the extraction:

```bash
# Remove the README placeholder from the target directory
rm Jarvis-2v-main/README.md

# Optionally remove the zip file to save space
# rm Jarvis-2v-main.zip
```

## Git Ignore

The `.gitignore` file is already configured to:
- ✅ Ignore `Jarvis-2v-main.zip`
- ✅ Ignore contents of `Jarvis-2v-main/` folder
- ✅ Keep the placeholder `README.md` in version control

This ensures the extracted contents don't bloat the repository.

## Notes

- The extraction scripts will warn you if the target directory already exists
- Both scripts support overwriting existing directories
- The Python script provides more detailed progress information
- Extracted contents are automatically ignored by git

## Support

If you encounter issues not covered here, please:
1. Check the error message carefully
2. Verify file permissions and disk space
3. Try the alternative extraction method
4. Check the project README for additional information
