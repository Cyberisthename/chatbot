#!/usr/bin/env python3
"""
Script to extract Jarvis-2v-main.zip into the Jarvis-2v-main folder.
"""

import os
import sys
import zipfile
from pathlib import Path


def extract_jarvis_zip():
    """Extract Jarvis-2v-main.zip into Jarvis-2v-main folder."""
    # Define paths
    project_root = Path(__file__).parent
    zip_file = project_root / "Jarvis-2v-main.zip"
    target_dir = project_root / "Jarvis-2v-main"
    
    # Check if zip file exists
    if not zip_file.exists():
        print(f"Error: {zip_file.name} not found in {project_root}")
        print("\nPlease ensure the zip file is present before running this script.")
        print("\nExpected location:")
        print(f"  {zip_file}")
        return 1
    
    # Check file size
    file_size_mb = zip_file.stat().st_size / (1024 * 1024)
    print(f"Found {zip_file.name} ({file_size_mb:.2f} MB)")
    
    # Remove existing target directory if present
    if target_dir.exists():
        print(f"\nWarning: Directory {target_dir.name} already exists")
        print(f"Overwriting existing {target_dir.name} directory...")
        import shutil
        shutil.rmtree(target_dir)
    
    # Create fresh target directory
    target_dir.mkdir(exist_ok=True)
    
    # Extract the zip file
    print(f"\nExtracting {zip_file.name} to {target_dir.name}/...")
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Get list of files
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            
            print(f"Archive contains {total_files} files")
            
            # Extract with progress
            for i, file in enumerate(file_list, 1):
                zip_ref.extract(file, target_dir)
                if i % 10 == 0 or i == total_files:
                    print(f"  Progress: {i}/{total_files} files extracted", end='\r')
            
            print(f"\n\n✓ Successfully extracted {zip_file.name} to {target_dir.name}/")
            
            # List contents
            print("\nExtracted contents:")
            for item in sorted(target_dir.iterdir())[:20]:
                if item.is_dir():
                    print(f"  📁 {item.name}/")
                else:
                    size_kb = item.stat().st_size / 1024
                    print(f"  📄 {item.name} ({size_kb:.1f} KB)")
            
            if len(list(target_dir.iterdir())) > 20:
                print(f"  ... and {len(list(target_dir.iterdir())) - 20} more items")
            
            return 0
            
    except zipfile.BadZipFile:
        print(f"\nError: {zip_file.name} is not a valid zip file or is corrupted")
        return 1
    except Exception as e:
        print(f"\nError during extraction: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(extract_jarvis_zip())
