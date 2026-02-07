#!/usr/bin/env python3

import argparse
import sys
from huggingface_hub import HfApi, hf_hub_download, RepositoryNotFoundError, RevisionNotFoundError

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="Download a specific quantization of a model from Hugging Face."
    )
    
    # Add arguments
    parser.add_argument(
        "repo_id", 
        help="The Hugging Face repository ID (e.g., 'TheBloke/Llama-2-7B-GGUF')"
    )
    parser.add_argument(
        "quantization", 
        help="The quantization string to search for in filenames (e.g., 'Q4_K_M', 'q4_k_m')"
    )
    parser.add_argument(
        "--output", "-o",
        default=".",
        help="Local directory to save the model file (default: current directory)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Initialize Hugging Face API
    api = HfApi()

    try:
        print(f"Searching repository: {args.repo_id}")
        print(f"Looking for quantization: {args.quantization}")
        print("-" * 50)

        # 1. List all files in the repository
        repo_files = api.list_repo_files(args.repo_id)

        # 2. Filter files containing the quantization string (case-insensitive)
        search_term = args.quantization.lower()
        matching_files = [f for f in repo_files if search_term in f.lower()]

        if not matching_files:
            print(f"\n[!] Error: No files found matching '{args.quantization}'.")
            print("\nAvailable files in repository (showing first 20):")
            for f in repo_files[:20]:
                print(f" - {f}")
            sys.exit(1)

        # 3. Select the first match
        target_file = matching_files[0]
        
        # Warn if multiple files matched
        if len(matching_files) > 1:
            print(f"[!] Multiple files matched. Downloading the first one found: {target_file}")
        else:
            print(f"[+] Match found: {target_file}")

        # 4. Download the file
        print(f"[*] Downloading to: {args.output}/")
        file_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=target_file,
            local_dir=args.output,
            local_dir_use_symlinks=False
        )

        print("\n[+] Download successful!")
        print(f"    Saved to: {file_path}")

    except RepositoryNotFoundError:
        print(f"\n[!] Error: Repository '{args.repo_id}' not found or is private.")
        print("    If it is private, make sure you are logged in via 'huggingface-cli login'.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()