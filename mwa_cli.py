import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(prog="mwa")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Define 'align' subcommand
    align_parser = subparsers.add_parser("align")
    align_parser.add_argument("model_name", type=str)
    # Use 'nargs' to make language optional with a default
    align_parser.add_argument("language", type=str, nargs='?', default="eng")
    
    align_parser.add_argument("--input_dir", required=True)
    align_parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    if args.command == "align":
        # Mapping the new CLI to your existing align_wav.py logic
        command = [
            sys.executable, "align_wav.py",
            "--wav_input", args.input_dir,
            "--transcript_input", args.input_dir,
            "--language", args.language,
            "--model_name", args.model_name,
            "--output_folder", args.output_dir,
            "--no_graph"
        ]
        
        subprocess.run(command)

if __name__ == "__main__":
    main()