import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Run StarCraft II main.py multiple times")
    parser.add_argument("--num_games", type=int, default=1, help="Number of games to play sequentially")

    # Keep all other arguments for main.py.
    args, unknown_args = parser.parse_known_args()

    for i in range(args.num_games):
        print("=========================================")
        print(f"Starting Game {i + 1}/{args.num_games}")
        print("=========================================")

        cmd = [sys.executable, "main.py", *unknown_args, "--seed", str(i+13)] 

        print(f"Running command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"Game {i + 1} finished successfully.\n")
        except subprocess.CalledProcessError as e:
            print(f"Game {i + 1} exited with error code {e.returncode}.\n")
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Exiting...")
            break


if __name__ == "__main__":
    main()
