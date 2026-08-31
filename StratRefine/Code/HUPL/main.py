import argparse
import os
from datetime import datetime

from bots.protoss_bot import Protoss_Bot
from bots.swarmbrain import SwarmBrain
from bots.textstarcraft import TextStarCraft
from llm_config import resolve_base_url
from sc2 import maps
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer


def parse_args():
    parser = argparse.ArgumentParser(description="STR StarCraft II bot")
    parser.add_argument("--save_path", default="tmp")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--LLM_api_text", default="Qwen/Qwen3-8B", help="Remote model name for OpenAI-compatible providers.")
    parser.add_argument("--LLM_model_path", default="models/STR", help="Local Hugging Face model directory or model ID.")
    parser.add_argument("--LLM_api_mode", default=None, help="Compatibility alias used by legacy opponent agents.")
    parser.add_argument("--LLM_api_key", default=os.getenv("LLM_API_KEY"), help="API key for remote providers; may also be set with LLM_API_KEY.")
    parser.add_argument("--LLM_provider", default="local_transformers", choices=["openai", "siliconflow", "modelscope", "local_transformers"])
    parser.add_argument("--LLM_base_url", default=None)

    parser.add_argument("--decision_mode", default="STRlm", choices=["single_llm", "scripted_text", "STRlm"])
    parser.add_argument("--script_path", default=None, help="Local txt/json/jsonl file used in scripted_text mode.")
    parser.add_argument("--script_inline", default=None, help="Inline scripted command text used in scripted_text mode.")
    parser.add_argument("--script_repeat_last", action="store_true", default=True, help="Repeat the last scripted entry after the list is exhausted.")
    parser.add_argument("--no_script_repeat_last", dest="script_repeat_last", action="store_false")

    parser.add_argument("--realtime", action="store_true", default=False)
    parser.add_argument("--mode", default="bot", choices=["bot", "agent"])

    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--enemy_race", default="Protoss", choices=["Protoss", "Zerg", "Terran"])
    parser.add_argument(
        "--difficulty",
        default="VeryHard",
        choices=["VeryEasy", "Easy", "Medium", "MediumHard", "Hard", "Harder", "VeryHard", "CheatVision", "CheatMoney", "CheatInsane"],
    )
    parser.add_argument("--enemy_agent", default="HEP-TextStarCraft", choices=["TextStarCraft", "SwarmBrain", "HEP-TextStarCraft"])
    return parser.parse_args()


def build_our_bot(args):
    return Protoss_Bot(args)


if __name__ == "__main__":
    args = parse_args()
    args.framework_name = "STR"
    args.enemy_race = Race[args.enemy_race]
    args.difficulty = Difficulty[args.difficulty]
    args.current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.LLM_api_mode is None:
        args.LLM_api_mode = args.LLM_api_text
    args.LLM_base_url = resolve_base_url(args.LLM_provider, args.LLM_base_url)

    if args.decision_mode == "scripted_text" and not (args.script_path or args.script_inline):
        raise ValueError("scripted_text mode requires --script_path or --script_inline.")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    temp_replay_folder = os.path.join(current_dir, args.save_path, args.current_time)
    os.makedirs(temp_replay_folder, exist_ok=True)
    temp_replay_path = f"{temp_replay_folder}/{args.current_time}_{args.difficulty}_{args.enemy_race}_temp.SC2Replay"

    our_bot = build_our_bot(args)
    if args.mode == "bot":
        enemy = Computer(args.enemy_race, args.difficulty)
        result = run_game(
            maps.get("Ancient Cistern LE"),
            [Bot(Race.Protoss, our_bot), enemy],
            realtime=args.realtime,
            save_replay_as=temp_replay_path,
            random_seed=args.seed,
        )
        result = str(result).split(".")[1]
        final_replay_path = f"{temp_replay_folder}/{args.current_time}_{args.difficulty}_{args.enemy_race}_{result}.SC2Replay"
        os.rename(temp_replay_path, final_replay_path)
    else:
        if args.enemy_agent == "TextStarCraft":
            enemy = Bot(Race.Protoss, TextStarCraft(args))
        elif args.enemy_agent == "SwarmBrain":
            enemy = Bot(Race.Zerg, SwarmBrain(args))
        else:
            enemy = Bot(Race.Protoss, TextStarCraft(args, hep=True))
        run_game(
            maps.get("Ancient Cistern LE"),
            [Bot(Race.Protoss, our_bot), enemy],
            realtime=args.realtime,
            save_replay_as=temp_replay_path,
            random_seed=args.seed,
        )
