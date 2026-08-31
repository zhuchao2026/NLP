from prompts.str_prompt import str_prompt


class Prompt:
    def __init__(self, race):
        self.race = race

    def generate_prompts(self):
        return str_prompt(self.race)

    def generate_vision_prompts(self, target):
        return f"This is a StarCraft II real game image. Where can I build a {target}? Respond strictly with a single (x, y) coordinate, no explanations, no additional text."
