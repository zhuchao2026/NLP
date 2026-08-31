import json
import os
import random
from collections import Counter, deque

import constants
import utils
from STR.decision import PromptBundle, build_decision_backend
from prompt import Prompt
from sc2.bot_ai import BotAI
from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId


class STR(BotAI):
    FAILURE_POLICIES = {
        "not_ready": "retry_later",
        "not_afford": "retry_later",
        "supply_pending": "retry_last",
        "prereq_pending": "skip_and_defer",
        "producer_busy": "retry_later",
        "not_ability": "retry_later",
        "prereq_queued": "skip_and_defer",
        "nexus_fail": "retry_later",
        "gas_full": "drop_action",
        "gas_blocked": "retry_later",
        "not_exist": "blocked_by_prereq",
        "fail": "retry_last",
        "no_space": "replan_now",
        "many": "drop_action",
        "already_on": "drop_action",
        "supply_200": "drop_action",
        "fail_train": "drop_action",
        "no_templar": "drop_action",
        "pending_nexus": "drop_action",
        "already_done": "drop_action",
    }

    def __init__(self, args):
        self.own_race = Race.Protoss
        self.enemy_race = args.enemy_race

        self.action = utils.ActionDescriptions(self.own_race)
        text_prompt = Prompt(self.own_race).generate_prompts()
        prompt_bundle = PromptBundle(
            system_prompt=text_prompt["system"],
            example_input=text_prompt.get("input", ""),
            example_output=text_prompt.get("output", ""),
        )
        self.action_extractor = utils.ActionExtractor(constants.ACTION_DICT[self.own_race], self.own_race)
        self.game_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.save_path, args.current_time)
        os.makedirs(self.game_folder, exist_ok=True)
        self.decision_backend = build_decision_backend(args, prompt_bundle, self.game_folder)
        self._save_framework_metadata(args)

        self.advice = None
        self.under_attack = False
        self.action_queue = deque()
        self.deferred_action_queue = deque()
        self.scout_period = 4000
        self.refresh_period = 2500

    def _save_framework_metadata(self, args):
        metadata = {
            "framework": "STR",
            "decision_mode": args.decision_mode,
            "prompt": "STR",
            "llm_model": getattr(args, "LLM_api_text", None),
            "llm_model_path": getattr(args, "LLM_model_path", None),
            "llm_provider": getattr(args, "LLM_provider", None),
            "llm_base_url": getattr(args, "LLM_base_url", None),
        }
        with open(os.path.join(self.game_folder, "framework.json"), "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

    def move_to_nexus(self, units):
        nexus = min(self.townhalls, key=lambda base: base.distance_to(self.enemy_start_locations[0]))
        ramp = min(constants.MAP_RAMPS, key=lambda ramp: ramp.distance_to(nexus))
        for unit in units:
            if unit.distance_to(ramp) > 7:
                unit.move(ramp)

    def scout(self):
        point_names = ["H", "J", "K", "L", "O"] if self.start_position == "C" else ["B", "E", "F", "G", "I"]
        points = [constants.MAP_POINTS[name] for name in point_names]
        if self.units(self.scout_unit):
            unit = random.choice(self.units(self.scout_unit))
            patrol_points = random.sample(points, 3) + [self.enemy_start_locations[0], self.start_location]
            for point in patrol_points:
                unit.patrol(point, queue=True)

    @staticmethod
    def _is_enemy_observation_candidate(unit):
        ignored = {"Probe", "Observer", "Drone", "Overlord", "Overseer", "SCV"}
        return "Changeling" not in unit.name and unit.name not in ignored

    def _is_visible_enemy_actively_attacking(self, unit, base_targets):
        if not self._is_enemy_observation_candidate(unit) or not unit.can_attack:
            return False
        if not base_targets:
            return False
        base_target_tags = {target.tag for target in base_targets}
        engaged_own_target = unit.engaged_target_tag in base_target_tags if unit.engaged_target_tag else False
        is_attacking = unit.is_attacking or unit.weapon_cooldown > 0 or engaged_own_target
        if not is_attacking:
            return False
        return any(unit.distance_to(target) <= self.warning_range for target in base_targets)

    def _get_attacking_enemy_summary(self):
        base_targets = list(self.structures)
        attacking_enemy = [
            unit.name for unit in self.enemy_units if self._is_visible_enemy_actively_attacking(unit, base_targets)
        ]
        return dict(Counter(attacking_enemy)) if attacking_enemy else {}

    def _count_ready_and_in_progress_gas_buildings(self):
        return sum(1 for building in self.gas_buildings if int(building.build_progress) and building.vespene_contents)

    def _get_open_geysers(self):
        open_geysers = []
        seen_geysers = set()
        for townhall in self.townhalls:
            for geyser in self.vespene_geyser.closer_than(10, townhall):
                if geyser.tag in seen_geysers:
                    continue
                seen_geysers.add(geyser.tag)
                if self.gas_buildings.closer_than(1, geyser).exists:
                    continue
                open_geysers.append(geyser)
        return open_geysers

    def _rebalance_gas_workers_per_base(self):
        if not self.workers or not self.townhalls.ready or not self.gas_buildings.ready:
            return

        seen_gas_tags = set()
        for townhall in self.townhalls.ready:
            gas_buildings = [
                gas for gas in self.gas_buildings.ready.closer_than(10, townhall) if gas.tag not in seen_gas_tags
            ]
            if not gas_buildings:
                continue
            seen_gas_tags.update(gas.tag for gas in gas_buildings)

            local_mineral_tags = {
                mineral.tag for mineral in self.mineral_field if mineral.distance_to(townhall) <= 8
            }
            mineral_workers = self.workers.filter(
                lambda worker: worker.order_target in local_mineral_tags
                or (worker.is_carrying_minerals and worker.order_target == townhall.tag)
            )
            mineral_worker_count = mineral_workers.amount
            gas_worker_count = sum(gas.assigned_harvesters for gas in gas_buildings)
            if gas_worker_count * 4 >= mineral_worker_count:
                continue

            gas_slots = []
            for gas in gas_buildings:
                gas_slots.extend([gas] * max(0, 3 - gas.assigned_harvesters))
            if not gas_slots:
                continue

            needed_workers = min((mineral_worker_count - gas_worker_count * 4 + 4) // 5, len(gas_slots))
            for worker in mineral_workers[:needed_workers]:
                target_gas = min(gas_slots, key=lambda gas: gas.distance_to(worker))
                worker.gather(target_gas)
                gas_slots.remove(target_gas)

    async def update_cycle(self, observation):
        if self.enemy_units:
            await self.unit_attack(self.units.of_type(constants.UNITS[self.own_race]))
        if self.iteration > self.next_scout:
            self.next_scout = self.iteration + self.scout_period
            self.scout()
        self.troop.update_army()
        if self.own_race == Race.Protoss:
            self.race_specific_tactic()
        else:
            await self.race_specific_tactic()
        await self.distribute_workers(resource_ratio=1)
        self._rebalance_gas_workers_per_base()
        enemy_count = sum(observation["enemy"].values())
        self.under_attack = bool(enemy_count)

        if self.troop.is_attack:
            self.troop.add_army(self.units.of_type(constants.UNITS[self.own_race]) - self.units_before)
            if self.troop.check_power() <= self.troop.retreat_threshold:
                self.move_to_nexus(self.troop)
                self.troop.clear_army()
            if self.time >= self.next_auto_retreat_time and self.troop.check_fighting() > 0.8 * len(self.troop):
                self.move_to_nexus(self.troop)
                self.troop.clear_army()
                self.next_auto_retreat_time = self.time + 5
        elif not self.under_attack:
            self.move_to_nexus(self.units.of_type(constants.UNITS[self.own_race]))
        self.units_before = self.units.of_type(constants.UNITS[self.own_race])

    def get_information(self):
        trainable_units = constants.ACTION_DICT[self.own_race]["Train Unit"].values()
        buildable_structures = constants.ACTION_DICT[self.own_race]["Build Structure"].values()
        research_options = constants.ACTION_DICT[self.own_race]["Research Technique"].values()
        building_counts = {name: self._count_structures(name) for name in buildable_structures}
        pending_buildings = {
            name: count
            for name in buildable_structures
            if (count := self.already_pending(UnitTypeId[name.upper()])) > building_counts[name]
        }
        nearby_enemies = (
            unit
            for unit in self.enemy_units
            if self._is_enemy_observation_candidate(unit)
            and any(unit.distance_to(structure) < self.warning_range for structure in self.structures)
        )

        information = {
            "resource": {
                "game_time": self.time_formatted,
                "supply_cap": self.supply_cap,
                "supply_used": self.supply_used,
                "minerals": self.minerals,
                "vespene": self.vespene,
                "worker_supply": self.workers.amount,
            },
            "unit": {name: self.units(UnitTypeId[name.upper()]).amount for name in trainable_units},
            "building": building_counts,
            "pending_buildings": pending_buildings,
            "research": {
                name: self.already_pending_upgrade(constants.RESEARCHS[self.own_race][name][0])
                for name in research_options
            },
            "enemy": dict(Counter(unit.name for unit in nearby_enemies)),
            "previous_action": [action for timestamp, action in self.successful_actions if self.time - 60 <= timestamp < self.time],
        }
        gas_building = self._count_ready_and_in_progress_gas_buildings()
        if self.own_race == Race.Protoss:
            information["building"]["Assimilator"] = gas_building
            information["building"]["Gateway"] = self._count_structures("Gateway") + self._count_structures("Warpgate")
        elif self.own_race == Race.Terran:
            information["building"]["Refinery"] = gas_building
        return information

    def _count_structures(self, name):
        return sum(int(structure.build_progress) for structure in self.structures(UnitTypeId[name.upper()]))

    def get_queue_snapshot(self, queue):
        return [self.action.dict[action_id] for action_id in queue if isinstance(action_id, int)]

    def clear_action_queues(self):
        self.action_queue.clear()
        self.deferred_action_queue.clear()

    def build_decision_context(self, information, queued_actions=None):
        summary = {
            "race": self.own_race.name,
            "enemy_race": self.enemy_race.name,
            "resource": information["resource"],
            "unit": self._positive_values(information["unit"]),
            "building": self._positive_values(information["building"]),
            "pending_buildings": self._positive_values(information["pending_buildings"]),
            "research": [name for name, status in information["research"].items() if status == 1],
            "previous_action": information["previous_action"],
            "queued_actions": queued_actions or [],
        }
        if information["enemy"]:
            summary["observed_enemy"] = information["enemy"]
        attacking_enemy = self._get_attacking_enemy_summary()
        if attacking_enemy:
            summary["attacking_enemy"] = attacking_enemy
        if self.advice:
            summary["failed_action"] = self.advice[1]
            summary["failed_reason"] = f"{self.advice[2]} is {self.advice[0].replace('_', ' ')}."
            self.advice = None
        return summary

    @staticmethod
    def _positive_values(items):
        return {key: value for key, value in items.items() if value > 0}

    def get_action(self, observation, queued_actions=None):
        context = self.build_decision_context(observation, queued_actions)
        command = self.decision_backend.next_command(context, self.time_formatted)
        action_ids = self.action_extractor.extract_actions_from_command(command)

        if action_ids:
            self.next_inference = self.iteration + 10
        else:
            self.next_inference = self.iteration + 60

        for action_id in action_ids:
            self.action_queue.append(action_id)
            if self.own_race == Race.Terran:
                if action_id == 110:
                    self.action_queue.append(action_id)
                if action_id == 201 and not self.structures(UnitTypeId.REFINERY).exists:
                    self.action_queue.append(action_id)

    def record_succeed(self, action):
        action_name = self.action.dict[action]
        self.current_action_result = "success"
        self.next_refresh = self.iteration + self.refresh_period
        self.successful_actions.append((self.time, action_name))
        utils.save_data_to_file(
            f"{self.time_formatted} <{action_name}>", os.path.join(self.game_folder, "command.txt")
        )

    def record_failure(self, action, reason, precondition=None):
        self.failed_action = action
        self.failure_reason = reason
        self.precondition = precondition
        self.current_action_result = "failure"
        if reason != "not_afford":
            utils.save_data_to_file(
                f"{self.time_formatted} <{self.action.dict[action]}> {reason}",
                os.path.join(self.game_folder, "command.txt"),
            )

    def check_supply_cost(self, tgt):
        supply_building = {
            Race.Protoss: [UnitTypeId.PYLON, 205],
            Race.Zerg: [UnitTypeId.OVERLORD, 101],
            Race.Terran: [UnitTypeId.SUPPLYDEPOT, 213],
        }
        action_id = self.action.r_dict[tgt]
        if self.supply_left < self.calculate_supply_cost(UnitTypeId[tgt.upper()]):
            if self.supply_cap == 200:
                return self.record_failure(action_id, "supply_200")
            if self.already_pending(supply_building[self.own_race][0]):
                return self.record_failure(action_id, "supply_pending", supply_building[self.own_race][1])
            return self.record_failure(action_id, "not_exist", supply_building[self.own_race][1])
        return True

    def check_affordability(self, tgt, research=False):
        action_id = self.action.r_dict[tgt]
        if tgt == "MULE":
            return True
        check = constants.RESEARCHS[self.own_race][tgt][0] if research else UnitTypeId[tgt.upper()]
        if not self.can_afford(check):
            return self.record_failure(action_id, "not_afford")
        return True

    def check_research_status(self, tgt):
        action_id = self.action.r_dict[tgt]
        pending_status = self.already_pending(constants.RESEARCHS[self.own_race][tgt][0])
        if pending_status == 1:
            return self.record_failure(action_id, "already_done")
        if pending_status != 0:
            return self.record_failure(action_id, "already_on")
        return True

    def check_ability(self, tgt, abilities, train=False):
        action_id = self.action.r_dict[tgt]
        check = constants.ABILITYS[UnitTypeId[tgt.upper()]] if train else constants.RESEARCHS[self.own_race][tgt][1]
        if check not in abilities:
            return self.record_failure(action_id, "not_ability")
        return True

    def check_building_condition(self, tgt, buildings):
        action_id = self.action.r_dict[tgt]
        for building in buildings:
            precondition_id = self.action.r_dict[building]
            structures = self.structures(UnitTypeId[building.upper()])
            if structures.ready.exists:
                continue
            if self.already_pending(UnitTypeId[building.upper()]):
                return self.record_failure(action_id, "prereq_pending", precondition_id)
            if precondition_id in self.action_queue or precondition_id in self.deferred_action_queue:
                return self.record_failure(action_id, "prereq_queued", precondition_id)
            if not structures.exists:
                return self.record_failure(action_id, "not_exist", precondition_id)
        return True

    async def expand_now(self, building, max_distance=10, location=None):
        if location is None:
            location = await self.get_next_expansion()
        if not location:
            return False
        return await self.build(
            building,
            near=location,
            max_distance=max_distance,
            random_alternative=False,
            placement_step=1,
        )

    def check_condition(self, action_id, tgt, prerequisites):
        if not (
            self.check_building_condition(tgt, prerequisites)
            and self.check_supply_cost(tgt)
            and self.check_affordability(tgt)
        ):
            return False
        if action_id == 100 and self.workers.amount > 75:
            return self.record_failure(action_id, "many")
        if action_id == 200:
            if self.already_pending(UnitTypeId[tgt.upper()]):
                return self.record_failure(action_id, "pending_nexus")
            if self.structures(UnitTypeId[tgt.upper()]).amount == 7:
                return self.record_failure(action_id, "many")
        return True

    async def handle_build(self, tgt, prerequisites):
        action_id = self.action.r_dict[tgt]
        if not self.check_condition(action_id, tgt, prerequisites):
            return
        if action_id == 200:
            if await self.expand_now(UnitTypeId[tgt.upper()]):
                return self.record_succeed(action_id)
            return self.record_failure(action_id, "nexus_fail")
        if action_id == 201:
            free_geysers = self._get_open_geysers()
            if not free_geysers:
                return self.record_failure(action_id, "gas_full")
            for geyser in free_geysers:
                if await self.build(UnitTypeId[tgt.upper()], geyser):
                    return self.record_succeed(action_id)
            return self.record_failure(action_id, "gas_blocked")
        await self.action_build(tgt)

    async def handle_research(self, tgt, prerequisites):
        action_id = self.action.r_dict[tgt]
        if not self.check_building_condition(tgt, prerequisites) or not self.check_research_status(tgt):
            return

        idle_buildings = self.structures(UnitTypeId[prerequisites[0].upper()]).ready.idle
        precondition_id = self.action.r_dict[prerequisites[0]]
        if not idle_buildings:
            return self.record_failure(action_id, "producer_busy", precondition_id)

        building = idle_buildings.first
        abilities = await self.get_available_abilities(building)
        if not self.check_ability(tgt, abilities):
            return
        if not self.check_affordability(tgt, research=True):
            return

        self.do(building.research(constants.RESEARCHS[self.own_race][tgt][0]))
        return self.record_succeed(action_id)

    def attack(self, target_position=None):
        target_position = target_position or self.enemy_position
        self.action_queue.appendleft(f"attack_{target_position}")
        self.troop.is_attack = target_position
        utils.save_data_to_file(
            f"{self.time_formatted} <ATTACK TO {target_position}>", os.path.join(self.game_folder, "command.txt")
        )
        attack_units = sorted(
            self.units.of_type(constants.UNITS[self.own_race]),
            key=lambda unit: (-self.calculate_supply_cost(unit.type_id), unit.distance_to(constants.MAP_POINTS[self.troop.is_attack])),
        )[: int(self.army_count * 0.85)]
        self.troop.add_army(attack_units)

    async def process_queue_action(self, queue, source):
        if not queue:
            return

        action = queue.popleft()
        self.current_action_result = None
        if not isinstance(action, int):
            return

        method_name = f"handle_action_{action}"
        method = getattr(self, method_name, None)
        if method is None:
            return

        await method()
        if self.current_action_result != "failure":
            return

        self._apply_failure_policy(source)

    def _apply_failure_policy(self, source):
        policy = self.FAILURE_POLICIES.get(self.failure_reason, "drop_action")
        if policy == "retry_later":
            self.action_queue.appendleft(self.failed_action)
        elif policy == "retry_last":
            queue = self.deferred_action_queue if source == "deferred" else self.action_queue
            queue.append(self.failed_action)
        elif policy == "skip_and_defer":
            self.deferred_action_queue.append(self.failed_action)
        if policy == "blocked_by_prereq":
            self.clear_action_queues()
            self.advice = (
                self.failure_reason,
                self.action.dict[self.failed_action],
                self.action.dict.get(self.precondition, str(self.precondition)),
            )
        elif policy == "replan_now" and source != "deferred":
            self.clear_action_queues()

    def _reset_action_result(self):
        self.failed_action = self.failure_reason = self.precondition = self.current_action_result = None

    async def on_start(self):
        self.client.game_step = 4
        starts_at_c = self.townhalls[0].position == constants.MAP_POINTS["C"]
        self.start_position, self.enemy_position = ("C", "N") if starts_at_c else ("N", "C")
        self.next_scout = self.scout_period
        self.next_refresh = 9999
        self.next_inference = 0

        self.next_auto_retreat_time = 0
        self.successful_actions = []
        self.units_before = self.units.of_type(constants.UNITS[self.own_race])
        self._reset_action_result()

    async def on_step(self, iteration):
        if not self.townhalls:
            return

        self.iteration = iteration
        observation = self.get_information()
        self._reset_action_result()
        await self.update_cycle(observation)

        if self.iteration > self.next_refresh:
            self.clear_action_queues()

        can_request_action = self.iteration > self.next_inference and self.supply_used <= self.supply_cap and self.supply_used < 190
        queued_actions = self.get_queue_snapshot(self.deferred_action_queue)
        if not self.action_queue and can_request_action:
            self.get_action(observation, queued_actions=queued_actions)
        if not self.troop.is_attack and not self.under_attack:
            army_supply = sum(self.calculate_supply_cost(unit.type_id) for unit in self.units.of_type(constants.UNITS[self.own_race]))
            if self.supply_used > 190 or army_supply > self.troop.attack_threshold:
                self.attack()

        await self.process_queue_action(self.deferred_action_queue, source="deferred")
        await self.process_queue_action(self.action_queue, source="primary")

