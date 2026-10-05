"""
游戏内世界模拟与多智能体系统 - 核心模块

功能：
- agent_update():       智能体更新（感知 + 决策 + 行动）
- interaction():        NPC 交互（对话/交易/战斗/合作，交互改变状态）
- emergent_behavior():  涌现行为（自组织聚集 + 临界点判定）
- swarm_intelligence(): 群体智能（信息素引导 + 自动分工）
- world_state():        世界状态（统计 + 趋势）
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import random


INTERACTION_TYPES = ("dialogue", "trade", "combat", "cooperation")
ROLES = ("gatherer", "scout", "defender")
_TREND_KEYS = ("num_agents", "avg_health", "avg_energy", "avg_wealth", "pheromone_total")


@dataclass
class Config:
    pass


@dataclass
class Agent:
    agent_id: int
    x: float
    y: float
    health: float = 100.0
    energy: float = 100.0
    wealth: float = 10.0
    role: str = "idle"
    action: str = "idle"
    perception: List[int] = field(default_factory=list)
    memory: Dict = field(default_factory=dict)
    relationships: Dict[int, float] = field(default_factory=dict)


class WorldSimulation:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.width = float(self.config.get("width", 100.0))
        self.height = float(self.config.get("height", 100.0))
        self.perception_radius = float(self.config.get("perception_radius", 10.0))
        self.interaction_radius = float(self.config.get("interaction_radius", 5.0))
        self.group_radius = float(self.config.get("group_radius", self.perception_radius))
        self.min_group_size = int(self.config.get("min_group_size", 3))
        self.critical_mass = float(self.config.get("critical_mass", 0.5))
        self.cell_size = float(self.config.get("cell_size", 5.0))
        self.pheromone_decay = float(self.config.get("pheromone_decay", 0.9))
        self.pheromone_deposit = float(self.config.get("pheromone_deposit", 1.0))
        self.speed = float(self.config.get("speed", 5.0))
        self._rng = random.Random(self.config.get("seed"))
        self.agents: Dict[int, Agent] = {}
        self.pheromones: Dict[Tuple[int, int], float] = {}
        self.interaction_log: List[Dict] = []
        self.time = 0.0
        self._next_id = 0
        self._state = {}
        self._history = []

    # ------------------------------------------------------------------
    # 智能体管理
    # ------------------------------------------------------------------
    def add_agent(self, x: Optional[float] = None, y: Optional[float] = None, **attrs) -> int:
        agent_id = self._next_id
        self._next_id += 1
        if x is None:
            x = self._rng.uniform(0.0, self.width)
        if y is None:
            y = self._rng.uniform(0.0, self.height)
        agent = Agent(agent_id=agent_id, x=float(x), y=float(y))
        for key, value in attrs.items():
            if hasattr(agent, key):
                setattr(agent, key, value)
        self.agents[agent_id] = agent
        return agent_id

    def spawn_agents(self, n: int, **attrs) -> List[int]:
        return [self.add_agent(**attrs) for _ in range(n)]

    def _distance(self, a: Agent, b: Agent) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def _clamp(self, agent: Agent):
        agent.x = min(max(agent.x, 0.0), self.width)
        agent.y = min(max(agent.y, 0.0), self.height)

    # ------------------------------------------------------------------
    # 智能体更新：感知 + 决策 + 行动
    # ------------------------------------------------------------------
    def _perceive(self, agent: Agent) -> List[int]:
        return [
            other.agent_id
            for other in self.agents.values()
            if other.agent_id != agent.agent_id
            and self._distance(agent, other) <= self.perception_radius
        ]

    def _decide(self, agent: Agent, neighbors: List[int]) -> str:
        if agent.energy < 30.0:
            return "rest"
        if agent.health < 30.0 and neighbors:
            return "flee"
        if neighbors and agent.role == "defender":
            return "guard"
        if neighbors:
            return "approach"
        return "wander"

    def agent_update(self, agent_id: int, dt: float = 1.0) -> Dict:
        agent = self.agents[agent_id]
        neighbors = self._perceive(agent)
        agent.perception = neighbors
        action = self._decide(agent, neighbors)
        agent.action = action

        step = self.speed * dt
        if action == "rest":
            agent.energy = min(100.0, agent.energy + 10.0 * dt)
        elif action in ("approach", "flee"):
            nearest = min((self.agents[nid] for nid in neighbors),
                          key=lambda o: self._distance(agent, o))
            dx, dy = nearest.x - agent.x, nearest.y - agent.y
            dist = math.hypot(dx, dy) or 1.0
            sign = 1.0 if action == "approach" else -1.0
            agent.x += sign * step * dx / dist
            agent.y += sign * step * dy / dist
            agent.energy = max(0.0, agent.energy - 1.0 * dt)
        elif action == "wander":
            angle = self._rng.uniform(0.0, 2.0 * math.pi)
            agent.x += step * math.cos(angle)
            agent.y += step * math.sin(angle)
            agent.energy = max(0.0, agent.energy - 1.0 * dt)
        # guard: 原地警戒，不移动
        self._clamp(agent)
        return {
            "agent_id": agent_id,
            "perception": list(neighbors),
            "action": action,
            "position": (agent.x, agent.y),
        }

    # ------------------------------------------------------------------
    # 交互：有类型（对话/交易/战斗/合作），有结果（改变状态）
    # ------------------------------------------------------------------
    def interaction(self, a_id: int, b_id: int, interaction_type: Optional[str] = None) -> Dict:
        if interaction_type is None:
            interaction_type = "dialogue"
        if interaction_type not in INTERACTION_TYPES:
            raise ValueError(f"未知交互类型: {interaction_type}")
        a, b = self.agents[a_id], self.agents[b_id]
        outcome: Dict = {}

        if interaction_type == "dialogue":
            a.relationships[b_id] = a.relationships.get(b_id, 0.0) + 1.0
            b.relationships[a_id] = b.relationships.get(a_id, 0.0) + 1.0
            a.memory.setdefault("met", []).append(b_id)
            b.memory.setdefault("met", []).append(a_id)
            outcome = {"relationship_delta": 1.0}
        elif interaction_type == "trade":
            richer, poorer = (a, b) if a.wealth >= b.wealth else (b, a)
            amount = min(1.0, richer.wealth)
            richer.wealth -= amount
            poorer.wealth += amount
            outcome = {"amount": amount, "from": richer.agent_id, "to": poorer.agent_id}
        elif interaction_type == "combat":
            dmg_to_b = 10.0 if a.energy >= b.energy else 5.0
            dmg_to_a = 10.0 if b.energy > a.energy else 5.0
            b.health = max(0.0, b.health - dmg_to_b)
            a.health = max(0.0, a.health - dmg_to_a)
            a.relationships[b_id] = a.relationships.get(b_id, 0.0) - 2.0
            b.relationships[a_id] = b.relationships.get(a_id, 0.0) - 2.0
            outcome = {"damage_dealt": {a_id: dmg_to_b, b_id: dmg_to_a}}
        elif interaction_type == "cooperation":
            a.energy = max(0.0, a.energy - 5.0)
            b.energy = max(0.0, b.energy - 5.0)
            a.wealth += 2.0
            b.wealth += 2.0
            a.relationships[b_id] = a.relationships.get(b_id, 0.0) + 2.0
            b.relationships[a_id] = b.relationships.get(a_id, 0.0) + 2.0
            outcome = {"reward": 2.0, "energy_cost": 5.0}

        result = {"type": interaction_type, "agents": (a_id, b_id), "outcome": outcome}
        self.interaction_log.append({"time": self.time, **result})
        return result

    # ------------------------------------------------------------------
    # 涌现行为：自组织聚集 + 临界点
    # ------------------------------------------------------------------
    def _find_groups(self) -> List[List[int]]:
        ids = sorted(self.agents)
        parent = {i: i for i in ids}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            parent[find(i)] = find(j)

        for idx, i in enumerate(ids):
            for j in ids[idx + 1:]:
                if self._distance(self.agents[i], self.agents[j]) <= self.group_radius:
                    union(i, j)
        clusters: Dict[int, List[int]] = {}
        for i in ids:
            clusters.setdefault(find(i), []).append(i)
        return [members for members in clusters.values() if len(members) >= self.min_group_size]

    def emergent_behavior(self) -> Dict:
        groups = self._find_groups()
        total = len(self.agents)
        in_groups = sum(len(g) for g in groups)
        order_parameter = (in_groups / total) if total else 0.0
        emerged = order_parameter >= self.critical_mass and len(groups) > 0
        return {
            "self_organized": len(groups) > 0,
            "num_groups": len(groups),
            "group_sizes": sorted((len(g) for g in groups), reverse=True),
            "order_parameter": order_parameter,
            "critical_point": self.critical_mass,
            "emerged": emerged,
        }

    # ------------------------------------------------------------------
    # 群体智能：信息素 + 分工
    # ------------------------------------------------------------------
    def _cell(self, x: float, y: float) -> Tuple[int, int]:
        return (int(x // self.cell_size), int(y // self.cell_size))

    def _assign_roles(self):
        idle = [a for a in sorted(self.agents) if self.agents[a].role == "idle"]
        for idx, agent_id in enumerate(idle):
            self.agents[agent_id].role = ROLES[idx % len(ROLES)]

    def swarm_intelligence(self) -> Dict:
        # 分工：空闲智能体自动分配角色
        self._assign_roles()

        # 信息素衰减
        for cell in list(self.pheromones):
            self.pheromones[cell] *= self.pheromone_decay
            if self.pheromones[cell] < 0.01:
                del self.pheromones[cell]

        # 采集者留下信息素，引导其他 NPC
        for agent in self.agents.values():
            if agent.role == "gatherer":
                cell = self._cell(agent.x, agent.y)
                self.pheromones[cell] = self.pheromones.get(cell, 0.0) + self.pheromone_deposit

        # 其他 NPC 沿信息素梯度移动
        followers = 0
        for agent in self.agents.values():
            cx, cy = self._cell(agent.x, agent.y)
            current = self.pheromones.get((cx, cy), 0.0)
            best_cell, best_level = None, current
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    level = self.pheromones.get((cx + dx, cy + dy), 0.0)
                    if level > best_level:
                        best_cell, best_level = (cx + dx, cy + dy), level
            if best_cell is not None:
                tx = (best_cell[0] + 0.5) * self.cell_size
                ty = (best_cell[1] + 0.5) * self.cell_size
                dx, dy = tx - agent.x, ty - agent.y
                dist = math.hypot(dx, dy)
                if dist > 0:
                    step = min(self.cell_size, dist)
                    agent.x += step * dx / dist
                    agent.y += step * dy / dist
                    self._clamp(agent)
                    followers += 1

        roles: Dict[str, int] = {}
        for agent in self.agents.values():
            roles[agent.role] = roles.get(agent.role, 0) + 1
        return {
            "roles": roles,
            "division_of_labor": len([r for r in roles if r != "idle"]) > 1,
            "pheromone_cells": len(self.pheromones),
            "total_pheromone": sum(self.pheromones.values()),
            "followers": followers,
        }

    # ------------------------------------------------------------------
    # 世界状态：统计 + 趋势
    # ------------------------------------------------------------------
    def world_state(self) -> Dict:
        agents = list(self.agents.values())
        n = len(agents)
        roles: Dict[str, int] = {}
        actions: Dict[str, int] = {}
        for agent in agents:
            roles[agent.role] = roles.get(agent.role, 0) + 1
            actions[agent.action] = actions.get(agent.action, 0) + 1
        stats = {
            "num_agents": n,
            "avg_health": sum(a.health for a in agents) / n if n else 0.0,
            "avg_energy": sum(a.energy for a in agents) / n if n else 0.0,
            "avg_wealth": sum(a.wealth for a in agents) / n if n else 0.0,
            "roles": roles,
            "actions": actions,
            "interactions": len(self.interaction_log),
            "pheromone_total": sum(self.pheromones.values()),
            "groups": self.emergent_behavior()["num_groups"],
        }

        trends: Dict[str, str] = {}
        prev = self._history[-1]["statistics"] if self._history else None
        for key in _TREND_KEYS:
            if prev is None:
                trends[key] = "stable"
            else:
                diff = stats[key] - prev[key]
                trends[key] = "rising" if diff > 1e-6 else "falling" if diff < -1e-6 else "stable"

        state = {"time": self.time, "statistics": stats, "trends": trends}
        self._history.append(state)
        self._state = state
        return state

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def update(self, dt: float):
        self.time += dt
        for agent_id in list(self.agents):
            self.agent_update(agent_id, dt)

        # 距离近的 NPC 自动发生交互（类型按确定性规则轮换）
        auto_types = ("dialogue", "trade", "cooperation")
        ids = sorted(self.agents)
        for idx, a_id in enumerate(ids):
            for b_id in ids[idx + 1:]:
                if self._distance(self.agents[a_id], self.agents[b_id]) <= self.interaction_radius:
                    self.interaction(a_id, b_id, auto_types[(a_id + b_id) % len(auto_types)])

        self.swarm_intelligence()
        self.emergent_behavior()
        return self.world_state()

    def reset(self):
        self.agents = {}
        self.pheromones = {}
        self.interaction_log = []
        self.time = 0.0
        self._next_id = 0
        self._state = {}
        self._history = []
