"""
游戏内世界模拟与多智能体系统 - 验收测试
运行方式：python -m pytest tests/test_world_simulation.py -q
共 16 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.world import WorldSimulation


class TestWorldSimulation:
    def setup_method(self):
        self.system = WorldSimulation()

    # ---- agent_update: 感知 ----
    def test_case_01(self):
        a = self.system.add_agent(x=10, y=10)
        b = self.system.add_agent(x=15, y=10)
        far = self.system.add_agent(x=90, y=90)
        result = self.system.agent_update(a)
        assert b in result["perception"]
        assert far not in result["perception"]
        assert b in self.system.agents[a].perception

    # ---- agent_update: 决策 ----
    def test_case_02(self):
        tired = self.system.add_agent(x=10, y=10, energy=10.0)
        social = self.system.add_agent(x=50, y=50)
        self.system.add_agent(x=53, y=50)
        r1 = self.system.agent_update(tired)
        r2 = self.system.agent_update(social)
        assert r1["action"] == "rest"
        assert r2["action"] == "approach"

    # ---- agent_update: 行动改变状态 ----
    def test_case_03(self):
        tired = self.system.add_agent(x=10, y=10, energy=10.0)
        self.system.agent_update(tired, dt=1.0)
        assert self.system.agents[tired].energy > 10.0
        lonely = self.system.add_agent(x=50, y=50)
        before = (self.system.agents[lonely].x, self.system.agents[lonely].y)
        self.system.agent_update(lonely, dt=1.0)
        after = (self.system.agents[lonely].x, self.system.agents[lonely].y)
        assert before != after

    # ---- interaction: 类型 ----
    def test_case_04(self):
        for itype in ("dialogue", "trade", "combat", "cooperation"):
            a = self.system.add_agent(x=10, y=10)
            b = self.system.add_agent(x=12, y=10)
            result = self.system.interaction(a, b, itype)
            assert result["type"] == itype
        a = self.system.add_agent(x=0, y=0)
        b = self.system.add_agent(x=1, y=0)
        with pytest.raises(ValueError):
            self.system.interaction(a, b, "dance")

    # ---- interaction: 交易结果改变财富 ----
    def test_case_05(self):
        a = self.system.add_agent(x=0, y=0, wealth=20.0)
        b = self.system.add_agent(x=1, y=0, wealth=10.0)
        result = self.system.interaction(a, b, "trade")
        assert result["outcome"]["amount"] == pytest.approx(1.0)
        assert self.system.agents[a].wealth == pytest.approx(19.0)
        assert self.system.agents[b].wealth == pytest.approx(11.0)

    # ---- interaction: 战斗/对话结果改变状态 ----
    def test_case_06(self):
        a = self.system.add_agent(x=0, y=0, energy=80.0)
        b = self.system.add_agent(x=1, y=0, energy=50.0)
        self.system.interaction(a, b, "combat")
        assert self.system.agents[a].health == pytest.approx(95.0)
        assert self.system.agents[b].health == pytest.approx(90.0)
        rel_before = self.system.agents[a].relationships[b]
        self.system.interaction(a, b, "dialogue")
        assert self.system.agents[a].relationships[b] == rel_before + 1.0
        assert b in self.system.agents[a].memory["met"]

    # ---- emergent_behavior: 自组织 ----
    def test_case_07(self):
        for i in range(3):
            self.system.add_agent(x=10 + i, y=10)
            self.system.add_agent(x=80 + i, y=80)
        self.system.add_agent(x=50, y=50)  # 孤立个体
        result = self.system.emergent_behavior()
        assert result["self_organized"] is True
        assert result["num_groups"] == 2
        assert result["order_parameter"] == pytest.approx(6 / 7)

    # ---- emergent_behavior: 临界点 ----
    def test_case_08(self):
        sparse = WorldSimulation({"seed": 1})
        for i in range(10):
            sparse.add_agent(x=(i % 5) * 20.0, y=(i // 5) * 80.0)
        assert sparse.emergent_behavior()["emerged"] is False

        dense = WorldSimulation({"seed": 1})
        for i in range(10):
            dense.add_agent(x=50.0 + (i % 5), y=50.0 + (i // 5))
        result = dense.emergent_behavior()
        assert result["order_parameter"] >= result["critical_point"]
        assert result["emerged"] is True

    # ---- swarm_intelligence: 信息素沉积 ----
    def test_case_09(self):
        for i in range(3):
            self.system.add_agent(x=10 * i + 5, y=5, role="gatherer")
        result = self.system.swarm_intelligence()
        assert result["total_pheromone"] == pytest.approx(3.0)
        assert result["pheromone_cells"] >= 1

    # ---- swarm_intelligence: 信息素衰减与引导 ----
    def test_case_10(self):
        scout = self.system.add_agent(x=50.0, y=50.0, role="scout")
        self.system.pheromones[(11, 10)] = 5.0  # 相邻格的信息素
        result = self.system.swarm_intelligence()
        assert self.system.pheromones[(11, 10)] == pytest.approx(4.5)  # 衰减
        assert self.system.agents[scout].x > 50.0  # 被信息素引导
        assert result["followers"] == 1

    # ---- swarm_intelligence: 分工 ----
    def test_case_11(self):
        self.system.spawn_agents(9)
        result = self.system.swarm_intelligence()
        assert result["division_of_labor"] is True
        assert result["roles"]["gatherer"] == 3
        assert result["roles"]["scout"] == 3
        assert result["roles"]["defender"] == 3

    # ---- world_state: 统计 ----
    def test_case_12(self):
        self.system.add_agent(x=0, y=0, health=80.0, energy=60.0, wealth=4.0)
        self.system.add_agent(x=1, y=0, health=100.0, energy=100.0, wealth=10.0)
        stats = self.system.world_state()["statistics"]
        assert stats["num_agents"] == 2
        assert stats["avg_health"] == pytest.approx(90.0)
        assert stats["avg_energy"] == pytest.approx(80.0)
        assert stats["avg_wealth"] == pytest.approx(7.0)
        assert stats["roles"] == {"idle": 2}

    # ---- world_state: 趋势 ----
    def test_case_13(self):
        self.system.spawn_agents(5, energy=10.0)
        self.system.world_state()  # 基线快照
        state = self.system.update(1.0)  # 低能量 NPC 休息，平均能量上升
        assert state["trends"]["avg_energy"] == "rising"
        assert "avg_health" in state["trends"]

    # ---- update: 世界整体演化 ----
    def test_case_14(self):
        world = WorldSimulation({"seed": 42})
        world.spawn_agents(100)
        state = world.update(1.0)
        assert state["statistics"]["num_agents"] == 100
        assert len(world._history) == 1
        assert len(world.interaction_log) > 0
        assert world._state is state

    # ---- reset ----
    def test_case_15(self):
        self.system.spawn_agents(10)
        self.system.update(1.0)
        self.system.reset()
        assert len(self.system.agents) == 0
        assert len(self.system._history) == 0
        assert len(self.system.pheromones) == 0
        assert self.system.time == 0.0
        assert self.system.world_state()["statistics"]["num_agents"] == 0

    # ---- 综合: 100 个 NPC 的群体行为 ----
    def test_case_16(self):
        world = WorldSimulation({"seed": 7})
        for c in range(5):  # 5 个聚落，每个 20 个 NPC
            for i in range(20):
                world.add_agent(x=10.0 + c * 20 + (i % 5), y=10.0 + (i // 5))
        for _ in range(3):
            world.update(1.0)
        emergent = world.emergent_behavior()
        assert emergent["emerged"] is True
        assert emergent["num_groups"] >= 1
        swarm = world.swarm_intelligence()
        assert swarm["division_of_labor"] is True
        state = world.world_state()
        assert state["statistics"]["num_agents"] == 100
        assert state["statistics"]["interactions"] > 0
        assert set(state["trends"]) >= {"avg_energy", "avg_health"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
