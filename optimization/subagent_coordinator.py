#!/usr/bin/env python3
"""
Subagent Task Distribution and Coordination System
Manages 20 parallel agents testing 205 strategies
"""

import json
import time
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from bot.config import Config


@dataclass
class AgentTask:
    """Task assignment for a single agent"""
    agent_id: int
    strategies: List[Dict]  # List of strategy configs to test
    timeframes: List[str]
    data_file: str
    stage: int
    cpu_cores: float
    memory_limit: str
    output_file: str
    checkpoint_file: str
    
    def to_dict(self):
        return asdict(self)
        

class SubagentCoordinator:
    """
    Coordinates task distribution among 20 subagents
    """
    
    def __init__(self, num_agents: int = 20):
        self.num_agents = num_agents
        self.total_cores = mp.cpu_count()
        self.usable_cores = min(12, self.total_cores - 2)  # Leave 2 cores for system
        
        # All timeframes to test
        self.timeframes = [
            '5min', '15min', '30min', '1h', 
            '2h', '4h', '8h', '16h', '1d'
        ]
        
        # Results collection
        self.results_dir = Path("task/subagent_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.default_data_file = Config.resolve_ohlcv_path() or Config.data_path('{asset}_ohlcv_{quote}_30day_5m.csv')
        
        # Strategy definitions (will be loaded from file)
        self.all_strategies = self._load_strategy_definitions()
        
    def _load_strategy_definitions(self) -> List[Dict]:
        """
        Load all 205 strategy definitions
        For now, create placeholder definitions
        """
        strategies = []
        
        # Top tier strategies (manually defined)
        top_strategies = [
            {"id": 1, "name": "SqueezeMomentum", "type": "momentum", "priority": 1},
            {"id": 2, "name": "WaveTrend", "type": "oscillator", "priority": 1},
            {"id": 3, "name": "CoralTrend", "type": "trend", "priority": 1},
            {"id": 4, "name": "SchaffTrendCycle", "type": "hybrid", "priority": 1},
            {"id": 5, "name": "MESAAdaptiveMA", "type": "adaptive", "priority": 1},
            {"id": 6, "name": "ElderImpulse", "type": "momentum", "priority": 2},
            {"id": 7, "name": "FRAMA", "type": "adaptive", "priority": 2},
            {"id": 8, "name": "ZeroLagEMA", "type": "trend", "priority": 2},
            {"id": 9, "name": "KaufmannAMA", "type": "adaptive", "priority": 2},
            {"id": 10, "name": "TradersDynamicIndex", "type": "composite", "priority": 2},
            {"id": 11, "name": "InsyncIndex", "type": "consensus", "priority": 3},
            {"id": 12, "name": "PremierStochastic", "type": "oscillator", "priority": 3},
            {"id": 13, "name": "MAC_Z", "type": "momentum", "priority": 3},
            {"id": 14, "name": "FireflyOscillator", "type": "oscillator", "priority": 3},
            {"id": 15, "name": "CompositeMomentumIndex", "type": "momentum", "priority": 3},
        ]
        
        strategies.extend(top_strategies)
        
        # Generate placeholder strategies for the remaining 190
        # These would be replaced with actual strategy definitions
        strategy_types = ["momentum", "trend", "oscillator", "volume", "volatility", "hybrid"]
        
        for i in range(16, 206):  # 16-205
            strategies.append({
                "id": i,
                "name": f"Strategy_{i}",
                "type": strategy_types[(i - 16) % len(strategy_types)],
                "priority": 4 + (i - 16) // 50  # Lower priority for later strategies
            })
            
        return strategies
        
    def distribute_tasks_stage1(self) -> List[AgentTask]:
        """
        Distribute strategies for Stage 1: Broad Discovery
        Each agent gets ~10-11 strategies to test on all timeframes
        """
        tasks = []
        strategies_per_agent = len(self.all_strategies) // self.num_agents
        extra_strategies = len(self.all_strategies) % self.num_agents
        
        strategy_index = 0
        
        for agent_id in range(1, self.num_agents + 1):
            # Calculate how many strategies this agent gets
            num_strategies = strategies_per_agent
            if agent_id <= extra_strategies:
                num_strategies += 1
                
            # Assign strategies
            agent_strategies = self.all_strategies[strategy_index:strategy_index + num_strategies]
            strategy_index += num_strategies
            
            # Create task
            task = AgentTask(
                agent_id=agent_id,
                strategies=agent_strategies,
                timeframes=self.timeframes,
                data_file=self.default_data_file,
                stage=1,
                cpu_cores=self.usable_cores / self.num_agents,
                memory_limit="1.5GB",
                output_file=str(self.results_dir / f"stage1_agent_{agent_id:02d}.json"),
                checkpoint_file=str(self.results_dir / f"checkpoint_agent_{agent_id:02d}.pkl")
            )
            
            tasks.append(task)
            
        return tasks
        
    def distribute_tasks_stage2(self, top_strategies: List[Dict]) -> List[AgentTask]:
        """
        Distribute strategies for Stage 2: Validation
        Top 60 strategies distributed among agents for parameter testing
        """
        tasks = []
        strategies_per_agent = len(top_strategies) // self.num_agents
        extra_strategies = len(top_strategies) % self.num_agents
        
        strategy_index = 0
        
        for agent_id in range(1, self.num_agents + 1):
            num_strategies = strategies_per_agent
            if agent_id <= extra_strategies:
                num_strategies += 1
                
            agent_strategies = top_strategies[strategy_index:strategy_index + num_strategies]
            strategy_index += num_strategies
            
            task = AgentTask(
                agent_id=agent_id,
                strategies=agent_strategies,
                timeframes=self.timeframes,
                data_file=self.default_data_file,
                stage=2,
                cpu_cores=self.usable_cores / self.num_agents,
                memory_limit="2GB",
                output_file=str(self.results_dir / f"stage2_agent_{agent_id:02d}.json"),
                checkpoint_file=str(self.results_dir / f"checkpoint_agent_{agent_id:02d}.pkl")
            )
            
            tasks.append(task)
            
        return tasks
        
    def distribute_tasks_stage3(self, top_strategies: List[Dict]) -> List[AgentTask]:
        """
        Distribute strategies for Stage 3: Deep Optimization
        Top 20 strategies for full optimization and ensemble testing
        """
        tasks = []
        
        # For stage 3, we might want different distribution
        # Some agents do deep optimization, others test ensembles
        
        # First 10 agents: Deep optimization of individual strategies
        for agent_id in range(1, 11):
            # Each agent gets 2 strategies for deep optimization
            start_idx = (agent_id - 1) * 2
            agent_strategies = top_strategies[start_idx:start_idx + 2]
            
            task = AgentTask(
                agent_id=agent_id,
                strategies=agent_strategies,
                timeframes=self.timeframes,
                data_file=self.default_data_file,
                stage=3,
                cpu_cores=self.usable_cores / 10,  # More cores for deep optimization
                memory_limit="3GB",
                output_file=str(self.results_dir / f"stage3_opt_agent_{agent_id:02d}.json"),
                checkpoint_file=str(self.results_dir / f"checkpoint_agent_{agent_id:02d}.pkl")
            )
            
            tasks.append(task)
            
        # Next 10 agents: Ensemble testing
        for agent_id in range(11, 21):
            # These agents test ensemble combinations
            task = AgentTask(
                agent_id=agent_id,
                strategies=top_strategies,  # All strategies for ensemble
                timeframes=["1h", "4h", "1d"],  # Focus on key timeframes
                data_file=self.default_data_file,
                stage=3,
                cpu_cores=self.usable_cores / 10,
                memory_limit="2GB",
                output_file=str(self.results_dir / f"stage3_ens_agent_{agent_id:02d}.json"),
                checkpoint_file=str(self.results_dir / f"checkpoint_agent_{agent_id:02d}.pkl")
            )
            
            tasks.append(task)
            
        return tasks
        
    def save_task_assignments(self, tasks: List[AgentTask], stage: int):
        """Save task assignments to file"""
        output_file = self.results_dir / f"stage{stage}_assignments.json"
        
        assignments = []
        for task in tasks:
            assignment = task.to_dict()
            assignments.append(assignment)
            
        with open(output_file, 'w') as f:
            json.dump(assignments, f, indent=2)
            
        print(f"Saved {len(tasks)} task assignments to {output_file}")
        
    def collect_results(self, stage: int) -> Dict:
        """Collect results from all agents for a given stage"""
        results = {
            "stage": stage,
            "timestamp": time.time(),
            "agent_results": {},
            "top_strategies": []
        }
        
        # Read all agent result files
        pattern = f"stage{stage}_*.json"
        for result_file in self.results_dir.glob(pattern):
            try:
                with open(result_file, 'r') as f:
                    agent_data = json.load(f)
                    agent_id = agent_data.get("agent_id", "unknown")
                    results["agent_results"][agent_id] = agent_data
            except Exception as e:
                print(f"Error reading {result_file}: {e}")
                
        # Aggregate top strategies across all agents
        all_strategies = []
        for agent_data in results["agent_results"].values():
            if "top_strategies" in agent_data:
                all_strategies.extend(agent_data["top_strategies"])
                
        # Sort by CPS score
        all_strategies.sort(key=lambda x: x.get("cps", 0), reverse=True)
        results["top_strategies"] = all_strategies[:60]  # Keep top 60
        
        return results
        
    def generate_agent_instructions(self, task: AgentTask) -> str:
        """
        Generate detailed instructions for a subagent
        """
        instructions = f"""
# Subagent Task Instructions

## Agent ID: {task.agent_id}
## Stage: {task.stage}

### Assigned Strategies
You are responsible for testing the following {len(task.strategies)} strategies:
"""
        
        for strategy in task.strategies:
            instructions += f"- {strategy['name']} (ID: {strategy['id']}, Type: {strategy['type']})\n"
            
        instructions += f"""

### Testing Parameters
- Timeframes: {', '.join(task.timeframes)}
- Data file: {task.data_file}
- CPU allocation: {task.cpu_cores:.2f} cores
- Memory limit: {task.memory_limit}

### Output Requirements
1. Save results to: {task.output_file}
2. Save checkpoints to: {task.checkpoint_file}
3. Report format must include:
   - CPS score for each strategy/timeframe combination
   - Detailed metrics (return, drawdown, Sharpe, etc.)
   - Insights and pairing suggestions
   - Any unexpected discoveries

### Scoring Criteria
Use the Composite Performance Score (CPS) with:
- 30% Profit Score
- 25% Capital Preservation Score
- 20% Risk-Adjusted Score
- 15% Trade Activity Score (profit-aware)
- 10% Trend Detection Score

### Special Instructions
- If strategies show promise when combined, note this in insights
- Pay attention to strategies that work well in downtrends
- Note any timeframe-specific behaviors
- Report strategies that might work as ensemble components

Remember: We're looking for strategies that can beat Buy & Hold (100.88% annual return)
while also preserving capital in downtrends and catching trend reversals.
"""
        
        return instructions
        

def test_coordinator():
    """Test the subagent coordinator"""
    coordinator = SubagentCoordinator(num_agents=20)
    
    print(f"Total strategies: {len(coordinator.all_strategies)}")
    print(f"Agents: {coordinator.num_agents}")
    print(f"CPU cores available: {coordinator.usable_cores}")
    
    # Test Stage 1 distribution
    print("\n" + "="*60)
    print("STAGE 1: Broad Discovery Distribution")
    print("="*60)
    
    stage1_tasks = coordinator.distribute_tasks_stage1()
    
    for task in stage1_tasks[:3]:  # Show first 3 agents
        print(f"\nAgent {task.agent_id}:")
        print(f"  Strategies: {len(task.strategies)} ({task.strategies[0]['name']} to {task.strategies[-1]['name']})")
        print(f"  Tests: {len(task.strategies)} × {len(task.timeframes)} = {len(task.strategies) * len(task.timeframes)}")
        print(f"  CPU: {task.cpu_cores:.2f} cores")
        
    # Save assignments
    coordinator.save_task_assignments(stage1_tasks, stage=1)
    
    # Generate sample instructions
    print("\n" + "="*60)
    print("SAMPLE AGENT INSTRUCTIONS")
    print("="*60)
    print(coordinator.generate_agent_instructions(stage1_tasks[0]))
    

if __name__ == "__main__":
    test_coordinator()
