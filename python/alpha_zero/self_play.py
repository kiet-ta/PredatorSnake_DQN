import numpy as np
from typing import Callable, List, Dict, Any
from .replay_buffer import SelfPlaySample
from predator_snake_dqn import PyAlphaZeroEngine
from .alpha_zero_net import AlphaZeroNet

def play_one_game(
    engine: PyAlphaZeroEngine,
    model_predict: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    temperature_threshold: int = 30,
) -> List[SelfPlaySample]:
    """
    Plays one full game of Snake using MCTS and returns the generated samples.
    """
    # Create partial samples (without outcome z)
    class PartialSample:
        def __init__(self, obs: np.ndarray, pi: np.ndarray):
            self.obs = obs
            self.pi = pi
            
    partial_samples: List[PartialSample] = []
    
    # Make sure engine is reset and has the model assigned
    engine.set_model(model_predict)
    engine.reset()
    
    step = 0
    final_score = 0
    
    while True:
        # Get observation BEFORE we step
        obs = engine.get_observation()
        
        # We need the Python MCTS search logic
        try:
            # result['policy'] has the search probabilities (tau=1)
            result = engine.mcts_search()
        except RuntimeError as e:
            # Environment is terminal (should not normally happen here unless max_steps reached)
            break
            
        policy = np.array(result['policy'], dtype=np.float32)
        
        # Store partial sample
        partial_samples.append(PartialSample(obs, policy))
        
        # Temperature tau application
        # 1.0 for first N steps -> use the policy distribution to sample action
        # 0.0 for rest -> argmax
        if step < temperature_threshold:
            # Add small epsilon to handle precision issues where sum != 1.0
            p = policy / policy.sum()
            chosen_action = int(np.random.choice(3, p=p))
        else:
            chosen_action = int(np.argmax(policy))
            
        # Explicitly step the engine with the chosen action
        next_obs, info, is_terminal = engine.step(chosen_action)
        final_score = info['score']
        
        if is_terminal:
            break
            
        step += 1
        
    # Calculate outcome z = tanh(score / 30.0)
    z = np.tanh(final_score / 30.0)
    
    # Backfill outcome to create full samples
    samples = []
    for p in partial_samples:
        samples.append(SelfPlaySample(
            observation=p.obs,
            target_policy=p.pi,
            outcome=z
        ))
        
    return samples

import queue
import threading

def generate_self_play_data(
    model: AlphaZeroNet,
    num_games: int,
    engine_kwargs: Dict[str, Any],
) -> List[SelfPlaySample]:
    """
    Generate `num_games` self-play games using a single AlphaZeroNet.
    Runs the model evaluator in a background thread to prevent GIL deadlocks
    with the Rust MCTS engine.
    """
    req_queue = queue.Queue()

    def evaluator_worker():
        while True:
            req = req_queue.get()
            if req is None:
                break
            obs, res_queue = req
            logits, values = model.predict(obs)
            res_queue.put((logits, values))

    worker = threading.Thread(target=evaluator_worker)
    worker.start()

    def proxy_predict(obs):
        res_queue = queue.Queue()
        req_queue.put((obs, res_queue))
        return res_queue.get()

    all_samples = []
    engine = PyAlphaZeroEngine(**engine_kwargs)
    
    try:
        for i in range(num_games):
            samples = play_one_game(engine, proxy_predict)
            all_samples.extend(samples)
    finally:
        req_queue.put(None)
        worker.join()
        
    return all_samples
