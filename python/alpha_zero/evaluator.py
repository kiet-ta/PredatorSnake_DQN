import numpy as np
import logging
from typing import Callable
from predator_snake_dqn import PyAlphaZeroEngine
from .alpha_zero_net import AlphaZeroNet

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

class AlphaZeroEvaluator:
    def __init__(self, margin: float = 1.05):
        """
        Evaluates a challenger network against a champion network.
        :param margin: The challenger must score `margin` times better than champion to win.
        """
        self.margin = margin

    def evaluate_challenger(
        self, 
        champion_net: AlphaZeroNet, 
        challenger_net: AlphaZeroNet, 
        engine: PyAlphaZeroEngine, 
        num_games: int = 40
    ) -> tuple[bool, float, float]:
        """
        Plays num_games for both champion and challenger using the given MCTS engine.
        Temperature tau is implicitly 0 (greedy argmax) for the whole game.
        Returns a tuple: (challenger_wins, champion_score, challenger_score)
        """
        logger.info(f"--- Evaluator Clash: Champion vs Challenger ({num_games} games each) ---")
        
        champion_score = self._play_eval_games(champion_net.predict, engine, num_games)
        challenger_score = self._play_eval_games(challenger_net.predict, engine, num_games)
        
        logger.info(f"Clash Results: Champion Score: {champion_score:.2f} | Challenger Score: {challenger_score:.2f}")
        
        # Check win condition
        if champion_score == 0:
            challenger_wins = challenger_score > 0
        else:
            challenger_wins = challenger_score > (champion_score * self.margin)
            
        if challenger_wins:
            logger.info(">>> CHALLENGER WINS! The new model is superior. <<<")
        else:
            logger.info(">>> CHAMPION HOLDS! The challenger was rejected. <<<")
            
        return challenger_wins, champion_score, challenger_score

    def _play_eval_games(
        self, 
        model_predict: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]], 
        engine: PyAlphaZeroEngine, 
        num_games: int
    ) -> float:
        total_score = 0.0
        
        for _ in range(num_games):
            engine.set_model(model_predict)
            engine.reset()
            
            while True:
                try:
                    result = engine.mcts_search()
                except RuntimeError:
                    break
                    
                policy = np.array(result['policy'], dtype=np.float32)
                
                # tau -> 0 means pure greedy argmax
                chosen_action = int(np.argmax(policy))
                
                _, info, is_terminal = engine.step(chosen_action)
                
                if is_terminal:
                    total_score += info['score']
                    break
                    
        return total_score / num_games
