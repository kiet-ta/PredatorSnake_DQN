import argparse
import os
import torch
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from predator_snake_dqn import PyAlphaZeroEngine
from alpha_zero.alpha_zero_net import AlphaZeroNet
from alpha_zero.replay_buffer import ReplayBuffer
from alpha_zero.trainer import AlphaZeroTrainer
from alpha_zero.self_play import generate_self_play_data
from alpha_zero.evaluator import AlphaZeroEvaluator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iterations", type=int, default=1000)
    parser.add_argument("--games-per-iter", type=int, default=200)
    parser.add_argument("--epochs-per-iter", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--log-dir", type=str, default="./tensorboard/alphazero_long")
    
    # Engine specific kwargs
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--num-simulations", type=int, default=100)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--c-puct", type=float, default=1.0)
    
    return parser.parse_args()

def load_checkpoint(path: str, model: AlphaZeroNet, optimizer: torch.optim.Optimizer = None) -> int:
    """Loads checkpoint and returns the iteration it was saved at."""
    if not os.path.exists(path):
        return 0
    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("iteration", 0)
    else:
        # Fallback for old weights
        model.load_state_dict(checkpoint)
        return 0

def save_checkpoint(path: str, model: AlphaZeroNet, optimizer: torch.optim.Optimizer, iteration: int):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }, path)

def main():
    args = parse_args()
    
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    latest_net = AlphaZeroNet(in_channels=5, num_res_blocks=5, channels=64, board_size=args.width).to(device)
    best_net = AlphaZeroNet(in_channels=5, num_res_blocks=5, channels=64, board_size=args.width).to(device)
    
    trainer = AlphaZeroTrainer(net=latest_net, lr=args.lr, weight_decay=args.weight_decay, device=device)
    
    # Checkpoint resilience
    latest_path = os.path.join(args.checkpoint_dir, "latest_model.pt")
    best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    
    start_iteration = load_checkpoint(latest_path, latest_net, trainer.optimizer)
    if os.path.exists(best_path):
        load_checkpoint(best_path, best_net)
    else:
        print("No best_model.pt found. Copying latest_net weights to best_net.")
        best_net.load_state_dict(latest_net.state_dict())
        save_checkpoint(best_path, best_net, trainer.optimizer, start_iteration)
    
    buffer = ReplayBuffer(capacity=args.buffer_size)
    evaluator = AlphaZeroEvaluator(margin=1.05)
    
    engine_kwargs = {
        "width": args.width,
        "height": args.height,
        "max_steps": args.max_steps,
        "starvation_limit": args.width * args.height * 2,
        "num_simulations": args.num_simulations,
        "num_threads": args.num_threads,
        "c_puct": args.c_puct,
        "discount_factor": 0.99,
        "virtual_loss": 1.0,
        # Match batch_size to num_threads to avoid phantom wait timeout
        "batch_size": args.num_threads,
        "max_batch_wait_us": 500,
    }
    
    for iteration in range(start_iteration, args.num_iterations):
        print(f"\n========== Iteration {iteration+1}/{args.num_iterations} ==========")
        
        # Self-play with latest model
        start_time = time.time()
        print(f"Generating {args.games_per_iter} self-play games with latest_net...")
        latest_net.eval()
        samples = generate_self_play_data(model=latest_net, num_games=args.games_per_iter, engine_kwargs=engine_kwargs)
        buffer.add_game(samples)
        sp_time = time.time() - start_time
        
        avg_score = np.mean([np.arctanh(s.outcome) * 30.0 for s in samples]) if samples else 0.0
        print(f"Self-play done in {sp_time:.2f}s. Buffer size: {len(buffer)}. Avg Score: {avg_score:.2f}")
        writer.add_scalar("self_play/avg_score", avg_score, iteration)
        writer.add_scalar("buffer/size", len(buffer), iteration)
        
        # Train latest model
        if len(buffer) < args.batch_size:
            print("Buffer too small, skipping training...")
            continue
            
        print(f"Training latest_net for {args.epochs_per_iter} epochs...")
        start_time = time.time()
        avg_losses = defaultdict(float)
        for epoch in range(args.epochs_per_iter):
            losses = trainer.train_epoch(buffer, batch_size=args.batch_size)
            for k, v in losses.items():
                avg_losses[k] += v
                
        tr_time = time.time() - start_time
        print(f"Training done in {tr_time:.2f}s.")
        for k, v in avg_losses.items():
            writer.add_scalar(k, v / args.epochs_per_iter, iteration)
            
        # Evaluation Clash
        eval_engine = PyAlphaZeroEngine(**engine_kwargs)
        latest_net.eval()
        best_net.eval()
        
        challenger_wins, champion_score, challenger_score = evaluator.evaluate_challenger(
            champion_net=best_net,
            challenger_net=latest_net,
            engine=eval_engine,
            num_games=args.eval_games
        )
        
        writer.add_scalar("Eval/Challenger_Score", challenger_score, iteration)
        writer.add_scalar("Eval/Champion_Score", champion_score, iteration)
        
        if challenger_wins:
            best_net.load_state_dict(latest_net.state_dict())
            save_checkpoint(best_path, best_net, trainer.optimizer, iteration + 1)
            
        # Always save latest model so we don't lose self-play training progress
        save_checkpoint(latest_path, latest_net, trainer.optimizer, iteration + 1)

    writer.close()
    print("\nAlphaZero continuous training complete!")

if __name__ == "__main__":
    main()
