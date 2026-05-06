import argparse
import os
import torch
import time
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from alpha_zero.alpha_zero_net import AlphaZeroNet
from alpha_zero.replay_buffer import ReplayBuffer
from alpha_zero.trainer import AlphaZeroTrainer
from alpha_zero.self_play import generate_self_play_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--games-per-iter", type=int, default=100)
    parser.add_argument("--epochs-per-iter", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-dir", type=str, default="./models/alphazero")
    parser.add_argument("--log-dir", type=str, default="./tensorboard/alphazero")
    
    # Engine specific kwargs
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--num-simulations", type=int, default=100)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--c-puct", type=float, default=1.0)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    net = AlphaZeroNet(
        in_channels=4, 
        num_res_blocks=5, 
        channels=64, 
        board_size=args.width
    )
    
    trainer = AlphaZeroTrainer(
        net=net, 
        lr=args.lr, 
        weight_decay=args.weight_decay, 
        device=device
    )
    
    buffer = ReplayBuffer(capacity=args.buffer_size)
    
    engine_kwargs = {
        "width": args.width,
        "height": args.height,
        "max_steps": args.max_steps,
        "num_simulations": args.num_simulations,
        "num_threads": args.num_threads,
        "c_puct": args.c_puct,
        "discount_factor": 0.99,
        "virtual_loss": 1.0,
        "batch_size": 32,
        "max_batch_wait_us": 5000,
    }
    
    total_games = 0
    total_epochs = 0
    
    for iteration in range(args.num_iterations):
        print(f"\n--- Iteration {iteration+1}/{args.num_iterations} ---")
        
        # Phase G: Self-play
        start_time = time.time()
        print(f"Generating {args.games_per_iter} self-play games...")
        net.eval()
        samples = generate_self_play_data(
            model=net,
            num_games=args.games_per_iter,
            engine_kwargs=engine_kwargs
        )
        
        sp_time = time.time() - start_time
        buffer.add_game(samples)
        total_games += args.games_per_iter
        
        # Log self play stats
        avg_score = np.mean([np.arctanh(s.outcome) * 30.0 for s in samples]) if samples else 0.0
        avg_len = len(samples) / args.games_per_iter if args.games_per_iter > 0 else 0.0
        
        print(f"Self-play done in {sp_time:.2f}s. Avg Score: {avg_score:.2f}, Avg Len: {avg_len:.2f}")
        writer.add_scalar("self_play/avg_score", avg_score, iteration)
        writer.add_scalar("self_play/avg_game_length", avg_len, iteration)
        writer.add_scalar("buffer/size", len(buffer), iteration)
        
        # Phase H: Train
        if len(buffer) < args.batch_size:
            print("Buffer too small, skipping training...")
            continue
            
        print(f"Training for {args.epochs_per_iter} epochs...")
        start_time = time.time()
        
        avg_losses = defaultdict(float)
        for epoch in range(args.epochs_per_iter):
            losses = trainer.train_epoch(buffer, batch_size=args.batch_size)
            for k, v in losses.items():
                avg_losses[k] += v
            total_epochs += 1
                
        tr_time = time.time() - start_time
        print(f"Training done in {tr_time:.2f}s.")
        
        for k, v in avg_losses.items():
            writer.add_scalar(k, v / args.epochs_per_iter, iteration)
            print(f"  {k}: {v / args.epochs_per_iter:.4f}")
            
        # Checkpoint
        model_path = os.path.join(args.save_dir, f"model_iter_{iteration:03d}.pt")
        torch.save(net.state_dict(), model_path)
        print(f"Saved checkpoint to {model_path}")
        
    writer.close()
    print("\nAlphaZero training complete!")

import numpy as np

if __name__ == "__main__":
    main()
