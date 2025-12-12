# efficientzero-kinda-model
DeepRL model trained to play efficientzero with MCTS search for Connect4
Working model bomb.pth in repo

4 Models in this model:
worldencoder: Encodes the world into latent channels (64 in this case)
latent2latent: Takes in Latent State, action -> next predicted latent state, game_end (predicts if game ended or no)
policy: AlphaZero policy net trained with crossentropy loss of MCTS search
value: value net trained off temporal difference errors with another target value net for polyak averaging

Special stuff:
Value net is $Q(s, a)$ for Q values, so model can train on any data even random data
Parallised MCTS search: parallel_mcts function in main_model to run mcts on numerous batches at once all on the gpu
