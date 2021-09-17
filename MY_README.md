Replicating SOLOIST:
- updated bash scripts in `soloist/scripts` (`train_multiwoz.sh` and `train_multiwoz_scratch.sh`) for submitting them to slurm and making sure the hyperparameters match those of the paper 
  - e.g. batch per gpu: 6, max length: 512 tokens, num epochs: 10, mc_coefficient=0.33
- Decoding speed is too slow and does not take advantage of batching: 
  - Use`model.generate` function instead 
  - Replace copy of s inside soloist and install transformers with pip (`pip install transformers`), not sure what version it is 
    - fortunately, model weight names didn't change with the version inside soloist and the most udpated version (4.10.2)
  - Loading weights from `gtg_pretrained` needs the local version of transformers inside soloist
    - changed the name of the local dir in `soloist/soloist` to `transformers_local` and updated `sys.path` code in `soloist_train.py`. 
    - Used the updated version `transformers` installed with `pip` for `soloist_decode.py` to use the `generate` function which does batch generation. 


Issues:
- Not able to find how Taskmaster was used for pre-fine-tuning 
- for the model trained from scratch, inference (decoding) takes 4 times longer (checkout the decode logs)
- pereplieReplicating SOLOIST:
- updated bash scripts in `soloist/scripts` (`train_multiwoz.sh` and `train_multiwoz_scratch.sh`) for submitting them to slurm and making sure the hyperparameters match those of the paper 
  - e.g. batch per gpu: 6, max length: 512 tokens, num epochs: 10, mc_coefficient=0.33
- Decoding speed is too slow and does not take advantage of batching: 
  - Use`model.generate` function instead 
  - Replace copy of s inside soloist and install transformers with pip (`pip install transformers`), not sure what version it is 
    - fortunately, model weight names didn't change with the version inside soloist and the most udpated version (4.10.2)
  - Loading weights from `gtg_pretrained` needs the local version of transformers inside soloist
    - changed the name of the local dir in `soloist/soloist` to `transformers_local` and updated `sys.path` code in `soloist_train.py`. 
    - Used the updated version `transformers` installed with `pip` for `soloist_decode.py` to use the `generate` function which does batch generation. 


Issues:
- Not able to find how Taskmaster was used for pre-fine-tuning 
- for the model trained from scratch, inference (decoding) takes 4 times longer (checkout the decode logs)
- pereplieReplicating SOLOIST:
- updated bash scripts in `soloist/scripts` (`train_multiwoz.sh` and `train_multiwoz_scratch.sh`) for submitting them to slurm and making sure the hyperparameters match those of the paper 
  - e.g. batch per gpu: 6, max length: 512 tokens, num epochs: 10, mc_coefficient=0.33
- Decoding speed is too slow and does not take advantage of batching: 
  - Use`model.generate` function instead 
  - Replace copy of s inside soloist and install transformers with pip (`pip install transformers`), not sure what version it is 
    - fortunately, model weight names didn't change with the version inside soloist and the most udpated version (4.10.2)
  - Loading weights from `gtg_pretrained` needs the local version of transformers inside soloist
    - changed the name of the local dir in `soloist/soloist` to `transformers_local` and updated `sys.path` code in `soloist_train.py`. 
    - Used the updated version `transformers` installed with `pip` for `soloist_decode.py` to use the `generate` function which does batch generation. 


Issues:
- Not able to find how Taskmaster was used for pre-fine-tuning 
- for the model trained from scratch, inference (decoding) takes 4 times longer (checkout the decode logs)
- perplexity is lower for model trained from scratch and it gets higher from training, even with the lowest batch. 