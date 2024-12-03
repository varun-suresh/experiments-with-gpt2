from dataclasses import dataclass

@dataclass
class EvalConfig:

    # Device params
    device: str = "cuda"
    compile: bool = False

    # I/O
    results_path: str = "results.txt"
    
    # Dataset 
    batch_size: int = 2
    subset: bool = True # Runs only on a subset of the split. To quickly evaluate whether the model is working as expected
    interval: int = 100 # If subset is true, then pick one sample out of every interval(100 by default) samples.
    
