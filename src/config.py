# Configuration parameters for training
train_config = {
    "project_name": "hw7-cifar10-tinyvgg",
    "run_name": "hw7-cifar10-tinyvgg",
    "checkpoint_dir": "models/checkpoints",
    "epochs": 10, # Change to proper number
    "batch_size": 64,
    "learning_rate": 1e-3,
    "validation_split": 0.2,
    "num_workers": 1
}


# Configuration parameters for TinyVGG model
config_TinyVGG = {
    "input_channels": 3,
    "num_classes": 10,
    "conv1_channels": 10,
    "conv2_channels": 10,
    "kernel_size": 3,
    "padding": 1
}
