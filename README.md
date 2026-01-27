# Self-Structuring Neural Network

This repository contains a C++ implementation of a self-structuring neural network that learns by adding new neurons and connections to its architecture. It operates on the principle of minimizing the error vector's length, with training occurring in three iterative phases:

1. Random Neuron Addition: Randomly adds neurons to the network to avoid getting stuck in local minima.
2. Pseudo-Random Neuron Addition: Adds neurons with partial search, exploring a more targeted space of potential connections.
3. Optimal Neuron Addition: Finds the optimal neuron placement and connection configuration to minimize the error vector's length.

## Features

• Dynamic Structure: The network automatically grows its architecture by adding neurons and connections based on learning requirements.

• Error-Driven Learning: Training is guided by the minimization of the error vector's length, which represents the difference between the network's output and the desired target values.

• Iterative Training: The training process consists of three distinct phases, each contributing to the network's learning.

• C++ Implementation: The project is written in C++, providing performance and control over the underlying data structures.

• Extensible Design: The code is structured to allow users to customize the neuron operations, learning algorithms, and data representations.

## Build

### Using CMake (Recommended)

CMake provides cross-platform build support for Linux, macOS, and Windows.

```bash
# Configure the build
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build the project
cmake --build build --config Release
```

The executable will be located in the `build` directory.

### Using Visual Studio (Windows)

Open `main.sln` in Visual Studio and build the solution.

### Using g++ directly

```bash
g++ -o NNets main.cpp
```

## Usage

The program supports several modes: **Training**, **Retraining**, **Inference**, and **Verification**.

### Command Line Options

```
Usage: ./NNets [options]

MODES:
  Training mode (default): Train network and optionally save to file
  Inference mode: Load trained network and classify inputs
  Retraining mode: Load existing network and continue training with new data

TRAINING OPTIONS:
  -c, --config <file>  Load training configuration from JSON file
  -s, --save <file>    Save trained network to JSON file after training
  -t, --test           Run automated test after training (no interactive mode)
  -b, --benchmark      Run benchmark to measure training speed

RETRAINING OPTIONS:
  -r, --retrain <file> Load existing network and continue training (retraining mode)
                       Combines -l (load) with training mode. Requires -c for new data.
                       New classes in config (without output_neuron) will be trained.

INFERENCE OPTIONS:
  -l, --load <file>    Load trained network from JSON file (inference mode)
  -i, --input <text>   Classify single input text and exit (non-interactive)
  --verify             Verify accuracy of loaded model on training config (-c required)

PERFORMANCE OPTIONS:
  -j, --threads <n>    Number of threads to use (0 = auto, default)
  --single-thread      Disable multithreading (use single thread)

GENERAL OPTIONS:
  -h, --help           Show help message

INTERRUPTION:
  Press Ctrl+C during training to interrupt gracefully.
  The network will be saved if -s is specified.
  Training can be continued later with -r option.
```

### Training Mode

Train a neural network from scratch using a configuration file:

```bash
# Train with default configuration (built-in words: time, hour, main)
./NNets

# Train with custom configuration
./NNets -c configs/default.json

# Train and save the network for later use
./NNets -c configs/default.json -s model.json

# Train with automated testing (no interactive mode)
./NNets -c configs/default.json -s model.json -t

# Benchmark training speed
./NNets -c configs/default.json -b
```

### Inference Mode

Load a pre-trained network and classify inputs:

```bash
# Interactive inference mode
./NNets -l model.json

# Classify a single text (non-interactive, for scripts/tests)
./NNets -l model.json -i "time"
./NNets -l model.json -i "yes"
```

### Retraining Mode

Continue training an existing network with new classes or additional training data:

```bash
# Add new classes to an existing model
# 1. First, train initial model with classes yes/no
./NNets -c configs/simple.json -s model_v1.json

# 2. Create a new config with additional classes (e.g., adding "maybe")
# 3. Retrain the model with new data
./NNets -r model_v1.json -c configs/extended.json -s model_v2.json
```

Retraining automatically detects which classes are already trained (have `output_neuron`) and only trains new classes. This is useful for:
- Adding new recognition classes without retraining from scratch
- Continuing interrupted training sessions
- Incrementally improving the model

### Verification Mode

Check the accuracy of a trained model on test data:

```bash
# Verify model accuracy on training data
./NNets -l model.json -c configs/test.json --verify
```

This mode loads the trained network and tests it against all samples in the configuration file, reporting accuracy statistics.

### Training Interruption

Training can be interrupted at any time by pressing Ctrl+C:
- The first Ctrl+C requests graceful interruption (finishes current iteration)
- The second Ctrl+C forces immediate exit
- If `-s` option is specified, the network state is saved automatically
- Training can be continued later using the `-r` (retrain) option

```bash
# Start long training with auto-save
./NNets -c configs/large.json -s checkpoint.json

# Press Ctrl+C to interrupt...
# Network saved to checkpoint.json

# Continue training later
./NNets -r checkpoint.json -c configs/large.json -s final_model.json
```

### Training Configuration Format

Training configurations are JSON files that define the classes and training images:

```json
{
  "description": "Example configuration",
  "receptors": 20,
  "classes": [
    { "id": 0, "word": "" },
    { "id": 1, "word": "yes" },
    { "id": 2, "word": "no" }
  ],
  "generate_shifts": true
}
```

Configuration options:
- `receptors`: Number of neural network inputs (text length)
- `classes`: Array of classes with unique `id` and `word` to recognize
- `generate_shifts`: If `true`, generates shifted variants of each word for robust recognition
- `description`: Optional description of the configuration

For advanced use, you can specify training images directly:

```json
{
  "receptors": 12,
  "images": [
    { "word": "yes         ", "id": 1 },
    { "word": " yes        ", "id": 1 },
    { "word": "no          ", "id": 2 }
  ]
}
```

### Saved Network Format

Trained networks are saved in JSON format:

```json
{
  "version": "1.0",
  "description": "Trained neural network model",
  "receptors": 12,
  "base_size": 14,
  "inputs": 26,
  "neurons_count": 308,
  "basis": [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, -0.125, -0.25, -0.5, -1.0, -2.0, -4.0, -8.0],
  "classes": [
    { "id": 0, "name": "", "output_neuron": 109 },
    { "id": 1, "name": "yes", "output_neuron": 298 },
    { "id": 2, "name": "no", "output_neuron": 307 }
  ],
  "neurons": [
    { "i": 12, "j": 13, "op": 3 },
    { "i": 5, "j": 26, "op": 1 }
  ]
}
```

Network structure:
- `receptors`: Number of text character inputs
- `basis`: Base values used for neuron operations
- `classes`: Classes the network can recognize, with output neuron indices
- `neurons`: Network structure (neuron IDs are implicit as `inputs + array_index`)

## Example

```
Input word: time
99% - time
1% - hour
0% - main
0% -
Input word: hour
0% - time
98% - hour
0% - main
1% -
```

## License

This project is licensed under the Unlicense. You are free to use, modify, and distribute this software for any purpose without any restrictions.

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please feel free to open an issue or submit a pull request.

## Disclaimer

This project is for educational and experimental purposes. It is not intended for production use or any critical applications. The network's accuracy and performance may vary depending on the dataset and learning parameters.
