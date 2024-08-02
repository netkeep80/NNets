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

## Usage

1. Setup:

• Clone the repository.
• Make sure you have a C++ compiler installed (e.g., g++).

2. Compile:

• Navigate to the project directory.
• Compile the code using a command like: g++ -o NNets main.cpp

3. Run:

• Execute the compiled binary: ./NNets
• The program will prompt you to enter words.

4. Training:

• The network is automatically trained on a given set of words.
• The training process may take some time depending on the complexity of the data.

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
