import torch
import numpy as np
import numpy.typing as npt

# uv run pytest -k test_get_batch
def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    upper_bound = len(dataset) - context_length
    # print(dataset)
    input = []
    output = []
    for _ in range(batch_size):
        start = np.random.randint(0, upper_bound)
        input.append(dataset[start: (start + context_length)])
        output.append(dataset[(start + 1): (start + context_length + 1)])
    return (torch.tensor(np.array(input), device = torch.device(device)),
            torch.tensor(np.array(output), device = torch.device(device)))