import json
from multiprocessing import Pool
import os
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def generate_binary_string():
    return [random.choice([0, 1]) for _ in range(4)]

def identity_kernel():
    return np.eye(4)

def flip_all_kernel():
    return np.ones((4, 4)) - np.eye(4)

def shift_left_kernel():
    return np.roll(np.eye(4), -1, axis=1)

def shift_right_kernel():
    return np.roll(np.eye(4), 1, axis=1)

def mirror_kernel():
    return np.fliplr(np.eye(4))

def swap_middle_kernel():
    kernel = np.eye(4)
    kernel[1:3, 1:3] = np.fliplr(np.eye(2))
    return kernel

def apply_kernel(binary_string, kernel):
    input_array = np.array(binary_string)
    output_array = np.dot(kernel, input_array) % 2
    return output_array.tolist()

def test_kernels(seed=42):
    set_seed(seed)
    test_inputs = [
        [1, 0, 1, 0],
        [0, 1, 1, 1],
        [1, 1, 0, 0],
        [0, 0, 0, 1]
    ]

    kernels = {
        "IDENTITY": identity_kernel(),
        "FLIP_ALL": flip_all_kernel(),
        "SHIFT_LEFT": shift_left_kernel(),
        "SHIFT_RIGHT": shift_right_kernel(),
        "MIRROR": mirror_kernel(),
        "SWAP_MIDDLE": swap_middle_kernel()
    }

    inverse_kernels = {
        "IDENTITY": identity_kernel(),
        "FLIP_ALL": flip_all_kernel(),
        "SHIFT_LEFT": shift_right_kernel(),  # Inverse of SHIFT_LEFT is SHIFT_RIGHT
        "SHIFT_RIGHT": shift_left_kernel(),  # Inverse of SHIFT_RIGHT is SHIFT_LEFT
        "MIRROR": mirror_kernel(),
        "SWAP_MIDDLE": swap_middle_kernel()
    }

    def assert_operation(name, kernel, inverse_kernel, input_string):
        output = apply_kernel(input_string, kernel)
        inverse_output = apply_kernel(output, inverse_kernel)
        assert inverse_output == input_string, f"{name} inverse failed: input {input_string}, got {inverse_output}"
        print(f"{name}: passed")

    for name, kernel in kernels.items():
        inverse_kernel = inverse_kernels[name]
        for input_string in test_inputs:
            assert_operation(name, kernel, inverse_kernel, input_string)

    print("All kernel tests passed!")


def generate_example(seed=None):
    if seed is not None:
        set_seed(seed)
    
    kernels = {
        "IDENTITY": identity_kernel,
        "FLIP_ALL": flip_all_kernel,
        "SHIFT_LEFT": shift_left_kernel,
        "SHIFT_RIGHT": shift_right_kernel,
        "MIRROR": mirror_kernel,
        "SWAP_MIDDLE": swap_middle_kernel
    }
    
    rule = random.choice(list(kernels.keys()))
    kernel = kernels[rule]()
    
    num_train = random.randint(2, 4)
    num_test = random.randint(1, 2)
    
    train_examples = []
    for _ in range(num_train):
        input_string = generate_binary_string()
        output_string = apply_kernel(input_string, kernel)
        train_examples.append({"input": input_string, "output": output_string})
    
    test_examples = []
    for _ in range(num_test):
        input_string = generate_binary_string()
        output_string = apply_kernel(input_string, kernel)
        test_examples.append({"input": input_string, "output": output_string})
    
    return {
        'train': train_examples,
        'test': test_examples,
        'kernel': kernel.tolist(),
        'rule': rule
    }

def generate_single_challenge(args):
    is_train, output_dir, _ = args
    challenge = generate_example()
    challenge_filename = os.path.join(output_dir, f"challenge_{random.randint(0, 999999):06d}.json")
    with open(challenge_filename, "w") as f:
        json.dump(challenge, f, indent=4)
    return challenge_filename

def generate_challenge_batch(batch_args):
    return [generate_single_challenge(args) for args in batch_args]

def generate_fewshot_challenges(num_challenges, train_ratio, output_dir):
    num_train_challenges = int(num_challenges * train_ratio)
    num_eval_challenges = num_challenges - num_train_challenges

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_dir = os.path.join(output_dir, "training")
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Prepare arguments for multiprocessing
    train_args = [(True, train_dir, i) for i in range(num_train_challenges)]
    eval_args = [(False, eval_dir, i) for i in range(num_eval_challenges)]
    all_args = train_args + eval_args

    # Split arguments into batches of 10
    batch_size = 10
    arg_batches = [all_args[i:i + batch_size] for i in range(0, len(all_args), batch_size)]

    # Use multiprocessing to generate challenges in batches
    with Pool() as pool:
        results = pool.map(generate_challenge_batch, arg_batches)

    # Flatten the results
    challenge_files = [file for batch in results for file in batch]
    
    print(f"Generated {len(challenge_files)} challenge files in {output_dir}")


if __name__ == "__main__":
    print("Generating ONE MILLION challenges. This will take a couple minutes...")
    num_challenges = 1000000
    train_ratio = 0.95
    output_dir = "bitdata"
    test_kernels()
    generate_fewshot_challenges(num_challenges, train_ratio, output_dir)
    print(f"Binary pattern challenges generated and saved in {output_dir}")
    