import json
import os
import random
from multiprocessing import Pool

def generate_binary_string():
    return [random.choice([0, 1]) for _ in range(4)]

def apply_flip_rule(binary_string, position):
    transformed = binary_string.copy()
    transformed[position] = 1 - transformed[position]  # Flip the bit
    return transformed

def generate_example(position=None):
    if position is None:
        position = random.randint(0, 3)  # Choose a position to flip if not provided
    training_examples = []
    for _ in range(3):
        input_string = generate_binary_string()
        output_string = apply_flip_rule(input_string, position)
        training_examples.append({"input": input_string, "output": output_string})
    test_input = generate_binary_string()
    test_output = apply_flip_rule(test_input, position)
    return {
        'train': training_examples,
        'test': {'input': test_input, 'output': test_output}
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
    num_challenges = 1000000
    train_ratio = 0.95
    output_dir = "bitdata"
    generate_fewshot_challenges(num_challenges, train_ratio, output_dir)
    print(f"Binary pattern challenges generated and saved in {output_dir}")
