import time
import random
import argparse
import numpy as np
from policies.policy_tabular_q import TabularQLearningAgent
from policies.policy_deep_q import DeepQLearningAgent
from utils.policies.policy_deep_q_large import DeepQLearningLargeAgent


def policy(user_tuple, rb_tuple):
    """
    This function returns a RB configuration for the next step given the current RB configuration.
        Args:
            num_users_mmtc: the number of users in mmtc
            num_users_urllc: the number of users in urllc
            num_users_embb: the number of users in embb
            num_rb_mmtc: the number of RBs for mmtc in the current step
            num_rb_urllc: the number of RBs for urllc in the current step
            num_rb_embb: the number of RBs for embb in the current step
        Returns:
            num_rb_mmtc_next: the number of RBs for mmtc in the next step
            num_rb_urllc_next: the number of RBs for urllc in the next step
            num_rb_embb_next: the number of RBs for embb in the next step
    """

    num_users_mmtc, num_users_urllc, num_users_embb = user_tuple
    num_rb_mmtc, num_rb_urllc, num_rb_embb = rb_tuple

    total_rb = 17
    assert num_rb_mmtc + num_rb_urllc + num_rb_embb == total_rb, "The total number of RB of the input should be 17."

    num_rb_mmtc_next = num_rb_mmtc + 0
    num_rb_urllc_next = num_rb_urllc + 0
    num_rb_embb_next = num_rb_embb + 0

    # random change of RB configuration
    action = np.random.choice(7)
    print(action)
    """
    0: keep the current RB configuration
    1: mmtc -> urllc
    2: mmtc -> embb
    3: urllc -> mmtc
    4: urllc -> embb
    5: embb -> mmtc
    6: embb -> urllc
    """
    if action == 1:
        if num_rb_mmtc >= 2:
            num_rb_mmtc_next = num_rb_mmtc - 1
            num_rb_urllc_next = num_rb_urllc + 1
            num_rb_embb_next = num_rb_embb + 0
    elif action == 2:
        if num_rb_mmtc >= 2:
            num_rb_mmtc_next = num_rb_mmtc - 1
            num_rb_urllc_next = num_rb_urllc + 0
            num_rb_embb_next = num_rb_embb + 1
    elif action == 3:
        if num_rb_urllc >= 2:
            num_rb_mmtc_next = num_rb_mmtc + 1
            num_rb_urllc_next = num_rb_urllc - 1
            num_rb_embb_next = num_rb_embb + 0
    elif action == 4:
        if num_rb_urllc >= 2:
            num_rb_mmtc_next = num_rb_mmtc + 0
            num_rb_urllc_next = num_rb_urllc - 1
            num_rb_embb_next = num_rb_embb + 1
    elif action == 5:
        if num_rb_embb >= 2:
            num_rb_mmtc_next = num_rb_mmtc + 1
            num_rb_urllc_next = num_rb_urllc + 0
            num_rb_embb_next = num_rb_embb - 1
    elif action == 6:
        if num_rb_embb >= 2:
            num_rb_mmtc_next = num_rb_mmtc + 0
            num_rb_urllc_next = num_rb_urllc + 1
            num_rb_embb_next = num_rb_embb - 1

    assert num_rb_mmtc_next + num_rb_urllc_next + num_rb_embb_next == total_rb, "The total number of RB of the input should be 17."
    assert num_rb_mmtc_next >= 1, "The number of RB for mmtc in the next step should be greater than or equal to 1"
    assert num_rb_urllc_next >= 1, "The number of RB for urllc in the next step should be greater than or equal to 1"
    assert num_rb_embb_next >= 1, "The number of RB for embb in the next step should be greater than or equal to 1"

    return num_rb_mmtc_next, num_rb_urllc_next, num_rb_embb_next


def generate_random_bits_tuple(total_bits=17, num_slices=3):
    """Generate a random bits_tuple where each slice has at least 1 bit and the sum equals total_bits."""

    if num_slices <= 0:
        raise ValueError("Number of slices must be positive.")

    if total_bits < num_slices:
        raise ValueError("Total bits must be at least as many as the number of slices.")

    # Start by assigning each slice the minimum value of 1
    bits = [1] * num_slices
    remaining_bits = total_bits - num_slices

    # Distribute the remaining bits randomly across slices
    while remaining_bits > 0:
        # Choose a random slice index to add a bit
        slice_index = random.randint(0, num_slices - 1)
        bits[slice_index] += 1
        remaining_bits -= 1

    return tuple(bits)


def generate_bits(bits_tuple, total_bits=25):
    """Generate bitmask strings for three slices based on the input tuple."""

    slice_0_bits, slice_1_bits, slice_2_bits = bits_tuple

    # Create the bitmask for slice 0
    slice_0_mask = ''.join(['1' if i < slice_0_bits else '0' for i in range(total_bits)])

    # Create the bitmask for slice 1 (after the bits used by slice 0)
    slice_1_start = slice_0_bits
    slice_1_mask = ''.join(
        ['1' if slice_1_start <= i < slice_1_start + slice_1_bits else '0' for i in range(total_bits)])

    # Create the bitmask for slice 2 (after the bits used by slice 0 and slice 1)
    slice_2_start = slice_0_bits + slice_1_bits
    slice_2_mask = ''.join(
        ['1' if slice_2_start <= i < slice_2_start + slice_2_bits else '0' for i in range(total_bits)])

    return slice_0_mask, slice_1_mask, slice_2_mask


def update_rb_allocation(slice_0_mask, slice_1_mask, slice_2_mask):
    """Update the RB allocation files with the given bitmasks."""
    # Write the content to the appropriate files
    with open('/root/radio_code/scope_config/slicing/slice_allocation_mask_tenant_0.txt', 'w') as f:
        f.write(slice_0_mask)
    with open('/root/radio_code/scope_config/slicing/slice_allocation_mask_tenant_1.txt', 'w') as f:
        f.write(slice_1_mask)
    with open('/root/radio_code/scope_config/slicing/slice_allocation_mask_tenant_2.txt', 'w') as f:
        f.write(slice_2_mask)

    # print("RB allocation updated")


def main():
    prior_bits_tuple = (3, 5, 9)
    parser = argparse.ArgumentParser(description='RB Allocation Script')
    parser.add_argument('--users', type=int, nargs=3, metavar=('slice_0_users', 'slice_1_users', 'slice_2_users'),
                        help='Number of users assigned to each slice as a tuple (slice_0, slice_1, slice_2)')

    # Add an argument agent_name to specify which policy to use
    parser.add_argument('--agent_name', default="TabularQ_r3",
                        help="The policy to use, 'TabularQ_r2_from_TabularQ', 'TabularQ_r2_from_DeepQ', 'DeepQ_r2_from_TabularQ', 'DeepQ_r2_from_DeepQ', 'TabularQ_r3', 'DeepQ_r3', 'Bellman_r3_TabularQ_interpol', 'Bellman_r3_DeepQ_no_interpol', 'Bellman_r3_large_net_interpol', or 'Bellman_r3_large_net_no_interpol'")

    args = parser.parse_args()

    if args.agent_name == "TabularQ_r2_from_TabularQ":
        agent = TabularQLearningAgent(seed=42, load=True, load_path_q="policies/q_table_forml2_r2_from_TabularQ.npy")
    elif args.agent_name == "TabularQ_r2_from_DeepQ":
        agent = TabularQLearningAgent(seed=42, load=True, load_path_q="policies/q_table_forml2_r2_from_DeepQ.npy")
    elif args.agent_name == "DeepQ_r2_from_TabularQ":
        agent = DeepQLearningAgent(seed=42, load=True, load_path_q="policies/dqn_forml2_r2_from_TabularQ.pth")
    elif args.agent_name == "DeepQ_r2_from_DeepQ":
        agent = DeepQLearningAgent(seed=42, load=True, load_path_q="policies/dqn_forml2_r2_from_DeepQ.pth")
    elif agent_name == "TabularQ_r3":
        agent = TabularQLearningAgent(seed=42, load=True, load_path_q="utils/policies/q_table_forml2_r3_TabularQ.npy")
    elif agent_name == "DeepQ_r3":
        agent = DeepQLearningAgent(seed=42, load=True, load_path_q="utils/policies/dqn_forml2_r3_DeepQ.pth")
    elif agent_name == "Bellman_r3_TabularQ_interpol":
        agent = TabularQLearningAgent(seed=42, load=True, load_path_q="utils/policies/q_table_forml2_r3_Bellman.npy")
    elif agent_name == "Bellman_r3_DeepQ_no_interpol":
        agent = DeepQLearningAgent(seed=42, load=True, load_path_q="utils/policies/dqn_forml2_r3_Bellman_no_interpol.pth")
    elif agent_name == "Bellman_r3_large_net_interpol":
        agent = DeepQLearningLargeAgent(seed=42, load=True,
                                        load_path_q="utils/policies/dqn_forml2_r3_large_net_Bellman.pth")
    elif agent_name == "Bellman_r3_large_net_no_interpol":
        agent = DeepQLearningLargeAgent(seed=42, load=True,
                                        load_path_q="utils/policies/dqn_forml2_r3_large_net_Bellman_no_interpol.pth")

    while True:
        # Example usage: allocate different numbers of bits to each slice
        # bits_tuple = generate_random_bits_tuple()
        
        num_users_mmtc, num_users_urllc, num_users_embb = args.users
        num_rb_mmtc, num_rb_urllc, num_rb_embb = prior_bits_tuple
        bits_tuple = agent.policy(num_users_mmtc, num_users_urllc, num_users_embb, num_rb_mmtc, num_rb_urllc, num_rb_embb)
        # bits_tuple = policy(args.users, prior_bits_tuple)

        # print(bits_tuple)
        slice_0_mask, slice_1_mask, slice_2_mask = generate_bits(bits_tuple)
        #print(slice_0_mask)
        update_rb_allocation(slice_0_mask, slice_1_mask, slice_2_mask)
        prior_bits_tuple = bits_tuple

        # Wait for 1 second before the next update
        time.sleep(0.25)


if __name__ == "__main__":
    main()
