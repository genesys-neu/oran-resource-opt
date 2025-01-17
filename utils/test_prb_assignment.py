import logging
from collections import defaultdict

# Mock function to simulate UE assignments and PRB values
def count_ue_assignments():
    # Return mock values for slice counts and PRBs
    # Example: slice_counts = [2, 1, 0] means 2 UEs on slice 0, 1 UE on slice 1, and 0 UEs on slice 2
    # Example: slice_prbs = [10, 5, 0] means 10 PRBs for slice 0, 5 PRBs for slice 1, and 0 PRBs for slice 2
    slice_counts = [2, 3, 1]
    slice_prbs = [24, 24, 2]
    return slice_counts, slice_prbs

# Mock agent's policy function
class MockAgent:
    def policy(self, slice_count0, slice_count1, slice_count2, prb0, prb1, prb2):
        # Mock output for the policy call
        return (prb0, prb1, prb2)

# Test function
def test_slice_prb_assignment():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')

    agent = MockAgent()
    count_pkl = 16  # Simulate after 16 updates

    # After updating ue_data, count UEs assigned to each slice and get PRBs
    slice_counts, slice_prbs = count_ue_assignments()
    logging.info(f'Slice 0: {slice_counts[0]} UEs, PRBs: {slice_prbs[0]}, '
                 f'Slice 1: {slice_counts[1]} UEs, PRBs: {slice_prbs[1]}, '
                 f'Slice 2: {slice_counts[2]} UEs, PRBs: {slice_prbs[2]}')

    # Calculate PRB values per slice
    slice_prb0 = slice_prbs[0] // 3
    slice_prb1 = slice_prbs[1] // 3
    slice_prb2 = (slice_prbs[2] // 3) + (1 if slice_prbs[2] % 3 > 0 else 0)
    logging.info(f'Slice 0: {slice_counts[0]} UEs, PRB bits: {slice_prb0}, '
                 f'Slice 1: {slice_counts[1]} UEs, PRB bits: {slice_prb1}, '
                 f'Slice 2: {slice_counts[2]} UEs, PRB bits: {slice_prb2}')

    if count_pkl > 15:
        # Check for cases where one or two slice counts are 0
        zero_count = slice_counts.count(0)
        logging.info(f'Total number of slices with 0 UEs: {zero_count}')
        logging.info('Received 16 updates, call policy function')

        if zero_count == 1:
            # Find the index of the slice with a count of 0
            zero_index = slice_counts.index(0)
            # Calculate the sum of the PRBs of the other two slices
            prbs_sum_of_others = sum(
                [slice_prb0, slice_prb1, slice_prb2][i] for i in range(3) if i != zero_index)
            # Assign the remaining PRBs to the slice with a count of 0
            corrected_slice_prbs = 17 - prbs_sum_of_others

            # Update the slice PRBs accordingly
            if zero_index == 0:
                slice_prb0 = corrected_slice_prbs
            elif zero_index == 1:
                slice_prb1 = corrected_slice_prbs
            else:
                slice_prb2 = corrected_slice_prbs

            logging.info(f'1 missing: corrected slice PRBs: {slice_prb0}, {slice_prb1}, {slice_prb2}')

        elif zero_count == 2:
            # Find the index of the slice that is not 0
            non_zero_index = slice_counts.index(next(x for x in slice_counts if x != 0))
            # PRBs assigned to the non-zero slice
            prb_of_other = [slice_prb0, slice_prb1, slice_prb2][non_zero_index]

            # Distribute the remaining PRBs equally among the two zero-count slices
            prbs_remaining = (17 - prb_of_other) // 2
            remainder = (17 - prb_of_other) % 2

            # Set the PRBs for slices with 0 count
            if non_zero_index == 0:
                slice_prb1 = prbs_remaining
                slice_prb2 = prbs_remaining
            elif non_zero_index == 1:
                slice_prb0 = prbs_remaining
                slice_prb2 = prbs_remaining
            else:
                slice_prb0 = prbs_remaining
                slice_prb1 = prbs_remaining

            # Adjust if there is a remainder
            if remainder != 0:
                # Increase PRB for the first zero-count slice in the order of slice_counts
                zero_indices = [i for i, count in enumerate(slice_counts) if count == 0]
                if zero_indices:
                    slice_index = zero_indices[0]  # Take the first zero-count slice
                    if slice_index == 1:
                        slice_prb1 += 1
                    elif slice_index == 2:
                        slice_prb2 += 1

            logging.info(f'2 missing: adjusted slice PRBs: {slice_prb0}, {slice_prb1}, {slice_prb2}')

        # Call the policy function with individual slice counts and PRBs
        bits_tuple = agent.policy(slice_counts[0], slice_counts[1], slice_counts[2], slice_prb0, slice_prb1, slice_prb2)
        logging.info(f'New bits_tuple: {bits_tuple}')

        # Format the control message with the new PRB assignment
        control_message = f'\n{bits_tuple[0]},{bits_tuple[1]},{bits_tuple[2]}'
        logging.info(f'New control message: {control_message}')

# Run the test function
if __name__ == "__main__":
    test_slice_prb_assignment()
