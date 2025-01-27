import logging
import numpy as np
from xapp_control import *
from utils.policies.policy_tabular_q import TabularQLearningAgent
from utils.policies.policy_deep_q import DeepQLearningAgent
from utils.policies.policy_deep_q_large import DeepQLearningLargeAgent
from utils.policies.policy_deep_q_v2 import DeepQLearningAgent as DeepQLearningAgent2


# Dictionary to store KPI history and other relevant data for each UE
ue_data = {}


def initialize_ue_data(imsi):
    """Initialize data structure for a new UE."""
    ue_data[imsi] = {
        'last_timestamp': 0,
        'kpi_slice': 0,
        'kpi_prb': 1,
        'stale_counter': 0,
    }


def truncate_timestamp(timestamp, new_length):
    """Truncate the timestamp to match the new_length."""
    timestamp_str = str(timestamp)
    return int(timestamp_str[-new_length:])


def count_ue_assignments():
    """Count the number of UEs assigned to each slice and return the PRBs for each slice."""
    slice_count = {0: 0, 1: 0, 2: 0}
    slice_prbs = {0: 0, 1: 0, 2: 0}
    slice_latest_timestamp = {0: -1, 1: -1, 2: -1}  # Initialize with -1 to indicate no timestamp yet.

    for ue in ue_data.values():
        assigned_slice = ue['kpi_slice']
        timestamp = ue['last_timestamp']

        if assigned_slice in slice_count:
            slice_count[assigned_slice] += 1

            # Update PRBs if this UE has a more recent timestamp for the slice
            if timestamp > slice_latest_timestamp[assigned_slice]:
                slice_latest_timestamp[assigned_slice] = timestamp
                slice_prbs[assigned_slice] = ue['kpi_prb']

    return slice_count, slice_prbs


def policy(user_tuple, rb_tuple):
    """
    This function returns a RB configuration for the next step given the current RB configuration.
        Args:
            user_tuple: (num_users_mmtc, num_users_urllc, num_users_embb)
            rb_tuple: (num_rb_mmtc, num_rb_urllc, num_rb_embb)
        Returns:
            num_rb_mmtc_next: the number of RBs for mmtc in the next step
            num_rb_urllc_next: the number of RBs for urllc in the next step
            num_rb_embb_next: the number of RBs for embb in the next step
    """

    num_users_mmtc, num_users_urllc, num_users_embb = user_tuple
    num_rb_mmtc, num_rb_urllc, num_rb_embb = rb_tuple

    total_rb = 17

    # Ensure at least 1 RB in each slice
    if num_rb_mmtc == 0:
        num_rb_mmtc = 1
    if num_rb_urllc == 0:
        num_rb_urllc = 1
    if num_rb_embb == 0:
        num_rb_embb = 1

    # Calculate the number of RBs that need to be added or removed
    used_rb = num_rb_mmtc + num_rb_urllc + num_rb_embb
    rb_to_add = total_rb - used_rb

    # Initialize next RB counts
    num_rb_mmtc_next = num_rb_mmtc
    num_rb_urllc_next = num_rb_urllc
    num_rb_embb_next = num_rb_embb

    if rb_to_add == 0:
        # Random change of RB configuration
        action = np.random.choice(7)
        # print(action)
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
                num_rb_embb_next = num_rb_embb
        elif action == 2:
            if num_rb_mmtc >= 2:
                num_rb_mmtc_next = num_rb_mmtc - 1
                num_rb_urllc_next = num_rb_urllc
                num_rb_embb_next = num_rb_embb + 1
        elif action == 3:
            if num_rb_urllc >= 2:
                num_rb_mmtc_next = num_rb_mmtc + 1
                num_rb_urllc_next = num_rb_urllc - 1
                num_rb_embb_next = num_rb_embb
        elif action == 4:
            if num_rb_urllc >= 2:
                num_rb_mmtc_next = num_rb_mmtc
                num_rb_urllc_next = num_rb_urllc - 1
                num_rb_embb_next = num_rb_embb + 1
        elif action == 5:
            if num_rb_embb >= 2:
                num_rb_mmtc_next = num_rb_mmtc + 1
                num_rb_urllc_next = num_rb_urllc
                num_rb_embb_next = num_rb_embb - 1
        elif action == 6:
            if num_rb_embb >= 2:
                num_rb_mmtc_next = num_rb_mmtc
                num_rb_urllc_next = num_rb_urllc + 1
                num_rb_embb_next = num_rb_embb - 1
    else:
        # Add 1 RB to slices with users
        if num_users_urllc > 0 and rb_to_add > 0:
            num_rb_urllc_next += 1
            rb_to_add -= 1
        if num_users_embb > 0 and rb_to_add > 0:
            num_rb_embb_next += 1
            rb_to_add -= 1
        if num_users_mmtc > 0 and rb_to_add > 0:
            num_rb_mmtc_next += 1
            rb_to_add -= 1

    return num_rb_mmtc_next, num_rb_urllc_next, num_rb_embb_next


def process_line(line):
    if line.startswith('m'):
        line = line[1:]
    kpi_new = np.fromstring(line, sep=',')
    if kpi_new.shape[0] < 31:
        # logging.warning('Discarding KPI: too short')
        return None, None
    logging.debug(f'Cleaned line: {kpi_new}')
    imsi = int(kpi_new[2])
    return kpi_new, imsi


def calculate_corrected_slice_prbs(slice_counts, slice_prb0, slice_prb1, slice_prb2, agent, agent_name):
    zero_count = list(slice_counts.values()).count(0)

    if zero_count == 1:
        # Find the key of the slice with a count of 0
        zero_index = next(key for key, count in slice_counts.items() if count == 0)
        prbs_sum_of_others = sum(
            [slice_prb0, slice_prb1, slice_prb2][i] for i in range(3) if i != zero_index
        )
        corrected_slice_prbs = 17 - prbs_sum_of_others

        if zero_index == 0:
            slice_prb0 = corrected_slice_prbs
        elif zero_index == 1:
            slice_prb1 = corrected_slice_prbs
        else:
            slice_prb2 = corrected_slice_prbs

        logging.debug(f'1 missing: corrected slice PRBs: {slice_prb0}, {slice_prb1}, {slice_prb2}')

    elif zero_count == 2:
        non_zero_index = next(key for key, count in slice_counts.items() if count != 0)
        prb_of_other = [slice_prb0, slice_prb1, slice_prb2][non_zero_index]
        prbs_remaining = (17 - prb_of_other) // 2
        remainder = (17 - prb_of_other) % 2

        if non_zero_index == 0:
            slice_prb1 = prbs_remaining
            slice_prb2 = prbs_remaining
        elif non_zero_index == 1:
            slice_prb0 = prbs_remaining
            slice_prb2 = prbs_remaining
        else:
            slice_prb0 = prbs_remaining
            slice_prb1 = prbs_remaining

        if remainder != 0:
            zero_indices = [key for key, count in slice_counts.items() if count == 0]
            if zero_indices:
                slice_index = zero_indices[0]
                if slice_index == 1:
                    slice_prb1 += 1
                elif slice_index == 2:
                    slice_prb2 += 1

        logging.debug(f'2 missing: corrected slice PRBs: {slice_prb0}, {slice_prb1}, {slice_prb2}')

    if slice_prb0 < 1:
        slice_prb0 = 1
    if slice_prb1 < 1:
        slice_prb1 = 1
    if slice_prb2 < 1:
        slice_prb2 =1

    try:
        total_prbs = slice_prb0 + slice_prb1 + slice_prb2
        if total_prbs > 17:
            logging.warning(f'Sum of slice PRBs ({total_prbs}) is greater than 17. Adjusting.')
            diff = total_prbs - 17
            while diff > 0:
                if slice_prb0 > 1:
                    slice_prb0 -= 1
                    diff -= 1
                elif diff >= 2:
                    if slice_prb2 > 1:
                        slice_prb2 -= 1
                        diff -= 1
                    if slice_prb1 > 1:
                        slice_prb1 -= 1
                        diff -= 1
                elif slice_prb2 > 1:
                    slice_prb2 -= 1
                    diff -= 1
                elif slice_prb1 > 1:
                    slice_prb1 -= 1
                    diff -= 1

        elif total_prbs < 17:
            logging.warning(f'Sum of slice PRBs ({total_prbs}) is less than 17. Adjusting slice prbs.')
            diff = 17 - total_prbs
            min_prb_slice = min((slice_prb0, slice_prb1, slice_prb2))
            if slice_prb0 == min_prb_slice:
                slice_prb0 += diff
            elif slice_prb1 == min_prb_slice:
                slice_prb1 += diff
            else:
                slice_prb2 += diff

        logging.info(f'Calling {agent_name} policy with users:'
                     f' {slice_counts[0]}, {slice_counts[1]}, {slice_counts[2]}'
                     f' and with prbs: {slice_prb0}, {slice_prb1}, {slice_prb2}')
        if agent:
            bits_tuple = agent.policy(slice_counts[0], slice_counts[1], slice_counts[2],
                                      slice_prb0, slice_prb1, slice_prb2)
        elif agent_name == 'Original':
            bits_tuple = policy(slice_counts, (slice_prb0, slice_prb1, slice_prb2))
        else:
            bits_tuple = (3, 5, 9)
        logging.info(f'New bits_tuple: {bits_tuple}')
        return bits_tuple

    except Exception as e:
        logging.error(f'Error in slice PRB calculation or agent policy call: {e}')
        return (slice_prb0, slice_prb1, slice_prb2)


def main():
    # configure logger and console output
    logging.basicConfig(level=logging.INFO, filename='/home/xapp-logger.log', filemode='a+',
                        format='%(asctime)-15s %(levelname)-8s %(message)s')
    formatter = logging.Formatter('%(asctime)-15s %(levelname)-8s %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    control_sck = open_control_socket(4200)

    last_timestamp = 0
    curr_timestamp = 0

    print('Start listening on E2 interface...')
    logging.info('Finished initialization')

    count_pkl = 0
    max_stale_steps = 60

    # set agent_name to specify which policy to use:
    # round 1: 'Original', 'Expert'
    # round 2: 'DeepQ', 'TabularQ'
    # round 3: 'TabularQ_r2_from_TabularQ', 'TabularQ_r2_from_DeepQ', 'DeepQ_r2_from_TabularQ', or 'DeepQ_r2_from_DeepQ'
    # round 4: 'TabularQ_r3' or 'DeepQ_r3' or 'Bellman_r3_TabularQ_interpol' or 'Bellman_r3_DeepQ_no_interpol'
    #          or 'Bellman_r3_large_net_interpol' or 'Bellman_r3_large_net_no_interpol' or 'Bellman_r3_DeepQ_v2'
    agent_name = "Bellman_r3_DeepQ_v2"

    logging.info(f'Using {agent_name} Policy')

    if agent_name == "TabularQ_r2_from_TabularQ":
        agent = TabularQLearningAgent(seed=42, load=True,
                                      load_path_q="utils/policies/q_table_forml2_r2_from_TabularQ.npy")
    elif agent_name == "TabularQ_r2_from_DeepQ":
        agent = TabularQLearningAgent(seed=42, load=True,
                                      load_path_q="utils/policies/q_table_forml2_r2_from_DeepQ.npy")
    elif agent_name == "DeepQ_r2_from_TabularQ":
        agent = DeepQLearningAgent(seed=42, load=True,
                                   load_path_q="utils/policies/dqn_forml2_r2_from_TabularQ.pth")
    elif agent_name == "DeepQ_r2_from_DeepQ":
        agent = DeepQLearningAgent(seed=42, load=True,
                                   load_path_q="utils/policies/dqn_forml2_r2_from_DeepQ.pth")
    elif agent_name == "TabularQ_r3":
        agent = TabularQLearningAgent(seed=42, load=True,
                                      load_path_q="utils/policies/q_table_forml2_r3_TabularQ.npy")
    elif agent_name == "DeepQ_r3":
        agent = DeepQLearningAgent(seed=42, load=True,
                                   load_path_q="utils/policies/dqn_forml2_r3_DeepQ.pth")
    elif agent_name == "Bellman_r3_TabularQ_interpol":
        agent = TabularQLearningAgent(seed=42, load=True,
                                      load_path_q="utils/policies/q_table_forml2_r3_Bellman.npy")
    elif agent_name == "Bellman_r3_DeepQ_no_interpol":
        agent = DeepQLearningAgent(seed=42, load=True,
                                   load_path_q="utils/policies/dqn_forml2_r3_Bellman_no_interpol.pth")
    elif agent_name == "Bellman_r3_large_net_interpol":
        agent = DeepQLearningLargeAgent(seed=42, load=True,
                                        load_path_q="utils/policies/dqn_forml2_r3_large_net_Bellman.pth")
    elif agent_name == "Bellman_r3_large_net_no_interpol":
        agent = DeepQLearningLargeAgent(seed=42, load=True,
                                        load_path_q="utils/policies/dqn_forml2_r3_large_net_Bellman_no_interpol.pth")
    elif agent_name == "Bellman_r3_DeepQ_v2":
        agent = DeepQLearningAgent2(seed=42, load=True,
                                    load_path_q="utils/policies/dqn_forml3_r3_Bellman.pth")
    elif agent_name == "DeepQ":
        agent = DeepQLearningAgent(seed=42, load=True,
                                   load_path_q="utils/policies/dqn_forml2.pth")
    elif agent_name == "TabularQ":
        agent = TabularQLearningAgent(seed=42, load=True,
                                      load_path_q="utils/policies/q_table_forml2.npy")
    elif agent_name == "Original":
        agent = ''

    elif agent_name == "Expert":
        agent = ''

    while True:
        data_sck = receive_from_socket(control_sck)
        if len(data_sck) <= 0:
            if len(data_sck) == 0:
                # logging.warning('Socket received 0')
                continue
            else:
                logging.warning('ERROR, negative value for socket - terminating')
                break
        else:
            # logging.debug('Received data: ' + repr(data_sck))
            # with open('/home/kpi_new_log.txt', 'a') as f:
            #     f.write('{}\n'.format(data_sck))

            data_sck = data_sck.replace(',,', ',')
            # Split into individual lines and remove duplicates by converting to a set
            unique_lines = list(set(data_sck.strip().split('\n')))

            for line in unique_lines:
                kpi_new, imsi = process_line(line)
                if kpi_new is None:
                    continue

                if imsi not in ue_data:
                    initialize_ue_data(imsi)

                ue = ue_data[imsi]
                # check the length of the timestamp and fix if necessary
                new_length = len(str(kpi_new[0]))  # Length of the new timestamp
                truncated_last_timestamp = truncate_timestamp(ue['last_timestamp'], new_length)

                curr_timestamp = int(kpi_new[0])

                if curr_timestamp > truncated_last_timestamp:
                    logging.debug(f'Received new KPIs from {imsi} at {curr_timestamp}')
                    count_pkl += 1
                    ue['kpi_slice'] = int(kpi_new[5])
                    ue['kpi_prb'] = int(kpi_new[6])
                    ue['last_timestamp'] = curr_timestamp
                    ue['stale_counter'] = 0

                    # After updating ue_data, count UEs assigned to each slice and get PRBs
                    slice_counts, slice_prbs = count_ue_assignments()
                    logging.debug(f'Slice 0: {slice_counts[0]} UEs, PRBs: {slice_prbs[0]}, '
                                  f'Slice 1: {slice_counts[1]} UEs, PRBs: {slice_prbs[1]}, '
                                  f'Slice 2: {slice_counts[2]} UEs, PRBs: {slice_prbs[2]}')

                    # Calculate PRB values per slice
                    slice_prb0 = slice_prbs[0] // 3
                    slice_prb1 = slice_prbs[1] // 3
                    slice_prb2 = (slice_prbs[2] // 3) + (1 if slice_prbs[2] % 3 > 0 else 0)
                    logging.debug(f'Slice 0: {slice_counts[0]} UEs, PRB bits: {slice_prb0}, '
                                  f'Slice 1: {slice_counts[1]} UEs, PRB bits: {slice_prb1}, '
                                  f'Slice 2: {slice_counts[2]} UEs, PRB bits: {slice_prb2}')

                    if count_pkl > 15:
                        bits_tuple = calculate_corrected_slice_prbs(slice_counts, slice_prb0, slice_prb1, slice_prb2,
                                                                    agent, agent_name)

                        # Format the control message with the new PRB assignment
                        control_message = f'0,1,2\n{bits_tuple[0]},{bits_tuple[1]},{bits_tuple[2]}\n\n\nEND'
                        logging.debug(f'New control message: {control_message}')

                        # Send the control message via the socket
                        send_socket(control_sck, control_message)

                        stale_imsis = []
                        # Iterate over each UE and update the PRBs based on the slice
                        # Remove stale entries
                        for imsi, ue in ue_data.items():
                            slice_id = ue['kpi_slice']
                            if 0 <= slice_id < (len(bits_tuple) - 1):
                                ue['kpi_prb'] = 3 * bits_tuple[slice_id]
                            elif slice_id == (len(bits_tuple) - 1):
                                ue['kpi_prb'] = (3 * bits_tuple[slice_id]) - 1
                            else:
                                # Handle cases where slice_id is out of bounds
                                logging.warning(f'Invalid slice_id {slice_id} for bits_tuple with length {len(bits_tuple)}')
                            ue['stale_counter'] += 1

                            if ue['stale_counter'] > max_stale_steps:
                                logging.info(f'Removing UE {imsi} from ue_data due to inactivity '
                                             f'(stale for {max_stale_steps} steps).')
                                stale_imsis.append(imsi)

                        for imsi in stale_imsis:
                            del ue_data[imsi]  # Remove the stale UE entry

                        logging.debug(f'Updated UE dictionaries: {ue_data}')

                        count_pkl -= len(ue_data)
                        logging.debug(f'Resenting count_pkl to {count_pkl}')


if __name__ == '__main__':
    main()
