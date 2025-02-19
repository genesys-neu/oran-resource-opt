import logging
import numpy as np
from xapp_control import *
from utils.policies.policy_tabular_q import TabularQLearningAgent
from utils.policies.policy_deep_q import DeepQLearningAgent
from utils.policies.policy_deep_q_large import DeepQLearningLargeAgent
from utils.policies.policy_deep_q_v2 import DeepQLearningAgent as DeepQLearningAgent2
from utils.policies.policy_deep_q_large_v2 import DeepQLearningLargeAgent as DeepQLearningLargeAgent2
from utils.policies.policy_tabular_q_v2 import TabularQLearningAgent as TabularQLearningAgent2
from python.ORAN_dataset import *
from python.ORAN_models import ConvNN as global_model
import torch
import pickle


# Dictionary to store KPI history and other relevant data for each UE
ue_data = {}


def initialize_ue_data(imsi):
    """Initialize data structure for a new UE."""
    ue_data[imsi] = {
        'last_timestamp': 0,
        'kpi_slice': 0,
        'kpi_prb': 1,
        'stale_counter': 0,
        'inference_kpi': [],
    }


def truncate_timestamp(timestamp, new_length):
    """Truncate the timestamp to match the new_length."""
    timestamp_str = str(timestamp)
    return int(timestamp_str[-new_length:])


def count_ue_assignments():
    """Count the number of UEs assigned to each slice and return the PRBs for each slice."""
    slice_count = {0: 0, 1: 0, 2: 0}
    slice_prbs = {0: 0, 1: 0, 2: 0}

    for ue in ue_data.values():
        assigned_slice = ue['kpi_slice']
        if assigned_slice in slice_count:
            slice_count[assigned_slice] += 1
            # Only update PRBs if this is the first UE in the slice
            if slice_count[assigned_slice] == 1:
                slice_prbs[assigned_slice] = ue['kpi_prb']

    return slice_count, slice_prbs


def initialize_agent(agent_name):
    logging.info(f'Using {agent_name} Policy')
    if agent_name == "TabularQ_r2_from_TabularQ":
        return TabularQLearningAgent(seed=42, load=True,
                                     load_path_q="utils/policies/q_table_forml2_r2_from_TabularQ.npy")
    elif agent_name == "TabularQ_r2_from_DeepQ":
        return TabularQLearningAgent(seed=42, load=True, load_path_q="utils/policies/q_table_forml2_r2_from_DeepQ.npy")
    elif agent_name == "DeepQ_r2_from_TabularQ":
        return DeepQLearningAgent(seed=42, load=True, load_path_q="utils/policies/dqn_forml2_r2_from_TabularQ.pth")
    elif agent_name == "DeepQ_r2_from_DeepQ":
        return DeepQLearningAgent(seed=42, load=True, load_path_q="utils/policies/dqn_forml2_r2_from_DeepQ.pth")
    elif agent_name == "TabularQ_r3":
        return TabularQLearningAgent(seed=42, load=True,
                                      load_path_q="utils/policies/q_table_forml2_r3_TabularQ.npy")
    elif agent_name == "DeepQ_r3":
        return DeepQLearningAgent(seed=42, load=True,
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
    elif agent_name == "Bellman_r3_large_net_v2":
        agent = DeepQLearningLargeAgent2(seed=42, load=True,
                                         load_path_q="utils/policies/dqn_forml3_r3_large_net_Bellman.pth")
    elif agent_name == "Bellman_r3_TabularQ_v2":
        agent = TabularQLearningAgent2(seed=42, load=True,
                                       load_path_q="utils/policies/q_table_forml3_r3_Bellman.npy")
    else:
        raise ValueError("Unknown agent name provided.")


def initialize_model(Nclass, slice_len, num_feats, torch_model_path):
    logging.info('Initializing ML models...')
    model = global_model(classes=Nclass, slice_len=slice_len, num_feats=num_feats)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.load_state_dict(torch.load(torch_model_path, map_location='cuda:0'))
        logging.info('Using GPU')
    else:
        device = 'cpu'
        model.load_state_dict(torch.load(torch_model_path))
        logging.info('Defaulting to CPU')
    model.to(device)

    # Dummy prediction to validate the model is loaded correctly
    rand_x = torch.Tensor(np.random.random((1, slice_len, num_feats))).to(device)
    pred = model(rand_x)
    logging.debug(f'Dummy slice prediction: {pred.argmax(1).numpy()}')
    return model, device


def setup_logging():
    logging.basicConfig(level=logging.INFO, filename='/home/xapp-logger.log', filemode='a+',
                        format='%(asctime)-15s %(levelname)-8s %(message)s')
    formatter = logging.Formatter('%(asctime)-15s %(levelname)-8s %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def process_line(line):
    if line.startswith('m'):
        line = line[1:]
    kpi_new = np.fromstring(line, sep=',')
    if kpi_new.shape[0] < 31:
        # logging.warning('Discarding KPI: too short')
        return None, None
    # logging.debug(f'Cleaned line: {kpi_new}')
    imsi = int(kpi_new[2])
    return kpi_new, imsi


def classify_traffic(ue, imsi, model, colsparam_dict, current_slice, device):
    """
    Helper function to classify traffic based on inference KPIs and model predictions.

    Parameters:
        ue (dict): Contains user equipment information including inference KPIs.
        imsi (str): The International Mobile Subscriber Identity for the user.
        model (torch.nn.Module): The machine learning model for classification.
        colsparam_dict (dict): Normalization parameters for each KPI column.
        current_slice (int): Current slice assignment if classification is skipped.
        device (torch.device): Device to run the model inference on (e.g., CUDA or CPU).

    Returns:
        str: The updated slice message based on classification.
    """
    logging.debug('Sliding window is full, starting inference')

    # Convert inference KPIs to numpy array and normalize each column
    np_kpi = np.array(ue['inference_kpi'])
    assert np_kpi.shape[1] == len(colsparam_dict), "KPI columns do not match normalization parameters."

    for c in range(np_kpi.shape[1]):
        np_kpi[:, c] = (np_kpi[:, c] - colsparam_dict[c]['min']) / (colsparam_dict[c]['max'] - colsparam_dict[c]['min'])

    # Convert to torch tensor and move to the correct device
    t_kpi = torch.Tensor(np_kpi.reshape(1, np_kpi.shape[0], np_kpi.shape[1])).to(device)

    try:
        # Perform model inference
        pred = model(t_kpi)
        this_class = pred.argmax(1).item()

        # Map prediction to slice assignment
        if this_class == 0:  # the traffic is eMBB
            # ue['kpi_slice'] = 2
            updated_slice_message = f'00{imsi}::2'
        elif this_class == 2:  # the traffic is URLLC
            # ue['kpi_slice'] = 1
            updated_slice_message = f'00{imsi}::1'
        else:  # the traffic is mMTC or cntrl
            # ue['kpi_slice'] = 0
            updated_slice_message = f'00{imsi}::0'
        logging.debug(f'Predicted class: {this_class}')

    except Exception as e:
        logging.error(f'ERROR while predicting class: {e}')
        updated_slice_message = f'00{imsi}::{current_slice}'

    logging.info(f'New slice message: {updated_slice_message}')
    return updated_slice_message


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
            logging.warning(f'Sum of slice PRBs ({total_prbs}) is less than 17. Adjusting slice 1.')
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
        bits_tuple = agent.policy(slice_counts[0], slice_counts[1], slice_counts[2],
                                  slice_prb0, slice_prb1, slice_prb2)
        logging.info(f'New bits_tuple: {bits_tuple}')
        return bits_tuple

    except Exception as e:
        logging.error(f'Error in slice PRB calculation or agent policy call: {e}')
        return (slice_prb0, slice_prb1, slice_prb2)


def update_prbs_and_remove_stale_ues(ue_data, bits_tuple, max_stale_steps):
    """
    Update the Physical Resource Blocks (PRBs) for each UE based on the KPI slice
    and remove stale entries from the UE data.

    Parameters:
        ue_data (dict): Dictionary containing UE data with IMSI as keys.
        bits_tuple (tuple): Tuple containing the bits for each slice.
        max_stale_steps (int): Maximum number of steps a UE can be stale before removal.

    Returns:
        None: The function modifies ue_data in place and logs stale IMSIs.
    """
    stale_imsis = []

    # Iterate over each UE and update the PRBs based on the slice
    # and remove stale entries
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


def main():
    # Configure logger and console output
    setup_logging()

    control_sck = open_control_socket(4200)
    # Round 3: 'TabularQ_r2_from_TabularQ', 'TabularQ_r2_from_DeepQ', 'DeepQ_r2_from_TabularQ', or 'DeepQ_r2_from_DeepQ'
    # Round 4: 'TabularQ_r3' or 'DeepQ_r3' or 'Bellman_r3_TabularQ_interpol' or 'Bellman_r3_DeepQ_no_interpol'
    #          or 'Bellman_r3_large_net_interpol' or 'Bellman_r3_large_net_no_interpol' or 'Bellman_r3_DeepQ_v2'
    #          or 'Bellman_r3_large_net_v2' or 'Bellman_r3_TabularQ_v2'
    agent_name = "TabularQ_r3"
    agent = initialize_agent(agent_name)
    count_pkl = 0
    max_stale_steps = 60
    slice_len, Nclass, num_feats = 32, 4, 17
    torch_model_path = 'model/CNN/model_weights__slice32.pt'
    norm_param_path = 'model/CNN/cols_maxmin.pkl'
    colsparam_dict = pickle.load(open(norm_param_path, 'rb'))
    model, device = initialize_model(Nclass, slice_len, num_feats, torch_model_path)

    print('Start listening on E2 interface...')
    logging.info('Finished initialization')

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
                    # Next we must update the inference_kpi list
                    ue['inference_kpi'].append(kpi_new[np.array(
                        [9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 30])])
                    current_slice = ue['kpi_slice']
                    window_length = len(ue['inference_kpi'])
                    logging.debug(f'updated UE info, current sliding window length is {window_length}')
                    if len(ue['inference_kpi']) > slice_len:
                        ue['inference_kpi'].pop(0)
                        logging.debug(f'Removing first entry in list, length is now {len(ue["inference_kpi"])}')

                    if len(ue['inference_kpi']) == slice_len:
                        updated_slice_message = classify_traffic(ue, imsi, model, colsparam_dict, current_slice, device)
                    else:
                        updated_slice_message = f'00{imsi}::{current_slice}'

                    # After updating ue_data, count UEs assigned to each slice and get PRBs
                    slice_counts, slice_prbs = count_ue_assignments()
                    logging.debug(f'Slice 0: {slice_counts[0]} UEs, PRBs: {slice_prbs[0]}, '
                                  f'Slice 1: {slice_counts[1]} UEs, PRBs: {slice_prbs[1]}, '
                                  f'Slice 2: {slice_counts[2]} UEs, PRBs: {slice_prbs[2]}')

                    # Calculate PRB values per slice
                    slice_prb0 = slice_prbs[0] // 3
                    slice_prb1 = slice_prbs[1] // 3
                    slice_prb2 = (slice_prbs[2] // 3) + (1 if slice_prbs[2] % 3 > 0 else 0)
                    # logging.debug(f'Slice 0: {slice_counts[0]} UEs, PRB bits: {slice_prb0}, '
                    #              f'Slice 1: {slice_counts[1]} UEs, PRB bits: {slice_prb1}, '
                    #              f'Slice 2: {slice_counts[2]} UEs, PRB bits: {slice_prb2}')

                    if count_pkl > 15:
                        bits_tuple = calculate_corrected_slice_prbs(slice_counts, slice_prb0, slice_prb1, slice_prb2,
                                                                    agent, agent_name)

                        # Format the control message with the new PRB assignment
                        # expected control looks like: '1,0,0\n3,5,9\n<imsi>::<slice ID>\n<imsi>::MCS\n<imsi>::gainEND'
                        # scheduling on the first line (0=round-robin, 1=water filling, 2=proportionally fair),
                        # prb assignment to slice on the second line (bits in the mask, total <=17),
                        # UE slice assignment on the third line (slice 0: mmtc, 1: urllc, 2: embb),
                        # MCS adjustment on the fourth line (0=default adaptive modulation, 1=QPSK, 2=16 QAM, 3=64QAM),
                        # Gain (power) adjustment on the last line
                        control_message = f'0,1,2\n{bits_tuple[0]},{bits_tuple[1]},{bits_tuple[2]}\n' \
                                          f'{updated_slice_message}\n\nEND'
                        logging.debug(f'New control message: {control_message}')

                        # Send the control message via the socket
                        send_socket(control_sck, control_message)
                        logging.debug('Sent message')

                        update_prbs_and_remove_stale_ues(ue_data, bits_tuple, max_stale_steps)

                        count_pkl -= 1
                        logging.debug(f'Resenting count_pkl to {count_pkl}')


if __name__ == '__main__':
    main()
