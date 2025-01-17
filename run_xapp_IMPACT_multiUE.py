import logging

from torch import nn
from python.ORAN_dataset import *
from python.ORAN_models import ConvNN as global_model
from xapp_control import *


class NeuModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 8)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)

        return x


# Function to normalize data
def normalize(data):
    temp = []
    for moment in data:
        for i in range(len(moment)):
            moment[i] /= maximum_metrics[i]
        temp.append(moment)
    return np.array(temp).flatten()


# Function to make predictions
def predict_newdata(model, input_data):
    if not torch.is_tensor(input_data):
        input_data = torch.tensor(input_data, dtype=torch.float)
        #print(input_data)
    pred = model(input_data)
    val, predicted = torch.max(pred,0)
    predicted_label = predicted.item()
    # if predicted_label == 0:
    #     predicted_label = "No Interference"
    # elif predicted_label == 1:
    #     predicted_label = "Interference"
    # else:
    #     predicted_label = "Control"
    return predicted_label


maximum_metrics = {0: 16.0, 1: 15.0193, 2: 39.6237, 3: 87.5}
BLOCK_SIZE = 5

# Dictionary to store KPI history and other relevant data for each UE
ue_data = {}


def initialize_ue_data(imsi):
    """Initialize data structure for a new UE."""
    ue_data[imsi] = {
        'last_timestamp': 0,
        'kpi': [],
        'kpi_i': [],
        'first_pred_c': False,
        'first_pred_i': False,
        'this_class': None,
    }


def update_kpi_history(ue_data, kpi_filt, kpi_filt_i, slice_len, BLOCK_SIZE):
    """Update the KPI history for a specific UE."""
    if len(ue_data['kpi']) < slice_len:
        ue_data['kpi'].append(kpi_filt)
    else:
        ue_data['kpi'].pop(0)
        ue_data['kpi'].append(kpi_filt)

    if len(ue_data['kpi_i']) < BLOCK_SIZE:
        ue_data['kpi_i'].append(kpi_filt_i)
    else:
        ue_data['kpi_i'].pop(0)
        ue_data['kpi_i'].append(kpi_filt_i)


def main():
    # configure logger and console output
    logging.basicConfig(level=logging.DEBUG, filename='/home/xapp-logger.log', filemode='a+',
                        format='%(asctime)-15s %(levelname)-8s %(message)s')
    formatter = logging.Formatter('%(asctime)-15s %(levelname)-8s %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    control_sck = open_control_socket(4200)

    first_pred_i = False
    first_pred_c = False
    slice_len = 16
    Nclass = 4
    num_feats = 17
    torch_model_path = 'model/CNN/model_weights__slice16.pt'
    norm_param_path = 'model/CNN/cols_maxmin.pkl'
    colsparam_dict = pickle.load(open(norm_param_path, 'rb'))
    # initialize the KPI matrix (4 samples, 19 KPIs each)
    #kpi = np.zeros([slice_len, num_feats])
    kpi = []
    kpi_i = []
    last_timestamp = 0
    curr_timestamp = 0
    this_class = 0
    output = 0

    # initialize the ML model
    print('Init ML models...')
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
    rand_x = torch.Tensor(np.random.random((1, slice_len, num_feats)))
    rand_x = rand_x.to(device)
    pred = model(rand_x)
    print('Dummy slice prediction', pred.argmax(1).numpy())

    model_i = torch.load('model/CNN/best_model_pytorch.pt')
    rand_i = np.random.random(20)
    pred_i = predict_newdata(model_i, rand_i)
    print('Dummy interference prediction', pred_i)

    print('Start listening on E2 interface...')
    logging.info('Finished initialization')

    count_pkl = 0
    cont_data_sck = ""
    isCont = False
    while True:
        data_sck = receive_from_socket(control_sck)
        if len(data_sck) <= 0:
            if len(data_sck) == 0:
                # logging.info('Socket received 0')
                continue
            else:
                logging.info('ERROR, negative value for socket - terminating')
                break
        else:
            # logging.info('Received data: ' + repr(data_sck))
            # with open('/home/kpi_new_log.txt', 'a') as f:
            #     f.write('{}\n'.format(data_sck))

            data_sck = data_sck.replace(',,', ',')
            data_sck = data_sck.split('\n')
            if data_sck[0][0] == 'm':
                data_sck = data_sck[0][1:]
            else:
                data_sck = data_sck[0][0:]

            kpi_new = np.fromstring(data_sck, sep=',')
            if kpi_new.shape[0] < 31:
                # logging.info('Discarding KPI: too short ')
                continue  # discard incomplete KPIs

            # logging.info("Cleaned data: {}".format(data_sck))

            # logging.info("Recieved KPI from {}".format(kpi_new[2]))
            imsi = int(kpi_new[2])

            if imsi not in ue_data:
                initialize_ue_data(imsi)

            ue = ue_data[imsi]
            curr_timestamp = kpi_new[0]

            if curr_timestamp > ue['last_timestamp']:
                kpi_filt = kpi_new[np.array([9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 30])]
                kpi_filt_i = kpi_new[np.array([16, 19, 23, 21])]
                ue['last_timestamp'] = curr_timestamp

                update_kpi_history(ue, kpi_filt, kpi_filt_i, slice_len, BLOCK_SIZE)

                # Traffic class prediction
                if len(ue['kpi']) == slice_len:
                    np_kpi = np.array(ue['kpi'])
                    assert np_kpi.shape[1] == len(list(colsparam_dict.keys()))
                    for c in range(np_kpi.shape[1]):
                        np_kpi[:, c] = (np_kpi[:, c] - colsparam_dict[c]['min']) / (
                                colsparam_dict[c]['max'] - colsparam_dict[c]['min'])
                    t_kpi = torch.Tensor(np_kpi.reshape(1, np_kpi.shape[0], np_kpi.shape[1])).to(device)
                    try:
                        pred = model(t_kpi)
                        ue['this_class'] = float(pred.argmax(1).numpy())
                        ue['first_pred_c'] = True
                    except:
                        logging.info('ERROR while predicting class for UE {}'.format(imsi))

                # Interference prediction
                if len(ue['kpi_i']) == BLOCK_SIZE:
                    np_kpi_i = np.array(ue['kpi_i'])
                    np_kpi_i = normalize(np_kpi_i)
                    try:
                        output = predict_newdata(model_i, np_kpi_i)
                        ue['first_pred_i'] = True
                    except:
                        logging.info('ERROR while predicting interference for UE {}'.format(imsi))

                # expected control looks like: '1,0,0\n5,10,3\n<imsi>::<slice ID>'
                # --> scheduling on the first line, slicing on the second line, UE slice assignment on the third line <imsi>::<slice ID>
                # slice 0: mmtc/cntrl, 1: urllc, 2: embb

                if ue['first_pred_c'] and ue['first_pred_i']:
                    this_class = ue['this_class']
                    if this_class == 0:
                        if output == 0:
                            logging.info(f'embb no interference for UE {imsi}')
                            send_socket(control_sck, f'0,1,2\n3,5,9\n{imsi}::2\n{imsi}::3\n20')
                        elif output == 1:
                            logging.info(f'embb with interference for UE {imsi}')
                            send_socket(control_sck, f'0,1,2\n2,7,8\n{imsi}::2\n{imsi}::0\n60')
                        elif output == 2:
                            logging.info(f'unexpected result: embb and cntrl for UE {imsi}')
                            send_socket(control_sck, f'0,1,2\n3,5,9\n{imsi}::2\n{imsi}::3\n20')
                    elif this_class == 1:
                        if output == 0:
                            logging.info(f'mmtc no interference for UE {imsi}')
                            send_socket(control_sck, f'0,1,2\n3,5,9\n{imsi}::0\n{imsi}::2\n20')
                        elif output == 1:
                            logging.info(f'mmtc with interference for UE {imsi}')
                            send_socket(control_sck, f'0,1,2\n2,7,8\n{imsi}::0\n{imsi}::1\n60')
                        elif output == 2:
                            logging.info(f'unexpected result: mmtc and cntrl for UE {imsi}')
                            send_socket(control_sck, f'0,1,2\n3,5,9\n{imsi}::0\n{imsi}::2\n20')
                    elif this_class == 2:
                        if output == 0:
                            logging.info(f'urllc no interference for UE {imsi}')
                            send_socket(control_sck, f'0,1,2\n3,5,9\n{imsi}::1\n{imsi}::3\n20')
                        elif output == 1:
                            logging.info(f'urllc with interference for UE {imsi}')
                            send_socket(control_sck, f'0,1,2\n2,7,8\n{imsi}::1\n{imsi}::0\n60')
                        elif output == 2:
                            logging.info(f'unexpected result: urllc and cntrl for UE {imsi}')
                            send_socket(control_sck, f'0,1,2\n3,5,9\n{imsi}::1\n{imsi}::3\n20')
                    elif this_class == 3:
                        if output == 0:
                            logging.info(f'unexpected result: cntrl no interference for UE {imsi}')
                            send_socket(control_sck, f'0,1,2\n3,5,9\n{imsi}::0\n{imsi}::2\n20')
                        elif output == 1:
                            logging.info(f'unexpected result: cntrl with interference for UE {imsi}')
                            send_socket(control_sck, f'0,1,2\n2,7,8\n{imsi}::0\n{imsi}::1\n60')
                        elif output == 2:
                            logging.info(f'cntrl for UE {imsi}')
                            send_socket(control_sck, f'0,1,2\n3,5,9\n{imsi}::0\n{imsi}::2\n20')


if __name__ == '__main__':
    main()
