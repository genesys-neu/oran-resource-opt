import logging
import time
from xapp_control import *


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
    # slice assignment
    time.sleep(10)
    send_socket(control_sck, "0,1,2\n3,5,9\n001010123456002::1\n001010123456002::3\n20")
    time.sleep(3)
    send_socket(control_sck, "0,1,2\n3,5,9\n001010123456002::0\n001010123456002::2\n20")
    time.sleep(3)
    send_socket(control_sck, "0,1,2\n3,5,9\n001010123456002::2\n001010123456002::3\n20")
    # PRB allocation
    time.sleep(10)
    send_socket(control_sck, "0,1,2\n3,9,5\n001010123456002::2\n001010123456002::3\n20")
    time.sleep(3)
    send_socket(control_sck, "0,1,2\n3,12,2\n001010123456002::2\n001010123456002::3\n20")
    time.sleep(3)
    send_socket(control_sck, "0,1,2\n3,5,9\n001010123456002::2\n001010123456002::3\n20")
    # scheduler
    time.sleep(10)
    send_socket(control_sck, "0,1,1\n3,5,9\n001010123456002::2\n001010123456002::3\n20")
    time.sleep(3)
    send_socket(control_sck, "0,1,0\n3,5,9\n001010123456002::2\n001010123456002::3\n20")
    time.sleep(3)
    send_socket(control_sck, "0,1,2\n3,5,9\n001010123456002::2\n001010123456002::3\n20")
    # Tx gain DL
    time.sleep(10)
    send_socket(control_sck, "0,1,2\n3,5,9\n001010123456002::2\n001010123456002::3\n60")
    time.sleep(3)
    send_socket(control_sck, "0,1,2\n3,5,9\n001010123456002::2\n001010123456002::3\n20")
    # MCS DL
    time.sleep(10)
    send_socket(control_sck, "0,1,2\n3,5,9\n001010123456002::2\n001010123456002::2\n20")
    time.sleep(3)
    send_socket(control_sck, "0,1,2\n3,5,9\n001010123456002::2\n001010123456002::1\n20")
    time.sleep(3)
    send_socket(control_sck, "0,1,2\n3,5,9\n001010123456002::2\n001010123456002::3\n20")
    time.sleep(10)

    # while True:
    #     data_sck = receive_from_socket(control_sck)
    #     if len(data_sck) <= 0:
    #         if len(data_sck) == 0:
    #             continue
    #         else:
    #             logging.info('Negative value for socket')
    #             break
    #     else:
    #         logging.info('Received data: ' + repr(data_sck))


if __name__ == '__main__':
    main()
