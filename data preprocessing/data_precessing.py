# FICDF's data processor
# author Shengli Ding, assisted by ChatGPT 4o
# This code is designed for IoT-Sentinel

import os
import numpy as np
from scapy.all import rdpcap, TCP, UDP, IP, Ether, Raw
from PIL import Image

#---------------------------------------------- File Path Selections ---------------------------------------------------

root_dir   = 'YOUR_PATH_TO/captures_IoT-Sentinel'
output_dir = 'YOUR_PATH_TO/processedData_IoTSentinel'   # path for saving preprocessed data

#--------------------------------------------- File Path Selections END-------------------------------------------------

# Updated feature ranges based on features available in Scapy and suitable for cross-protocol system
FEATURE_RANGES = {
    'pck_size': (0, 65535),  # Max packet size in bytes
    'ip_src': (0, 2**32-1),  # IP address range
    'ip_dst': (0, 2**32-1),  # IP address range
    'port_class_src': (0, 65535),  # Port number range
    'port_class_dst': (0, 65535),  # Port number range
    'ttl': (0, 255),  # TTL range
    'protocol': (0, 255),  # Protocol number range
    'ethernet_frame_type': (0, 65535),  # Ethernet frame type range
    'pck_rawdata': (0, 1),  # Boolean feature (inferred from payload length)
    'epoch_timestamp': (0, 2**32-1),  # Timestamp range
    'tcp_seq': (0, 2**32-1),  # TCP sequence number
    'tcp_ack': (0, 2**32-1),  # TCP acknowledgment number
    'tcp_flags': (0, 255),  # TCP flags
    'tcp_window': (0, 65535),  # TCP window size
    'udp_length': (0, 65535),  # UDP length
    'udp_checksum': (0, 65535),  # UDP checksum
    'ip_id': (0, 65535),  # IP identification field
    'ip_frag': (0, 8191),  # IP fragment offset
    'ip_checksum': (0, 65535),  # IP header checksum
}

# Calculate bit lengths based on feature ranges
BIT_LENGTHS = [len(bin(v[1])) - 2 for v in FEATURE_RANGES.values()]
# print(BIT_LENGTHS)
# print(len(BIT_LENGTHS))


# Function to convert a value to a binary string of a given length
def to_binary(value, length):
    # If value is an IP address, convert it to a string of digits without dots
    if isinstance(value, str) and '.' in value:
        value = int(value.replace('.', ''))
    # If value is not an integer, convert to int
    if not isinstance(value, int):
        value = int(value)

    if value < 0:
        value = abs(value)
    if 2 ** length <= value:
        value = 2 ** length - 1
    # Convert to binary string with padding
    return format(value, f'0{length}b')

# Function to convert IP address to integer string without dots
def ip_to_int_str(ip):
    return int(ip.replace('.', ''))

# Function to read pcap files and extract packets using Scapy
def read_pcap(file_path):
    packets = rdpcap(file_path)
    return packets

# Function to process a packet and extract features using Scapy
def process_packet(packet):
    features = [
        len(packet),  # pck_size
        ip_to_int_str(packet[IP].src) if IP in packet else 0,  # ip_src
        ip_to_int_str(packet[IP].dst) if IP in packet else 0,  # ip_dst
        packet[TCP].sport if TCP in packet else (packet[UDP].sport if UDP in packet else 0),  # port_class_src
        packet[TCP].dport if TCP in packet else (packet[UDP].dport if UDP in packet else 0),  # port_class_dst
        packet[IP].ttl if IP in packet else 0,  # ttl
        packet[IP].proto if IP in packet else 0,  # protocol
        packet[Ether].type if Ether in packet else 0,  # ethernet_frame_type
        1 if Raw in packet else 0,  # pck_rawdata
        int(packet.time),  # epoch_timestamp
        packet[TCP].seq if TCP in packet else 0,  # tcp_seq
        packet[TCP].ack if TCP in packet else 0,  # tcp_ack
        int(packet[TCP].flags) if TCP in packet else 0,  # tcp_flags
        packet[TCP].window if TCP in packet else 0,  # tcp_window
        packet[UDP].len if UDP in packet else 0,  # udp_length
        packet[UDP].chksum if UDP in packet else 0,  # udp_checksum
        packet[IP].id if IP in packet else 0,  # ip_id
        packet[IP].frag if IP in packet else 0,  # ip_frag
        packet[IP].chksum if IP in packet else 0,  # ip_checksum
    ]
    return features

# Function to convert features to binary and create 19x19 array
def features_to_binary_array(features):
    binary_str = '0'.join(to_binary(f, l) for f, l in zip(features, BIT_LENGTHS)) + '0'
    binary_str = binary_str.ljust(19*19, '0')  # Pad to make sure the string is 256 bits

    binary_array = np.array(list(binary_str), dtype=int).reshape((19, 19))
    return binary_array

# Function to create 16x16x3 samples using a sliding window
def create_samples(packets):
    samples = []
    for i in range(len(packets) - 2):
        sample = np.stack([features_to_binary_array(process_packet(packets[j])) for j in range(i, i + 3)], axis=-1)
        samples.append(sample)
    return samples

# Function to save the samples as images
def save_samples(samples, labels, output_dir, set_name):
    set_dir = os.path.join(output_dir, set_name)
    for i, sample in enumerate(samples):
        label_dir = os.path.join(set_dir, labels[i])
        os.makedirs(label_dir, exist_ok=True)
        image_path = os.path.join(label_dir, f"{i}.png")
        image = Image.fromarray(sample.astype('uint8') * 255)
        image.save(image_path)

# Main preprocessing function
def preprocess_data(root_dir, output_dir):
    all_samples = []
    labels = []
    for device_name in os.listdir(root_dir):
        print(device_name)
        device_dir = os.path.join(root_dir, device_name)
        if os.path.isdir(device_dir):
            mac_address_file = os.path.join(device_dir, '_iotdevice-mac.txt')
            with open(mac_address_file, 'r') as f:
                mac_address = f.read().strip()

            pcap_files = [os.path.join(device_dir, f) for f in os.listdir(device_dir) if
                          f.endswith('.pcap') and 'Setup-C' in f]
            current_all_samples = 0

            for pcap_file in pcap_files:
                packets = read_pcap(pcap_file)
                samples = create_samples(packets)

                current_all_samples = current_all_samples + len(samples)

                if current_all_samples > 500:
                    excess = current_all_samples - 500
                    packets = packets[:-excess]
                    samples = create_samples(packets)

                all_samples.extend(samples)
                labels.extend([device_name] * len(samples))

                if current_all_samples >= 500:
                    break

            # pcap_files = [os.path.join(device_dir, f) for f in os.listdir(device_dir) if f.endswith('.pcap')]
            # for pcap_file in pcap_files:
            #     packets = read_pcap(pcap_file)
            #     samples = create_samples(packets)
            #     all_samples.extend(samples)
            #     labels.extend([device_name] * len(samples))

    # Convert to numpy arrays
    all_samples = np.array(all_samples)
    labels = np.array(labels)

    # Split the data into 15% test, 15% val, and 70% train for each class
    X_train, y_train, X_test, y_test, X_val, y_val = [], [], [], [], [], []

    unique_labels = np.unique(labels)
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        np.random.shuffle(idx)

        n_total = len(idx)
        n_test = int(n_total * 0.15)
        n_val = int(n_total * 0.15)
        n_train = n_total - n_test - n_val

        X_train.extend(all_samples[idx[:n_train]])
        y_train.extend(labels[idx[:n_train]])
        X_test.extend(all_samples[idx[n_train:n_train + n_test]])
        y_test.extend(labels[idx[n_train:n_train + n_test]])
        X_val.extend(all_samples[idx[n_train + n_test:]])
        y_val.extend(labels[idx[n_train + n_test:]])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    # Save the data
    save_samples(X_train, y_train, output_dir, 'train')
    save_samples(X_test, y_test, output_dir, 'test')
    save_samples(X_val, y_val, output_dir, 'val')

# Run the preprocessing
preprocess_data(root_dir=root_dir, output_dir=output_dir)
