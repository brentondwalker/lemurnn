#!/usr/bin/env python3

import json
from collections import namedtuple
import sys
import hashlib
import argparse
import csv
import os
import re

# Try to import pyshark
try:
    import pyshark
except ImportError:
    print("Error: 'pyshark' library not found. Please install it with 'pip install pyshark'", file=sys.stderr)
    sys.exit(1)

# Try to import torch  # <-- ADDED
try:
    import torch
except ImportError:
    print("Error: 'torch' library not found. Please install it with 'pip install torch'", file=sys.stderr)
    sys.exit(1)

# Define the structure for the resulting analysis
PacketRecord = namedtuple('PacketRecord', [
    'packet_number',
    'size',
    'transmit_time',
    'receive_time',
    'latency',
    'dropped_status'
])

# Interface definitions for the A--B--C path we are interested in
TX_INTERFACE = 2  # A -> B (Transmit time)
RX_INTERFACE = 1  # B -> C (Receive time)


def compute_latencies(tx_packets:dict, rx_packets:dict):

    print(f"Found {len(tx_packets)} unique packets transmitted (A->B).")
    print(f"Found {len(rx_packets)} unique packets received (B->C).")

    # 2. Second Pass: Analyze and compile the final records
    analysis_records = []
    packet_counter = 1

    # Iterate through all transmitted packets
    for p_id, tx_data in tx_packets.items():
        size = tx_data['size']
        tx_time = tx_data['tx_time']

        # Check if the packet was successfully observed on the RX link
        if p_id in rx_packets:
            rx_data = rx_packets[p_id]
            rx_time = rx_data['rx_time']

            # Successful transmission (Dropped Status = 0)
            latency = rx_time - tx_time
            dropped_status = 0

            # Ensure latency is non-negative (due to clock sync issues or simulation order)
            # If clocks are perfectly synchronized, this ensures correctness.
            latency = max(0.0, latency)

        else:
            # Packet was dropped at node B (Dropped Status = 1)
            rx_time = 0.0
            latency = 0.0
            dropped_status = 1

        # Create the final record
        record = PacketRecord(
            packet_number=packet_counter,
            size=size,
            transmit_time=tx_time,
            receive_time=rx_time,
            latency=latency,
            dropped_status=dropped_status
        )
        analysis_records.append(record)
        packet_counter += 1

    return analysis_records


def analyze_packet_trace(trace_data):
    """
    Processes a list of packet events from a simulated ERF trace to determine
    latency and dropped status for the A -> B -> C path.

    Args:
        trace_data (list): A list of dictionaries, where each dictionary
                           represents a single packet observation event.

    Returns:
        list: A list of PacketRecord named tuples with the analysis results.
    """
    # Dictionary to store packets observed on the TX link (A->B)
    # Key: Unique packet hash
    # Value: { 'size': int, 'tx_time': float }
    tx_packets = {}

    # Dictionary to store packets observed on the RX link (B->C)
    # Key: Unique packet hash
    # Value: { 'rx_time': float }
    rx_packets = {}

    experiment_traces = []

    print(f"--- Processing {len(trace_data)} Packet Observation Events ---")

    # The capture is actually a series of experiments with 5s pauses between them
    # Therefore, we keep track of the last tx timestamp seen.  If we see a gap
    # of more than 4s, we dump the current results and reinitialize the data structures.
    last_tx_time = 0.0

    # 1. First Pass: Separate and record TX and RX events
    for event in trace_data:
        interface = event['interface']
        packet_id = event['packet_id']
        timestamp = event['timestamp']
        size = event['size']

        # check if this is the start of the next experiment
        if interface == TX_INTERFACE:
            if (timestamp - last_tx_time) > 4.0:
                latency_results = compute_latencies(tx_packets, rx_packets)
                print_results(latency_results)
                experiment_traces.append(latency_results)
                tx_packets = {}
                rx_packets = {}

            # Record the transmit event (A->B)
            if packet_id in tx_packets:
                # Handle a potential rare retransmission scenario, we only care about the first one.
                continue
            tx_packets[packet_id] = {
                'size': size,
                'tx_time': timestamp
            }
            last_tx_time = timestamp

        elif interface == RX_INTERFACE:
            # Record the receive event (B->C)
            if packet_id in rx_packets:
                # Handle a potential rare duplicate capture, we only care about the first one.
                continue
            rx_packets[packet_id] = {
                'rx_time': timestamp
            }

        # Interfaces 3 (C->B) and 4 (B->A) are ignored for this specific A->B->C analysis.

    # Process the final experiment's data after the loop finishes.
    # The loop-based logic only processes a batch when the *next* one begins.
    if tx_packets:
        print("\nProcessing the final experiment batch...")
        latency_results = compute_latencies(tx_packets, rx_packets)
        print_results(latency_results)
        experiment_traces.append(latency_results)
    return experiment_traces

def print_results(records):
    """Prints the final analysis table."""
    print("\n--- FINAL A->B->C ANALYSIS REPORT ---")
    print("-" * 75)
    print(
        f"{'#':<4} | {'Size (B)':<8} | {'TX Time (I1)':<15} | {'RX Time (I2)':<15} | {'Latency (s)':<11} | {'Dropped':<7}")
    print("-" * 75)

    for r in records:
        rx_time_str = f"{r.receive_time:.6f}" if r.receive_time > 0 else "N/A"
        latency_str = f"{r.latency:.6f}" if r.latency > 0 else "N/A"
        dropped_str = "YES" if r.dropped_status == 1 else "NO"

        print(
            f"{r.packet_number:<4} | {r.size:<8} | {r.transmit_time:.6f} | {rx_time_str:<15} | {latency_str:<11} | {dropped_str:<7}"
        )
    print("-" * 75)
    print(f"Total packets analyzed: {len(records)}")
    dropped_count = sum(r.dropped_status for r in records)
    print(f"Total packets dropped (A->B): {dropped_count}")


def parse_filename(filename):  # <-- ADDED
    """
    Parses the filename to extract experiment parameters.
    Expected format: mgtrace_C<C>_L<L>_Q<Q>_...
    Example: "mgtrace_C9_L0_Q9800_1762957671_7.erf"
    Note: This parses the *source* ERF file.
    """
    # Regex to find C, L, and Q parameters
    # It looks for C, L, and Q followed by digits
    match = re.search(r'C(\d+)_L(\d+)_Q(\d+)', filename)

    if match:
        c_val = float(match.group(1))
        l_val = float(match.group(2))
        q_val = float(match.group(3))
        return c_val, l_val, q_val
    else:
        print(f"Warning: Could not parse C,L,Q parameters from filename: {filename}", file=sys.stderr)
        return None, None, None


def save_results_to_csv(records, base_filename, experiment_number):
    """Saves a list of PacketRecord results to a CSV file."""
    if not records:
        print(f"No records to save for experiment {experiment_number}.")
        return

    csv_filename = f"{base_filename}_{experiment_number}.csv"

    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            # Write the header
            writer.writerow(PacketRecord._fields)
            # Write the data rows
            writer.writerows(records)
        print(f"Results for experiment {experiment_number} saved to {csv_filename}")
    except IOError as e:
        print(f"Error: Could not write to file {csv_filename}. Reason: {e}", file=sys.stderr)


def save_as_tensor(experiment_traces, erf_filename, base_output_filename):  # <-- ADDED
    """
    Processes the list of experiment traces, filters/truncates them,
    calculates features, and saves them as a single 3D PyTorch tensor.

    Tensor Shape: (num_valid_experiments, 1024, 7)
    Features: [t, b, s, c, q, l, d]
    - t = inter-packet times
    - b = inter-pkt time * capacity
    - s = packet size
    - c = capacity (C)
    - q = queue (Q)
    - l = baseline latency (L)
    - d = drop status (1 or 0)
    """
    print(f"\n--- Creating PyTorch Tensor File ---")

    # 1. Parse C, L, Q from the *source ERF filename*
    c_val, l_val, q_val = parse_filename(erf_filename)
    if c_val is None:
        print("Error: Could not parse C,L,Q from filename. Aborting tensor creation.", file=sys.stderr)
        return

    print(f"Parsed parameters: C={c_val}, L={l_val}, Q={q_val}")

    all_valid_experiments = []  # This will hold 2D lists

    # 2. Loop through all experiments from this file
    for i, experiment_records in enumerate(experiment_traces):
        exp_num = i + 1

        # 3. Apply filtering rules
        if len(experiment_records) < 1024:
            print(f"  Discarding experiment {exp_num}: too short ({len(experiment_records)} packets < 1024)")
            continue

        # Truncate if longer
        records = experiment_records[:1024]
        print(f"  Processing experiment {exp_num}: {len(records)} packets.")

        # 4. Calculate features for this experiment
        experiment_feature_list = []
        last_tx_time = 0.0

        for j, record in enumerate(records):
            # 4.1 Read data from PacketRecord
            tx_time = record.transmit_time
            size = float(record.size)
            dropped_status = float(record.dropped_status)

            # 4.2 Calculate 't' (inter-packet time)
            if j == 0:
                t = 0.0  # No inter-packet time for the first packet
            else:
                t = tx_time - last_tx_time
            last_tx_time = tx_time

            # 4.3 Calculate 'b' (inter-packet time * capacity)
            b = t * c_val

            # 4.4 Assemble feature vector: [t, b, s, c, q, l, d]
            features = [
                t,
                b,
                size,
                c_val,
                q_val,
                l_val,
                dropped_status
            ]
            experiment_feature_list.append(features)

        all_valid_experiments.append(experiment_feature_list)

    # 5. Convert to a single 3D tensor and save
    if not all_valid_experiments:
        print("No valid experiments (>= 1024 packets) found. No tensor file created.")
        return

    try:
        final_tensor = torch.tensor(all_valid_experiments, dtype=torch.float32)
        output_pt_filename = f"{base_output_filename}.pt"

        torch.save(final_tensor, output_pt_filename)

        print(f"\n--- PyTorch Tensor Creation Complete ---")
        print(f"Successfully saved {len(all_valid_experiments)} valid experiments.")
        print(f"Tensor shape: {final_tensor.shape}")
        print(f"Data saved to {output_pt_filename}")

    except Exception as e:
        print(f"--- Error: Could not save output tensor file to {output_pt_filename}. Reason: {e} ---", file=sys.stderr)


def get_packet_identifier(pkt):
    """
    Returns the IP Identification (ip.id) field as a string.
    The user specified all packets are UDP/IP, so this field should be present
    and stable across the two capture points (A->B and B->C).
    """
    try:
        if 'IP' in pkt:
            # pyshark provides ip.id as a string (e.g., '0x1234' or '4660')
            return pkt.ip.id
    except (AttributeError, TypeError):
        # This can happen if field is missing or packet is not IP
        pass

    # If not an IP packet, we can't get the ID.
    return None


def load_trace_from_erf(filepath):
    """
    Loads an ERF file using pyshark (tshark) and converts it into
    the 'trace_data' list format our analyzer function expects.
    """
    print(f"--- Loading ERF file: {filepath} ---")
    print("This may take a while for large files...")

    trace_data = []
    packet_count = 0
    processed_count = 0

    try:
        # FileCapture uses tshark to read the file
        cap = pyshark.FileCapture(filepath)

        for pkt in cap:
            packet_count += 1
            try:
                # ERF header contains the interface
                if not hasattr(pkt, 'erf'):
                    # Skip packets that don't have an ERF header
                    continue

                # to get the interface we need to look at pkt.erf._all_fields['erf.flags.cap']
                # for come reason we can't access erf.flags.cap directly
                interface = int(pkt.erf._all_fields['erf.flags.cap'], 0)

                # We only care about interfaces 0, 1, 2, 3
                if interface not in [0, 1, 2, 3]:
                    continue

                # Get timestamp (as float) and size
                timestamp = float(pkt.sniff_timestamp)
                size = int(pkt.length)

                # Get the unique packet identifier (IP.ID)
                packet_id = get_packet_identifier(pkt)

                if packet_id:
                    event = {
                        'interface': interface,
                        'timestamp': timestamp,
                        'size': size,
                        'packet_id': packet_id
                    }
                    trace_data.append(event)
                    processed_count += 1

                if packet_count % 10000 == 0:
                    print(f"  ... scanned {packet_count} packets ...")

            except (AttributeError, TypeError, ValueError) as e:
                # Handle packets tshark might not fully dissect (e.g., non-IP)
                # or issues with field extraction
                print(f"Warning: Skipping packet {packet_count}. Error: {e}", file=sys.stderr)
                pass

        cap.close()

    except pyshark.shark.SharkNotInstalled:
        print("--- ERROR ---", file=sys.stderr)
        print("TShark (part of Wireshark) is not installed or not in your system's PATH.", file=sys.stderr)
        print("Please install Wireshark from https://www.wireshark.org/ and try again.", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"--- ERROR: File not found: {filepath} ---", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"--- ERF Load Complete ---")
    print(f"Total packets scanned: {packet_count}")
    print(f"Total packets processed (with valid IP.ID): {processed_count}")
    return trace_data


# --- SIMULATED ERF TRACE DATA ---
# This is no longer used, as we will load from a file.
# SIMULATED_TRACE = [ ... ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analyze A->B->C packet latency and drops from an ERF trace file."
    )
    parser.add_argument(
        "erf_files",  # Changed from "erf_file"
        nargs='+',      # Accept one or more file paths
        help="Path(s) to the .erf packet capture file(s)."
    )
    args = parser.parse_args()

    # Iterate over each file provided on the command line
    total_files = len(args.erf_files)
    for i, erf_file in enumerate(args.erf_files):
        print(f"\n\n{'='*80}")
        print(f"--- Processing File {i+1} of {total_files}: {erf_file} ---")
        print(f"{'='*80}\n")

        # Get the base filename (e.g., "my_trace" from "my_trace.erf")
        base_filename = os.path.splitext(erf_file)[0]

        # 1. Load the trace data from the ERF file
        trace_data = load_trace_from_erf(erf_file)

        if not trace_data:
            print("No processable packets found in the ERF file. Exiting.")
            sys.exit(0)

        # 2. Run the analysis (this function is unchanged)
        results = analyze_packet_trace(trace_data)

        experiment_number = 0
        for trace in results:
            save_results_to_csv(trace, base_filename, experiment_number)
            experiment_number += 1

        print(f"Extracted {len(results)} experiments.")

        # 3. Save the results to a PyTorch tensor file  # <-- ADDED
        save_as_tensor(results, erf_file, base_filename)

    print(f"\n\n{'='*80}")
    print(f"--- All {total_files} files processed. ---")
    print(f"{'='*80}")