#!/usr/bin/env python3
"""
run_experiment.py

Performs the network measurement experiment you described.

Requirements:
    pip install paramiko

Assumptions:
 - SSH access to node1,node2,node3,dag01 with provided credentials (key or password).
 - passwordless sudo on dag01 for dagsnap (or adjust code).
 - NFS-shared home dirs as you described (so node1 can run ITGDec on logs produced by node2/node3).
 - moongen, dagsnap, ITGRecv, ITGSend, ITGDec are in PATH on their respective machines.

Notes:
 - Adjust ITGSend options inside the loop if you need different IDT/packet-size semantics.
 - The script captures PIDs for backgrounded processes and kills them at the end.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from paramiko import Transport, SSHClient, AutoAddPolicy
import socket
import os
import paramiko
import sys
from typing import Optional, Tuple


# Optional: path to paramiko private key file if you prefer that mode (if you placed it in pkey field above)
# ========== END CONFIG ==========



def connect_ssh(host_cfg: dict, port: int = 22) -> paramiko.SSHClient:
    """
    Connect to a host via SSH using Paramiko 4.0.0.

    For 'dag01' (Ubuntu 14 / OpenSSH 6.6), this function:
      - Forces weak/legacy algorithms (DH-group1, AES-CBC)
      - Forces ssh-rsa signing for RSA keys

    For other hosts, it uses secure defaults.
    """

    hostname = host_cfg["host"]
    username = host_cfg["username"]
    password = host_cfg.get("password")
    pkey_path = host_cfg.get("pkey")

    # --- Load private key if provided ---
    pkey_obj: Optional[paramiko.PKey] = None
    if pkey_path:
        if not os.path.exists(pkey_path):
            raise FileNotFoundError(f"Private key not found: {pkey_path}")
        pkey_obj = paramiko.RSAKey.from_private_key_file(pkey_path)
        # Force old-style ssh-rsa signing (critical for Ubuntu14)
        if hasattr(pkey_obj, "use_ssh_rsa_signing"):
            pkey_obj.use_ssh_rsa_signing(True)

    # --- Handle legacy dag01 host separately ---
    if hostname == "dag01":
        # Create a Transport manually so we can override ciphers/kex
        transport = paramiko.Transport((hostname, port), disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']})
        sec_opts = transport.get_security_options()

        # Apply legacy algorithms (compatible with OpenSSH 6.6)
        if hasattr(sec_opts, "kex"):
            sec_opts.kex = [
                "diffie-hellman-group1-sha1",
                "diffie-hellman-group14-sha1",
            ]
        if hasattr(sec_opts, "ciphers"):
            sec_opts.ciphers = [
                "aes128-cbc", "3des-cbc", "aes256-cbc",
                "aes128-ctr", "aes256-ctr",
            ]
        if hasattr(sec_opts, "public_keys"):
            sec_opts.public_keys = ["ssh-rsa"]

        try:
            transport.connect(username=username, password=password, pkey=pkey_obj)
        except Exception as e:
            transport.close()
            raise RuntimeError(f"Legacy connect to {hostname} failed: {e}")

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client._transport = transport
        return client

    # --- Normal path for modern systems ---
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname,
            port=port,
            username=username,
            password=password,
            pkey=pkey_obj,
            look_for_keys=False,
            allow_agent=False,
            timeout=20,
        )
    except Exception:
        client.close()
        raise

    return client


def connect_ssh_orig(host_cfg: dict) -> paramiko.SSHClient:
    """Return a connected Paramiko SSHClient for host_cfg dict {host, username, password, pkey}."""
    if host_cfg['legacy']:
        print("setting legacy transport stuff")
        paramiko.Transport._preferred_kex = [
            'diffie-hellman-group1-sha1', 'diffie-hellman-group14-sha1'
        ]
        paramiko.Transport._preferred_keys = ['ssh-rsa']
        paramiko.Transport._preferred_ciphers = [
            'aes128-cbc', '3des-cbc', 'aes256-cbc', 'aes128-ctr', 'aes256-ctr'
        ]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    pkey = None
    if host_cfg.get("pkey"):
        if os.path.exists(host_cfg["pkey"]):
            try:
                pkey = paramiko.RSAKey.from_private_key_file(host_cfg["pkey"])
            except Exception:
                try:
                    pkey = paramiko.Ed25519Key.from_private_key_file(host_cfg["pkey"])
                except Exception as e:
                    print(f"Unable to load key {host_cfg['pkey']}: {e}")
                    raise
        else:
            raise FileNotFoundError(f"Key file {host_cfg['pkey']} not found")
    try:
        if host_cfg['legacy']:
            print("set legacy stuff AGAIN!")
            paramiko.Transport._preferred_kex = ['diffie-hellman-group1-sha1']
            paramiko.Transport._preferred_keys = ['ssh-rsa']
            paramiko.Transport._preferred_pubkey = ['ssh-rsa']
            paramiko.Transport._preferred_ciphers = ['aes128-cbc', '3des-cbc']
        client.connect(
            hostname=host_cfg["host"],
            username=host_cfg["username"],
            password=host_cfg.get("password"),
            pkey=pkey,
            timeout=20,
        )
    except Exception as e:
        print(f"Error connecting to {host_cfg['host']}: {e}")
        raise
    return client


def connect_legacy_ssh(host_cfg: dict) -> paramiko.SSHClient:
    hostname = host_cfg['host']
    username = host_cfg["username"]
    password = host_cfg["password"]
    pkey = host_cfg['pkey']
    print(f"legacy connect2 to {hostname} as {username} using {pkey}")

    # create socket and transport manually
    sock = paramiko.Transport((hostname, 22))

    # Force weaker algorithms if needed:
    security_opts = sock.get_security_options()
    security_opts.key_exchanges = (
        ['diffie-hellman-group1-sha1', 'diffie-hellman-group14-sha1']
    )
    security_opts.ciphers = (
        ['aes128-cbc', '3des-cbc', 'aes256-cbc', 'aes128-ctr', 'aes256-ctr']
    )
    security_opts.public_keys = ['ssh-rsa']

    print("Negotiation parameters:", security_opts.kex, security_opts.ciphers)

    sock.connect(username=username, password=password)

    # Create SSHClient wrapper around that transport:
    client = paramiko.SSHClient()
    client._transport = sock

    stdin, stdout, stderr = client.exec_command("hostname")
    print(stdout.read().decode())
    client.close()


def connect_legacy_ssh1(host_cfg: dict) -> paramiko.SSHClient:
    host = host_cfg['host']
    username = host_cfg["username"]
    password = host_cfg["password"]
    pkey = host_cfg['pkey']
    print(f"legacy connect1 to {host} as {username} using {pkey}")
    sock = socket.create_connection((host, 22))
    transport = Transport(sock)

    # Force older hostkey and key exchange algorithms
    transport.get_security_options().key_algorithms = [
        "ssh-rsa", "ssh-dss"
    ]
    transport.get_security_options().kex = [
        "diffie-hellman-group1-sha1",
        "diffie-hellman-group-exchange-sha1",
    ]
    transport.get_security_options().ciphers = [
        "aes128-cbc", "3des-cbc", "aes256-cbc"
    ]
    transport.get_security_options().macs = [
        "hmac-sha1", "hmac-md5"
    ]

    transport.start_client()

    if password:
        transport.auth_password(username, password)
    elif pkey:
        transport.auth_publickey(username, pkey)

    client = SSHClient()
    client._transport = transport
    client.set_missing_host_key_policy(AutoAddPolicy())
    return client


def run_command(ssh: paramiko.SSHClient, cmd: str, timeout: Optional[int] = None, get_pty: bool = False) -> Tuple[int, str, str]:
    """
    Run a remote command via SSH. Returns (exit_status, stdout, stderr).
    Note: this will wait until the remote command exits.
    """
    stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=get_pty, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="ignore")
    err = stderr.read().decode("utf-8", errors="ignore")
    exit_status = stdout.channel.recv_exit_status()
    return exit_status, out, err


def run_command_background_and_get_pid(ssh: paramiko.SSHClient, cmd: str) -> int:
    """
    Start cmd under nohup in background on the remote host, return the PID.
    Uses: nohup <cmd> > <log> 2>&1 & echo $!
    """
    # Use a unique log file so we can inspect later. But many commands will provide their own logs.
    # Build a wrapper that prints the PID.
    wrapper = f"nohup sh -c '{cmd}' > /dev/null 2>&1 & echo $!"
    exit_status, out, err = run_command(ssh, wrapper)
    if exit_status != 0:
        raise RuntimeError(f"Failed to start background command: {cmd}\nexit:{exit_status}\nerr:{err}\nout:{out}")
    pid_str = out.strip().splitlines()[-1]
    try:
        pid = int(pid_str)
    except Exception as e:
        raise RuntimeError(f"Couldn't parse PID from output: '{out}' (err:{err})")
    return pid


def stop_pid(ssh: paramiko.SSHClient, pid: int) -> None:
    """Send SIGTERM to pid, then SIGKILL if it doesn't exit after a short wait."""
    try:
        run_command(ssh, f"sudo kill {pid} || true")
    except Exception:
        pass
    # allow grace period
    time.sleep(1)
    # ensure dead
    run_command(ssh, f"sudo kill -0 {pid} >/dev/null 2>&1 || echo 'gone'")
    # if still exists, force kill
    run_command(ssh, f"sudo kill -9 {pid} >/dev/null 2>&1 || true")


def load_emulab_spec(filename):
    """
    Load an Emulab specification from a json file.
    Or use the default.

    :param filename:
    :return:
    """
    em_spec = {
        "node1": {"host": "pc33", "username": "brenton", "password": None, "pkey": "/home/brenton/.ssh/id_rsa",
                  "legacy": False},
        "node2": {"host": "pc93", "username": "brenton", "password": None, "pkey": "/home/brenton/.ssh/id_rsa",
                  "legacy": False},
        "node3": {"host": "pc3", "username": "brenton", "password": None, "pkey": "/home/brenton/.ssh/id_rsa",
                  "legacy": False},
        "dag01": {"host": "dag01", "username": "brenton", "password": None, "pkey": "/home/brenton/.ssh/id_rsa_legacy",
                  "legacy": True},
        'node3_ip': "192.168.1.3",
        'node1_cnet_ip': "130.75.73.172",
        'node3_cnet_ip': "130.75.73.142",
        'pause_seconds': 5,
        'emulab_home': "/users/brenton",
        'dag_home': "/home/brenton",
        'trace_dir': "lemurnn-trace",
        'moongen_interfaces': "3 4",
    }
    if filename:
        with open(filename, 'r') as fh:
            em_spec = json.load(fh)
    else:
        print("Using default Emulab Spec...")
    return em_spec

def load_experiment_spec(filename):
    """
    Load an experiment spec from a json file.

    :param filename:
    :return:
    """
    ex_spec = {
        'min_pkt_size': 60,
        'max_pkt_size': 1400,
        'min_capacity': 1,
        'max_capacity': 10,
        'capacity_increment': 1,
        'min_queue': 5,
        'max_queue': 100,
        'queue_increment': 10,
        'min_latency': 0,
        'max_latency': 0,
        'min_rate': 0.5,
        'max_rate':100,
        'iterations': 32,
        'packet_seq_length': 1024,
        'link_type': 'bytequeue',
    }
    if filename:
        with open(filename, 'r') as fh:
            ex_spec = json.load(fh)
    else:
        print("Using default Experiment Spec...")
    ex_spec['mean_pkt_size'] = (ex_spec['min_pkt_size'] + ex_spec['max_pkt_size']) / 2
    return ex_spec



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-x", '--experiment_spec', type=str, default=None)
    parser.add_argument("-e", '--emulab_spec', type=str, default=None)
    args = parser.parse_args()

    experiment_spec = load_experiment_spec(args.experiment_spec)
    emulab_spec = load_emulab_spec(args.emulab_spec)

    ETIME = int(time.time())
    EMULAB_WORKDIR = f"{emulab_spec['emulab_home']}/{emulab_spec['trace_dir']}/"
    DAG_WORKDIR = f"{emulab_spec['dag_home']}/{emulab_spec['trace_dir']}/"

    print(f"Using paramiko version {paramiko.__version__}")

    # Connect to all hosts
    conns = {}
    HOSTS = {name: emulab_spec[name] for name in ('node1', 'node2', 'node3', 'dag01')}
    try:
        for name, cfg in HOSTS.items():
            print(f"Connecting to {name} ({cfg['host']})...")
            #if cfg['legacy']:
            #    conns[name] = connect_legacy_ssh(cfg)
            #else:
            conns[name] = connect_ssh(cfg)
        print("All SSH connections established.")
    except Exception as e:
        print("Error connecting to hosts:", e)
        for c in conns.values():
            try:
                c.close()
            except Exception:
                pass
        return
    print(f"CONNECTIONS STARTED!!")

    # make main working directories
    run_command(conns["node1"], f"mkdir -p {EMULAB_WORKDIR}", timeout=120)
    run_command(conns["dag01"], f"mkdir -p {DAG_WORKDIR}", timeout=120)

    for CAP in range(experiment_spec['min_capacity'], experiment_spec['max_capacity']+1, experiment_spec['capacity_increment']):

        for QUE in range(experiment_spec['min_queue'], experiment_spec['max_queue']+1, experiment_spec['queue_increment']):

            for LAT in range(experiment_spec['min_latency'], experiment_spec['max_latency']+1):
                # 0) Start ITGRecv on node3 in background
                node3 = conns["node3"]
                itgrecv_log = f"{EMULAB_WORKDIR}/itgrecv_{ETIME}.log"
                itgrecv_cmd = f"ITGRecv"
                print(f"Starting ITGRecv on node3: {itgrecv_cmd}")
                try:
                    itgrecv_pid = run_command_background_and_get_pid(node3, f"{itgrecv_cmd} 2>&1 | tee -a {itgrecv_log}")
                    print(f"ITGRecv started on node3 with PID {itgrecv_pid}")
                except Exception as e:
                    print("Failed to start ITGRecv:", e)
                    itgrecv_pid = None

                # Wait a short moment for daemons to get ready
                time.sleep(2)

                # 1) Start moongen on node2 and keep running
                node2 = conns["node2"]
                moongen_log = f"{EMULAB_WORKDIR}/moongen_C{CAP}_L{LAT}_Q{QUE}_{ETIME}.log"
                #moongen_cmd = f"cd MoonGen ; moongen -r {CAP} -l {LAT} -q {QUE}"
                interfaces = emulab_spec['moongen_interfaces']
                if experiment_spec['link_type'] == 'bytequeue':
                    # for bytequeue, interpret the queue size to be in kiloByte
                    moongen_cmd = f"cd MoonGen ; sudo ./build/MoonGen examples/l2-forward-bsring-lrl.lua -d {interfaces} -r {CAP} {CAP} -l 0 0 -q {QUE*1000} {QUE*1000}"
                elif experiment_spec['link_type'] == 'packetqueue':
                    moongen_cmd = f"cd MoonGen ; sudo ./build/MoonGen examples/l2-forward-psring-lrl.lua -d {interfaces} -r {CAP} {CAP} -l 0 0 -q {QUE} {QUE}"
                else:
                    print(f"ERROR: unknown emulated link type: {experiment_spec['link_type']}")
                    sys.exit(0)
                # run under nohup and capture pid
                print(f"Starting moongen on node2: {moongen_cmd}")
                try:
                    moongen_pid = run_command_background_and_get_pid(node2, f"{moongen_cmd} 2>&1 | tee -a {moongen_log}")
                    print(f"moongen started on node2 with PID {moongen_pid}")
                except Exception as e:
                    print("Failed to start moongen:", e)
                    moongen_pid = None

                time.sleep(10)

                # 2) On node1 send 10 pings to node3
                node1 = conns["node1"]
                print("Running ping from node1 -> node3 (10 packets)...")
                try:
                    exit_status, out, err = run_command(node1, f"ping -c 10 {emulab_spec['node3_ip']}")
                    if exit_status == 0:
                        print("Ping completed.")
                    else:
                        print("Ping returned non-zero status:", exit_status)
                        print("stdout:", out)
                        print("stderr:", err)
                except Exception as e:
                    print("Ping failed:", e)

                # 3) Start dagsnap on dag01 with sudo and keep running
                dag = conns["dag01"]
                dagsnap_out = f"{DAG_WORKDIR}/mgtrace_C{CAP}_L{LAT}_Q{QUE}_{ETIME}.erf"
                # Using 'sudo -n' to avoid waiting for password; if sudo needs password, remove -n and handle accordingly
                dagsnap_cmd = f"sudo -n dagsnap -d0 -o {dagsnap_out}"
                print(f"Starting dagsnap on dag01: {dagsnap_cmd}")
                try:
                    dagsnap_pid = run_command_background_and_get_pid(dag, f"{dagsnap_cmd} 2>&1")
                    print(f"dagsnap started on dag01 with PID {dagsnap_pid}")
                except RuntimeError as e:
                    # Try without -n in case sudo prompt is needed (this will fail if interactive password required).
                    print("Could not start dagsnap with 'sudo -n'. Trying without -n (will fail if sudo needs password).")
                    try:
                        dagsnap_pid = run_command_background_and_get_pid(dag, f"sudo dagsnap -d0 -o {dagsnap_out} 2>&1")
                        print(f"dagsnap started on dag01 with PID {dagsnap_pid}")
                    except Exception as e2:
                        print("Failed to start dagsnap:", e2)
                        dagsnap_pid = None

                # 5) loop for 1024 iterations
                WORK_SUBDIR = f"{EMULAB_WORKDIR}/C{CAP}_L{LAT}_Q{QUE}_{ETIME}"
                run_command(node1, f"mkdir -p {WORK_SUBDIR}", timeout=120)
                for i in range(1, experiment_spec['iterations'] + 1):
                    # pick random rate and convert to pkt/s
                    #RATE_Mbps = random.randint(min_rate, max_rate)
                    RATE_Mbps = random.uniform(experiment_spec['min_rate'], experiment_spec['max_rate'])
                    RATE_pps = int(RATE_Mbps * 1000000 / (experiment_spec['mean_pkt_size'] * 8))
                    print(f"[{i}/{experiment_spec['iterations']}] RATE_Mbps={RATE_Mbps:.4f}  RATE_pps={RATE_pps}")

                    # Compose filenames
                    tx_log = f"{WORK_SUBDIR}/ditg_i{i}_tx_C{CAP}_L{LAT}_Q{QUE}_Rb{RATE_Mbps:.4f}_Rp{RATE_pps}_{ETIME}.itg"
                    rx_log = f"{WORK_SUBDIR}/ditg_i{i}_rx_C{CAP}_L{LAT}_Q{QUE}_Rb{RATE_Mbps:.4f}_Rp{RATE_pps}_{ETIME}.itg"
                    tx_csv = f"{WORK_SUBDIR}/ditg_i{i}_tx_C{CAP}_L{LAT}_Q{QUE}_Rb{RATE_Mbps:.4f}_Rp{RATE_pps}_{ETIME}.csv"
                    rx_csv = f"{WORK_SUBDIR}/ditg_i{i}_rx_C{CAP}_L{LAT}_Q{QUE}_Rb{RATE_Mbps:.4f}_Rp{RATE_pps}_{ETIME}.csv"
                    tx_mat = f"{WORK_SUBDIR}/ditg_i{i}_tx_C{CAP}_L{LAT}_Q{QUE}_Rb{RATE_Mbps:.4f}_Rp{RATE_pps}_{ETIME}.dat"
                    rx_mat = f"{WORK_SUBDIR}/ditg_i{i}_rx_C{CAP}_L{LAT}_Q{QUE}_Rb{RATE_Mbps:.4f}_Rp{RATE_pps}_{ETIME}.dat"

                    # Build ITGSend command.
                    # -z 1024 => send 1024 packets
                    # -T UDP => UDP
                    # -a node3 => destination
                    # -l <sendlog> => write sender-side binary log file
                    # -x <recvlog> => ask receiver to write its log file (receiver will write in its filesystem)
                    # For exponential inter-packet times, the manual uses -B E ... in examples; here we use a simple burst specification:
                    # We'll use "-B E <mean>" where <mean> is derived from RATE (you may need to tune/match manual's expected parameters).
                    # If you prefer to use constant packet rate, change to "-C <pps>" instead.
                    # NOTE: adapt this command to your local D-ITG installation if needed.
                    # compute the desired mean pkt rate based on the other params
                    # (b/s) / ((B/pkt) * (b/B)) = (pkt/s)
                    #pkt_rate = RATE *1000000 / (8*(max_pkt_size + min_pkt_size)/2)
                    itgsend_cmd = (
                        f"ITGSend -a {emulab_spec['node3_ip']} -T UDP -z {experiment_spec['packet_seq_length']} -E {RATE_pps} -u {experiment_spec['min_pkt_size']} {experiment_spec['max_pkt_size']} -l {tx_log} -x {rx_log} -Sda {emulab_spec['node3_cnet_ip']}"
                        #f"ITGSend -a pc33 -T UDP -z 1024 -E {pkt_rate} -u {min_pkt_size} {max_pkt_size} -l {tx_log} -x {rx_log}"
                    )

                    # Run ITGSend on node1 (it will contact ITGRecv on node3 via signaling)
                    # Important: launch in foreground so the script waits for it to finish sending the 1024 packets, then continue.
                    print(f"Running ITGSend on node1: {itgsend_cmd}")
                    try:
                        exit_status, out, err = run_command(node1, itgsend_cmd, timeout=120)
                        if exit_status != 0:
                            print(f"ITGSend returned non-zero status {exit_status}. stderr:\n{err}\nstdout:\n{out}")
                        else:
                            print("ITGSend finished.")
                    except Exception as e:
                        print("Error running ITGSend:", e)
                        # continue to next iteration; logs may still be present.

                    # Pause 5 seconds as requested
                    print(f"Pausing {emulab_spec['pause_seconds']} seconds...")
                    time.sleep(emulab_spec['pause_seconds'])

                    # Convert the log files to CSV/text format using ITGDec on node1
                    # The home dir is shared, so node1 can read both send and receiver logs.
                    # ITGDec -l <txtlog> decodes binary log to text; we'll use that and name it .csv (it's space-separated
                    # but ITGDec's text output can be used as CSV-like). If you want strict CSV, post-process the text output.
                    print("Decoding tx log to text/CSV on node1...")
                    try:
                        dec_tx_cmd = f"ITGDec {tx_log} -l {tx_csv} -o {tx_mat}"
                        exit_status, out, err = run_command(node1, dec_tx_cmd, timeout=60)
                        if exit_status != 0:
                            print(f"ITGDec (tx) returned {exit_status}. stderr:\n{err}\nstdout:\n{out}")
                        else:
                            print(f"Decoded sender log -> {tx_csv}")
                    except Exception as e:
                        print("Error decoding sender log:", e)

                    print("Decoding rx log to text/CSV on node1 (rx log is on node3 but available via NFS)...")
                    try:
                        dec_rx_cmd = f"ITGDec {rx_log} -l {rx_csv} -o {rx_mat}"
                        exit_status, out, err = run_command(node1, dec_rx_cmd, timeout=60)
                        if exit_status != 0:
                            print(f"ITGDec (rx) returned {exit_status}. stderr:\n{err}\nstdout:\n{out}")
                        else:
                            print(f"Decoded receiver log -> {rx_csv}")
                    except Exception as e:
                        print("Error decoding receiver log:", e)

                    # End loop

                print("Main loop complete. Stopping long-running processes...")

                # stop dagsnap on dag01
                try:
                    if dagsnap_pid:
                        print(f"Stopping dagsnap (PID {dagsnap_pid}) on dag01...")
                        stop_pid(dag, dagsnap_pid)
                        run_command(dag, "sudo killall dagsnap || true")
                        run_command(dag, "sudo killall dagsnap || true")
                    else:
                        print("No dagsnap PID recorded; attempting to pkill dagsnap on dag01.")
                        run_command(dag, "sudo pkill -f dagsnap || true")
                        run_command(dag, "sudo killall dagsnap || true")
                        run_command(dag, "sudo killall dagsnap || true")
                except Exception as e:
                    print("Error stopping dagsnap:", e)

                # stop moongen on node2
                try:
                    if moongen_pid:
                        print(f"Stopping moongen (PID {moongen_pid}) on node2...")
                        stop_pid(node2, moongen_pid)
                        run_command(node2, "sudo killall MoonGen || true")
                        run_command(node2, "sudo killall MoonGen || true")
                        time.sleep(5)
                        run_command(node2, "sudo killall MoonGen || true")
                        run_command(node2, "sudo killall MoonGen || true")
                        time.sleep(5)
                    else:
                        print("No moongen PID recorded; attempting to pkill moongen on node2.")
                        run_command(node2, "sudo pkill -f moongen || true")
                        run_command(node2, "sudo killall MoonGen || true")
                        run_command(node2, "sudo killall MoonGen || true")
                        time.sleep(5)
                        run_command(node2, "sudo killall MoonGen || true")
                        run_command(node2, "sudo killall MoonGen || true")
                        time.sleep(5)

                except Exception as e:
                    print("Error stopping moongen:", e)

                # stop ITGRecv on node3
                try:
                    if itgrecv_pid:
                        print(f"Stopping ITGRecv (PID {itgrecv_pid}) on node3...")
                        stop_pid(node3, itgrecv_pid)
                        run_command(node2, "sudo killall ITGRecv || true")
                        run_command(node2, "sudo killall ITGRecv || true")
                        time.sleep(5)
                        run_command(node2, "sudo killall ITGRecv || true")
                        run_command(node2, "sudo killall ITGRecv || true")
                        time.sleep(5)
                    else:
                        print("No ITGRecv PID recorded; attempting to pkill ITGRecv on node3.")
                        run_command(node3, "pkill -f ITGRecv || true")
                        run_command(node2, "sudo killall ITGRecv || true")
                        run_command(node2, "sudo killall ITGRecv || true")
                        time.sleep(5)
                        run_command(node2, "sudo killall ITGRecv || true")
                        run_command(node2, "sudo killall ITGRecv || true")
                        time.sleep(5)
                except Exception as e:
                    print("Error stopping ITGRecv:", e)

    # Close SSH sessions
    for name, sshc in conns.items():
        try:
            sshc.close()
        except Exception:
            pass

    print("Experiment finished. Logs are left in the shared home directories:")
    print(f" - moongen: {moongen_log if 'moongen_log' in locals() else '<unknown>'}")
    print(f" - dagsnap ERF: {dagsnap_out if 'dagsnap_out' in locals() else '<unknown>'}")
    print(f" - ITG logs: ditg_tx_C{CAP}_L{LAT}_Q{QUE}_*.dat and ditg_rx_C{CAP}_L{LAT}_Q{QUE}_*.dat")
    print("You can now analyze the CSV/text files produced by ITGDec on node1.")
    

if __name__ == "__main__":
    main()
