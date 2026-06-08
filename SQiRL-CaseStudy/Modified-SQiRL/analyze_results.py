#!/usr/bin/env python3
"""
Script to analyze SQIRL test results from result_stats_1.stats files.
Calculates mean and standard deviation of win rate and episode length
per input endpoint, separated by training type (sanitized vs non-sanitized).

Outputs:
- Terminal table with color coding
- CSV file
- LaTeX table
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def extract_endpoint(url):
    """Extract the endpoint name from a URL (e.g., sqli7.php from full URL)"""
    match = re.search(r'(sqli\d+\.php)', url)
    if match:
        return match.group(1)
    return url.split('/')[-1]


def parse_result_stats(filepath):
    """
    Parse a result_stats_1.stats file and extract per-input statistics.

    Returns a dict: {endpoint: {'won': bool, 'episodes': int, 'timestamps': list}}
    """
    results = {}
    current_input = None
    current_endpoint = None
    episode_count = 0
    timestamps_list = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Check for starting/changing input
        start_match = re.match(r'\[!\] Starting to input (\w+),(.*)', line)
        change_match = re.match(r'\[!\] Changing to input (\w+),(.*)', line)

        if start_match or change_match:
            # Save previous input results if any
            if current_endpoint and episode_count > 0:
                if current_endpoint not in results:
                    results[current_endpoint] = {
                        'won': False,
                        'episodes': episode_count,
                        'timestamps': timestamps_list.copy()
                    }

            # Start tracking new input
            match = start_match or change_match
            param = match.group(1)
            url = match.group(2)
            current_endpoint = extract_endpoint(url)
            current_input = f"{param},{current_endpoint}"
            episode_count = 0
            timestamps_list = []
            continue

        # Check for episode results
        ep_match = re.match(r'ep (\d+): current game (lost|wins) (\d+) with (\d+) timestamps', line)
        if ep_match:
            episode_num = int(ep_match.group(1))
            result = ep_match.group(2)
            count = int(ep_match.group(3))
            timestamps = int(ep_match.group(4))
            episode_count = count  # This is the cumulative count for current input
            timestamps_list.append(timestamps)
            continue

        # Check for win/loss criteria
        if '[!] reached win criteria' in line:
            if current_endpoint:
                results[current_endpoint] = {
                    'won': True,
                    'episodes': episode_count,
                    'timestamps': timestamps_list.copy()
                }
                # Reset for next input
                episode_count = 0
                timestamps_list = []
            continue

        if '[!] reached loss criteria' in line:
            if current_endpoint:
                results[current_endpoint] = {
                    'won': False,
                    'episodes': episode_count,
                    'timestamps': timestamps_list.copy()
                }
                # Reset for next input
                episode_count = 0
                timestamps_list = []
            continue

    return results


def collect_all_results(test_results_dir):
    """
    Collect results from all test directories, separated by training type.

    Returns: {
        'non_san': {endpoint: [{'won': bool, 'episodes': int, 'timestamps': list}, ...]},
        'sanitized': {endpoint: [...]}
    }
    """
    all_results = {
        'non_san': defaultdict(list),
        'sanitized': defaultdict(list)
    }

    test_dir = Path(test_results_dir)

    for subdir in sorted(test_dir.iterdir()):
        if not subdir.is_dir():
            continue

        # Determine training type from directory name
        dirname = subdir.name
        if 'TEST_non_san_' in dirname:
            training_type = 'non_san'
        elif 'TEST_sanitized_' in dirname:
            training_type = 'sanitized'
        else:
            continue

        # Find and parse result_stats file
        result_file = subdir / 'result_stats_1.stats'
        if not result_file.exists():
            print(f"Warning: No result_stats_1.stats in {subdir.name}")
            continue

        try:
            results = parse_result_stats(result_file)
            for endpoint, data in results.items():
                all_results[training_type][endpoint].append(data)
        except Exception as e:
            print(f"Error parsing {result_file}: {e}")

    return all_results


def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate the 95% confidence interval for the mean.

    Returns: (mean, ci_lower, ci_upper, margin_of_error)
    """
    n = len(data)
    if n < 2:
        mean = np.mean(data) if n > 0 else 0.0
        return mean, mean, mean, 0.0

    mean = np.mean(data)
    std_err = scipy_stats.sem(data)  # Standard error of the mean

    # Use t-distribution for small samples
    t_value = scipy_stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_of_error = t_value * std_err

    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error

    return mean, ci_lower, ci_upper, margin_of_error


def calculate_statistics(results_list, confidence=0.95):
    """
    Calculate mean and 95% confidence interval for win rate and episode length.

    Returns: {
        'win_rate_mean': float,
        'win_rate_ci': float,  # margin of error for 95% CI
        'win_rate_ci_lower': float,
        'win_rate_ci_upper': float,
        'episodes_mean': float,
        'episodes_ci': float,  # margin of error for 95% CI
        'episodes_ci_lower': float,
        'episodes_ci_upper': float,
        'count': int
    }
    """
    if not results_list:
        return {
            'win_rate_mean': 0.0,
            'win_rate_ci': 0.0,
            'win_rate_ci_lower': 0.0,
            'win_rate_ci_upper': 0.0,
            'episodes_mean': 0.0,
            'episodes_ci': 0.0,
            'episodes_ci_lower': 0.0,
            'episodes_ci_upper': 0.0,
            'count': 0
        }

    wins = [1 if r['won'] else 0 for r in results_list]
    episodes = [r['episodes'] for r in results_list]

    # Calculate confidence intervals
    win_mean, win_ci_lower, win_ci_upper, win_ci = calculate_confidence_interval(wins, confidence)
    ep_mean, ep_ci_lower, ep_ci_upper, ep_ci = calculate_confidence_interval(episodes, confidence)

    return {
        'win_rate_mean': win_mean * 100,  # Convert to percentage
        'win_rate_ci': win_ci * 100,
        'win_rate_ci_lower': win_ci_lower * 100,
        'win_rate_ci_upper': win_ci_upper * 100,
        'episodes_mean': ep_mean,
        'episodes_ci': ep_ci,
        'episodes_ci_lower': ep_ci_lower,
        'episodes_ci_upper': ep_ci_upper,
        'count': len(results_list)
    }


def print_results_table(all_results):
    """Print a formatted table of results."""

    # Collect all unique endpoints
    all_endpoints = set()
    for training_type in all_results:
        all_endpoints.update(all_results[training_type].keys())

    # Sort endpoints by number
    def endpoint_sort_key(ep):
        match = re.search(r'sqli(\d+)', ep)
        return int(match.group(1)) if match else 999

    sorted_endpoints = sorted(all_endpoints, key=endpoint_sort_key)

    # Calculate statistics for each endpoint and training type
    stats = {
        'non_san': {},
        'sanitized': {}
    }

    for training_type in ['non_san', 'sanitized']:
        for endpoint in sorted_endpoints:
            results_list = all_results[training_type].get(endpoint, [])
            stats[training_type][endpoint] = calculate_statistics(results_list)

    # Define sanitisation subset endpoints
    sanitisation_endpoints = ['sqli14.php', 'sqli15.php', 'sqli16.php', 'sqli17.php',
                              'sqli18.php', 'sqli26.php', 'sqli27.php', 'sqli28.php', 'sqli29.php']

    # Calculate totals
    totals = {}
    sanitisation_totals = {}

    for training_type in ['non_san', 'sanitized']:
        all_wins = []
        all_episodes = []
        san_wins = []
        san_episodes = []

        for endpoint in sorted_endpoints:
            results_list = all_results[training_type].get(endpoint, [])
            for r in results_list:
                all_wins.append(1 if r['won'] else 0)
                all_episodes.append(r['episodes'])

                # Check if this endpoint is in the sanitisation subset
                if endpoint in sanitisation_endpoints:
                    san_wins.append(1 if r['won'] else 0)
                    san_episodes.append(r['episodes'])

        if all_wins:
            win_mean, win_ci_lower, win_ci_upper, win_ci = calculate_confidence_interval(all_wins)
            ep_mean, ep_ci_lower, ep_ci_upper, ep_ci = calculate_confidence_interval(all_episodes)
            totals[training_type] = {
                'win_rate_mean': win_mean * 100,
                'win_rate_ci': win_ci * 100,
                'episodes_mean': ep_mean,
                'episodes_ci': ep_ci,
                'count': len(all_wins)
            }
        else:
            totals[training_type] = {
                'win_rate_mean': 0.0,
                'win_rate_ci': 0.0,
                'episodes_mean': 0.0,
                'episodes_ci': 0.0,
                'count': 0
            }

        # Calculate sanitisation subset totals
        if san_wins:
            san_win_mean, _, _, san_win_ci = calculate_confidence_interval(san_wins)
            san_ep_mean, _, _, san_ep_ci = calculate_confidence_interval(san_episodes)
            sanitisation_totals[training_type] = {
                'win_rate_mean': san_win_mean * 100,
                'win_rate_ci': san_win_ci * 100,
                'episodes_mean': san_ep_mean,
                'episodes_ci': san_ep_ci,
                'count': len(san_wins)
            }
        else:
            sanitisation_totals[training_type] = {
                'win_rate_mean': 0.0,
                'win_rate_ci': 0.0,
                'episodes_mean': 0.0,
                'episodes_ci': 0.0,
                'count': 0
            }

    # Print header
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}=" * 120 + Colors.END)
    print(f"{Colors.BOLD}{Colors.CYAN}SQIRL Test Results Analysis - Comparison by Training Type (95% CI){Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}=" * 120 + Colors.END)
    print()

    # Print table header
    header = f"{'Endpoint':<15} | {'Training Type':<12} | {'Win Rate (%) 95% CI':^22} | {'Episodes 95% CI':^22} | {'N':>5}"
    subheader = f"{'':<15} | {'':<12} | {'[Lower':>10}, {'Upper]':<10} | {'[Lower':>10}, {'Upper]':<10} | {'':<5}"

    print(f"{Colors.BOLD}{header}{Colors.END}")
    print(f"{Colors.BOLD}{subheader}{Colors.END}")
    print("-" * 120)

    # Print per-endpoint results
    for endpoint in sorted_endpoints:
        for i, training_type in enumerate(['non_san', 'sanitized']):
            s = stats[training_type].get(endpoint, calculate_statistics([]))

            if i == 0:
                ep_name = endpoint
            else:
                ep_name = ""

            # Color code based on training type
            type_color = Colors.YELLOW if training_type == 'non_san' else Colors.GREEN

            # Color code win rate
            if s['win_rate_mean'] >= 80:
                win_color = Colors.GREEN
            elif s['win_rate_mean'] >= 50:
                win_color = Colors.YELLOW
            else:
                win_color = Colors.RED

            type_display = "Non-Sanitized" if training_type == 'non_san' else "Sanitized"

            # Format as [lower, upper]
            win_ci_str = f"[{s['win_rate_ci_lower']:>5.1f}, {s['win_rate_ci_upper']:<5.1f}]"
            ep_ci_str = f"[{s['episodes_ci_lower']:>5.2f}, {s['episodes_ci_upper']:<5.2f}]"

            print(f"{ep_name:<15} | {type_color}{type_display:<12}{Colors.END} | "
                  f"{win_color}{win_ci_str:^22}{Colors.END} | "
                  f"{ep_ci_str:^22} | "
                  f"{s['count']:>5}")

        print("-" * 120)

    # Print sanitisation subset totals
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}SANITISATION SUBSET (SQLi-14,15,16,17,18,26,27,28,29){Colors.END}")
    print("-" * 120)

    for training_type in ['non_san', 'sanitized']:
        t = sanitisation_totals[training_type]
        type_color = Colors.YELLOW if training_type == 'non_san' else Colors.GREEN

        if t['win_rate_mean'] >= 80:
            win_color = Colors.GREEN
        elif t['win_rate_mean'] >= 50:
            win_color = Colors.YELLOW
        else:
            win_color = Colors.RED

        type_display = "Non-Sanitized" if training_type == 'non_san' else "Sanitized"

        # Calculate bounds from mean and CI
        win_lower = t['win_rate_mean'] - t['win_rate_ci']
        win_upper = t['win_rate_mean'] + t['win_rate_ci']
        ep_lower = t['episodes_mean'] - t['episodes_ci']
        ep_upper = t['episodes_mean'] + t['episodes_ci']

        win_ci_str = f"[{win_lower:>5.1f}, {win_upper:<5.1f}]"
        ep_ci_str = f"[{ep_lower:>5.2f}, {ep_upper:<5.2f}]"

        print(f"{'SANITISATION':<15} | {type_color}{type_display:<12}{Colors.END} | "
              f"{win_color}{win_ci_str:^22}{Colors.END} | "
              f"{ep_ci_str:^22} | "
              f"{t['count']:>5}")

    # Print totals
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}TOTALS (Aggregated Across All Endpoints){Colors.END}")
    print("-" * 120)

    for training_type in ['non_san', 'sanitized']:
        t = totals[training_type]
        type_color = Colors.YELLOW if training_type == 'non_san' else Colors.GREEN

        if t['win_rate_mean'] >= 80:
            win_color = Colors.GREEN
        elif t['win_rate_mean'] >= 50:
            win_color = Colors.YELLOW
        else:
            win_color = Colors.RED

        type_display = "Non-Sanitized" if training_type == 'non_san' else "Sanitized"

        # Calculate bounds from mean and CI
        win_lower = t['win_rate_mean'] - t['win_rate_ci']
        win_upper = t['win_rate_mean'] + t['win_rate_ci']
        ep_lower = t['episodes_mean'] - t['episodes_ci']
        ep_upper = t['episodes_mean'] + t['episodes_ci']

        win_ci_str = f"[{win_lower:>5.1f}, {win_upper:<5.1f}]"
        ep_ci_str = f"[{ep_lower:>5.2f}, {ep_upper:<5.2f}]"

        print(f"{'TOTAL':<15} | {type_color}{type_display:<12}{Colors.END} | "
              f"{win_color}{win_ci_str:^22}{Colors.END} | "
              f"{ep_ci_str:^22} | "
              f"{t['count']:>5}")

    print("=" * 120)
    print()

    # Print summary statistics
    print(f"{Colors.BOLD}Summary:{Colors.END}")
    print(f"  Non-Sanitized trained agents: {totals['non_san']['count']} total observations across {len([e for e in sorted_endpoints if all_results['non_san'].get(e)])} endpoints")
    print(f"  Sanitized trained agents:     {totals['sanitized']['count']} total observations across {len([e for e in sorted_endpoints if all_results['sanitized'].get(e)])} endpoints")
    print()

    # Count number of test runs
    non_san_runs = len(set(len(all_results['non_san'].get(ep, [])) for ep in sorted_endpoints if all_results['non_san'].get(ep)))
    san_runs = len(set(len(all_results['sanitized'].get(ep, [])) for ep in sorted_endpoints if all_results['sanitized'].get(ep)))

    non_san_count = max(len(all_results['non_san'].get(ep, [])) for ep in sorted_endpoints) if sorted_endpoints else 0
    san_count = max(len(all_results['sanitized'].get(ep, [])) for ep in sorted_endpoints) if sorted_endpoints else 0

    print(f"  Number of Non-Sanitized agent test runs: ~{non_san_count}")
    print(f"  Number of Sanitized agent test runs:     ~{san_count}")
    print()


def create_results_dataframe(all_results):
    """
    Create a pandas DataFrame with all results using 95% confidence intervals.

    Returns a DataFrame with columns for each training type's statistics.
    """
    # Collect all unique endpoints
    all_endpoints = set()
    for training_type in all_results:
        all_endpoints.update(all_results[training_type].keys())

    def endpoint_sort_key(ep):
        match = re.search(r'sqli(\d+)', ep)
        return int(match.group(1)) if match else 999

    sorted_endpoints = sorted(all_endpoints, key=endpoint_sort_key)

    # Build data for DataFrame
    data = []
    for endpoint in sorted_endpoints:
        row = {'Endpoint': endpoint}

        for training_type in ['non_san', 'sanitized']:
            results_list = all_results[training_type].get(endpoint, [])
            s = calculate_statistics(results_list)

            prefix = 'NonSan' if training_type == 'non_san' else 'San'
            row[f'{prefix}_WinRate_Mean'] = s['win_rate_mean']
            row[f'{prefix}_WinRate_CI'] = s['win_rate_ci']
            row[f'{prefix}_WinRate_CI_Lower'] = s['win_rate_ci_lower']
            row[f'{prefix}_WinRate_CI_Upper'] = s['win_rate_ci_upper']
            row[f'{prefix}_Episodes_Mean'] = s['episodes_mean']
            row[f'{prefix}_Episodes_CI'] = s['episodes_ci']
            row[f'{prefix}_Episodes_CI_Lower'] = s['episodes_ci_lower']
            row[f'{prefix}_Episodes_CI_Upper'] = s['episodes_ci_upper']
            row[f'{prefix}_N'] = s['count']

        data.append(row)

    # Define sanitisation subset endpoints
    sanitisation_endpoints = ['sqli14.php', 'sqli15.php', 'sqli16.php', 'sqli17.php',
                              'sqli18.php', 'sqli26.php', 'sqli27.php', 'sqli28.php', 'sqli29.php']

    # Add sanitisation subset row
    san_row = {'Endpoint': 'SANITISATION'}
    for training_type in ['non_san', 'sanitized']:
        san_wins = []
        san_episodes = []
        for endpoint in sorted_endpoints:
            if endpoint in sanitisation_endpoints:
                results_list = all_results[training_type].get(endpoint, [])
                for r in results_list:
                    san_wins.append(1 if r['won'] else 0)
                    san_episodes.append(r['episodes'])

        prefix = 'NonSan' if training_type == 'non_san' else 'San'
        if san_wins:
            win_mean, win_ci_lower, win_ci_upper, win_ci = calculate_confidence_interval(san_wins)
            ep_mean, ep_ci_lower, ep_ci_upper, ep_ci = calculate_confidence_interval(san_episodes)
            san_row[f'{prefix}_WinRate_Mean'] = win_mean * 100
            san_row[f'{prefix}_WinRate_CI'] = win_ci * 100
            san_row[f'{prefix}_WinRate_CI_Lower'] = win_ci_lower * 100
            san_row[f'{prefix}_WinRate_CI_Upper'] = win_ci_upper * 100
            san_row[f'{prefix}_Episodes_Mean'] = ep_mean
            san_row[f'{prefix}_Episodes_CI'] = ep_ci
            san_row[f'{prefix}_Episodes_CI_Lower'] = ep_ci_lower
            san_row[f'{prefix}_Episodes_CI_Upper'] = ep_ci_upper
            san_row[f'{prefix}_N'] = len(san_wins)
        else:
            san_row[f'{prefix}_WinRate_Mean'] = 0.0
            san_row[f'{prefix}_WinRate_CI'] = 0.0
            san_row[f'{prefix}_WinRate_CI_Lower'] = 0.0
            san_row[f'{prefix}_WinRate_CI_Upper'] = 0.0
            san_row[f'{prefix}_Episodes_Mean'] = 0.0
            san_row[f'{prefix}_Episodes_CI'] = 0.0
            san_row[f'{prefix}_Episodes_CI_Lower'] = 0.0
            san_row[f'{prefix}_Episodes_CI_Upper'] = 0.0
            san_row[f'{prefix}_N'] = 0

    data.append(san_row)

    # Add totals row
    totals_row = {'Endpoint': 'TOTAL'}
    for training_type in ['non_san', 'sanitized']:
        all_wins = []
        all_episodes = []
        for endpoint in sorted_endpoints:
            results_list = all_results[training_type].get(endpoint, [])
            for r in results_list:
                all_wins.append(1 if r['won'] else 0)
                all_episodes.append(r['episodes'])

        prefix = 'NonSan' if training_type == 'non_san' else 'San'
        if all_wins:
            win_mean, win_ci_lower, win_ci_upper, win_ci = calculate_confidence_interval(all_wins)
            ep_mean, ep_ci_lower, ep_ci_upper, ep_ci = calculate_confidence_interval(all_episodes)
            totals_row[f'{prefix}_WinRate_Mean'] = win_mean * 100
            totals_row[f'{prefix}_WinRate_CI'] = win_ci * 100
            totals_row[f'{prefix}_WinRate_CI_Lower'] = win_ci_lower * 100
            totals_row[f'{prefix}_WinRate_CI_Upper'] = win_ci_upper * 100
            totals_row[f'{prefix}_Episodes_Mean'] = ep_mean
            totals_row[f'{prefix}_Episodes_CI'] = ep_ci
            totals_row[f'{prefix}_Episodes_CI_Lower'] = ep_ci_lower
            totals_row[f'{prefix}_Episodes_CI_Upper'] = ep_ci_upper
            totals_row[f'{prefix}_N'] = len(all_wins)
        else:
            totals_row[f'{prefix}_WinRate_Mean'] = 0.0
            totals_row[f'{prefix}_WinRate_CI'] = 0.0
            totals_row[f'{prefix}_WinRate_CI_Lower'] = 0.0
            totals_row[f'{prefix}_WinRate_CI_Upper'] = 0.0
            totals_row[f'{prefix}_Episodes_Mean'] = 0.0
            totals_row[f'{prefix}_Episodes_CI'] = 0.0
            totals_row[f'{prefix}_Episodes_CI_Lower'] = 0.0
            totals_row[f'{prefix}_Episodes_CI_Upper'] = 0.0
            totals_row[f'{prefix}_N'] = 0

    data.append(totals_row)

    df = pd.DataFrame(data)
    return df


def create_latex_table(all_results, output_file='results_table.tex'):
    """
    Create a LaTeX table from results using pandas.

    Generates a nicely formatted LaTeX table with mean ± 95% CI format.
    """
    # Collect all unique endpoints
    all_endpoints = set()
    for training_type in all_results:
        all_endpoints.update(all_results[training_type].keys())

    def endpoint_sort_key(ep):
        match = re.search(r'sqli(\d+)', ep)
        return int(match.group(1)) if match else 999

    sorted_endpoints = sorted(all_endpoints, key=endpoint_sort_key)

    # Build data for formatted LaTeX table
    data = []
    for endpoint in sorted_endpoints:
        row = {'Endpoint': endpoint.replace('.php', '').replace('sqli', 'SQLi-')}

        for training_type in ['non_san', 'sanitized']:
            results_list = all_results[training_type].get(endpoint, [])
            s = calculate_statistics(results_list)

            col_name = 'Non-Sanitized' if training_type == 'non_san' else 'Sanitized'

            # Format as [lower, upper] for 95% CI
            row[f'{col_name} Win Rate (\\%)'] = f"$[{s['win_rate_ci_lower']:.1f}, {s['win_rate_ci_upper']:.1f}]$"
            row[f'{col_name} Episodes'] = f"$[{s['episodes_ci_lower']:.2f}, {s['episodes_ci_upper']:.2f}]$"
            row[f'{col_name} N'] = int(s['count'])

        data.append(row)

    # Define sanitisation subset endpoints
    sanitisation_endpoints = ['sqli14.php', 'sqli15.php', 'sqli16.php', 'sqli17.php',
                              'sqli18.php', 'sqli26.php', 'sqli27.php', 'sqli28.php', 'sqli29.php']

    # Add sanitisation subset row
    san_row = {'Endpoint': '\\textbf{Sanitisation}'}
    for training_type in ['non_san', 'sanitized']:
        san_wins = []
        san_episodes = []
        for endpoint in sorted_endpoints:
            if endpoint in sanitisation_endpoints:
                results_list = all_results[training_type].get(endpoint, [])
                for r in results_list:
                    san_wins.append(1 if r['won'] else 0)
                    san_episodes.append(r['episodes'])

        col_name = 'Non-Sanitized' if training_type == 'non_san' else 'Sanitized'
        if san_wins:
            _, win_ci_lower, win_ci_upper, _ = calculate_confidence_interval(san_wins)
            _, ep_ci_lower, ep_ci_upper, _ = calculate_confidence_interval(san_episodes)
            win_ci_lower *= 100
            win_ci_upper *= 100
            n = len(san_wins)
        else:
            win_ci_lower = win_ci_upper = ep_ci_lower = ep_ci_upper = 0.0
            n = 0

        san_row[f'{col_name} Win Rate (\\%)'] = f"$\\mathbf{{[{win_ci_lower:.1f}, {win_ci_upper:.1f}]}}$"
        san_row[f'{col_name} Episodes'] = f"$\\mathbf{{[{ep_ci_lower:.2f}, {ep_ci_upper:.2f}]}}$"
        san_row[f'{col_name} N'] = n

    data.append(san_row)

    # Add totals row
    totals_row = {'Endpoint': '\\textbf{TOTAL}'}
    for training_type in ['non_san', 'sanitized']:
        all_wins = []
        all_episodes = []
        for endpoint in sorted_endpoints:
            results_list = all_results[training_type].get(endpoint, [])
            for r in results_list:
                all_wins.append(1 if r['won'] else 0)
                all_episodes.append(r['episodes'])

        col_name = 'Non-Sanitized' if training_type == 'non_san' else 'Sanitized'
        if all_wins:
            _, win_ci_lower, win_ci_upper, _ = calculate_confidence_interval(all_wins)
            _, ep_ci_lower, ep_ci_upper, _ = calculate_confidence_interval(all_episodes)
            win_ci_lower *= 100
            win_ci_upper *= 100
            n = len(all_wins)
        else:
            win_ci_lower = win_ci_upper = ep_ci_lower = ep_ci_upper = 0.0
            n = 0

        totals_row[f'{col_name} Win Rate (\\%)'] = f"$\\mathbf{{[{win_ci_lower:.1f}, {win_ci_upper:.1f}]}}$"
        totals_row[f'{col_name} Episodes'] = f"$\\mathbf{{[{ep_ci_lower:.2f}, {ep_ci_upper:.2f}]}}$"
        totals_row[f'{col_name} N'] = n

    data.append(totals_row)

    df = pd.DataFrame(data)

    # Generate LaTeX table
    latex_str = df.to_latex(
        index=False,
        escape=False,  # Don't escape LaTeX commands
        column_format='l' + 'r' * (len(df.columns) - 1),
        caption='SQIRL Test Results: Comparison of Agents Trained on Sanitized vs Non-Sanitized Endpoints (95\\% CI)',
        label='tab:sqirl_results',
        position='htbp'
    )

    # Write to file
    with open(output_file, 'w') as f:
        f.write("% Values shown as 95% Confidence Interval [Lower, Upper]\n")
        f.write(latex_str)

    print(f"LaTeX table exported to: {output_file}")

    return df


def create_compact_latex_table(all_results, output_file='results_table_compact.tex'):
    """
    Create a more compact LaTeX table with Non-San and San side by side per metric.
    Uses 95% confidence intervals.
    """
    # Collect all unique endpoints
    all_endpoints = set()
    for training_type in all_results:
        all_endpoints.update(all_results[training_type].keys())

    def endpoint_sort_key(ep):
        match = re.search(r'sqli(\d+)', ep)
        return int(match.group(1)) if match else 999

    sorted_endpoints = sorted(all_endpoints, key=endpoint_sort_key)

    # Build data for compact table (two rows per endpoint - one for each training type)
    data = []
    for endpoint in sorted_endpoints:
        for training_type in ['non_san', 'sanitized']:
            results_list = all_results[training_type].get(endpoint, [])
            s = calculate_statistics(results_list)

            type_label = 'Non-San' if training_type == 'non_san' else 'Sanitized'
            ep_label = endpoint.replace('.php', '').replace('sqli', '') if training_type == 'non_san' else ''

            # Format as [lower, upper] for CI
            row = {
                'Endpoint': ep_label,
                'Training': type_label,
                'Win Rate (\\%)': f"$[{s['win_rate_ci_lower']:.1f}, {s['win_rate_ci_upper']:.1f}]$",
                'Episodes': f"$[{s['episodes_ci_lower']:.2f}, {s['episodes_ci_upper']:.2f}]$",
                'N': int(s['count'])
            }
            data.append(row)

    # Define sanitisation subset endpoints
    sanitisation_endpoints = ['sqli14.php', 'sqli15.php', 'sqli16.php', 'sqli17.php',
                              'sqli18.php', 'sqli26.php', 'sqli27.php', 'sqli28.php', 'sqli29.php']

    # Add sanitisation subset totals
    for training_type in ['non_san', 'sanitized']:
        san_wins = []
        san_episodes = []
        for endpoint in sorted_endpoints:
            if endpoint in sanitisation_endpoints:
                results_list = all_results[training_type].get(endpoint, [])
                for r in results_list:
                    san_wins.append(1 if r['won'] else 0)
                    san_episodes.append(r['episodes'])

        type_label = 'Non-San' if training_type == 'non_san' else 'Sanitized'
        ep_label = '\\textbf{Sanitisation}' if training_type == 'non_san' else ''

        if san_wins:
            win_mean, win_ci_lower, win_ci_upper, _ = calculate_confidence_interval(san_wins)
            ep_mean, ep_ci_lower, ep_ci_upper, _ = calculate_confidence_interval(san_episodes)
            win_ci_lower *= 100
            win_ci_upper *= 100
            n = len(san_wins)
        else:
            win_ci_lower = win_ci_upper = ep_ci_lower = ep_ci_upper = 0.0
            n = 0

        row = {
            'Endpoint': ep_label,
            'Training': f'\\textbf{{{type_label}}}',
            'Win Rate (\\%)': f"$\\mathbf{{[{win_ci_lower:.1f}, {win_ci_upper:.1f}]}}$",
            'Episodes': f"$\\mathbf{{[{ep_ci_lower:.2f}, {ep_ci_upper:.2f}]}}$",
            'N': n
        }
        data.append(row)

    # Add totals
    for training_type in ['non_san', 'sanitized']:
        all_wins = []
        all_episodes = []
        for endpoint in sorted_endpoints:
            results_list = all_results[training_type].get(endpoint, [])
            for r in results_list:
                all_wins.append(1 if r['won'] else 0)
                all_episodes.append(r['episodes'])

        type_label = 'Non-San' if training_type == 'non_san' else 'Sanitized'
        ep_label = '\\textbf{Total}' if training_type == 'non_san' else ''

        if all_wins:
            win_mean, win_ci_lower, win_ci_upper, _ = calculate_confidence_interval(all_wins)
            ep_mean, ep_ci_lower, ep_ci_upper, _ = calculate_confidence_interval(all_episodes)
            win_ci_lower *= 100
            win_ci_upper *= 100
            n = len(all_wins)
        else:
            win_ci_lower = win_ci_upper = ep_ci_lower = ep_ci_upper = 0.0
            n = 0

        row = {
            'Endpoint': ep_label,
            'Training': f'\\textbf{{{type_label}}}',
            'Win Rate (\\%)': f"$\\mathbf{{[{win_ci_lower:.1f}, {win_ci_upper:.1f}]}}$",
            'Episodes': f"$\\mathbf{{[{ep_ci_lower:.2f}, {ep_ci_upper:.2f}]}}$",
            'N': n
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Generate LaTeX with booktabs style
    latex_str = df.to_latex(
        index=False,
        escape=False,
        column_format='clrrr',
        caption='SQIRL Test Results on no\\_feedback.php: Win Rate and Episodes by Training Type (95\\% CI)',
        label='tab:sqirl_results_compact',
        position='htbp'
    )

    # Add booktabs commands for nicer formatting
    latex_str = latex_str.replace('\\toprule', '\\toprule\n\\multicolumn{5}{c}{\\textit{95\\% Confidence Interval [Lower, Upper]}} \\\\')

    with open(output_file, 'w') as f:
        f.write("% Requires: \\usepackage{booktabs}\n")
        f.write("% Values shown as 95% Confidence Interval [Lower, Upper]\n")
        f.write(latex_str)

    print(f"Compact LaTeX table exported to: {output_file}")

    return df


def export_csv(all_results, output_file='results_analysis.csv'):
    """Export results to CSV file using pandas."""

    df = create_results_dataframe(all_results)
    df.to_csv(output_file, index=False)
    print(f"CSV exported to: {output_file}")


def main():
    # Default to test_results in current directory
    test_results_dir = './test_results'

    if len(sys.argv) > 1:
        test_results_dir = sys.argv[1]

    if not os.path.isdir(test_results_dir):
        print(f"Error: Directory not found: {test_results_dir}")
        sys.exit(1)

    print(f"Analyzing results from: {test_results_dir}")

    # Collect all results
    all_results = collect_all_results(test_results_dir)

    # Print formatted table to terminal
    print_results_table(all_results)

    # Create pandas DataFrame and export to CSV
    df = create_results_dataframe(all_results)
    df.to_csv('results_analysis.csv', index=False)
    print(f"CSV exported to: results_analysis.csv")

    # Create LaTeX tables
    create_latex_table(all_results, 'results_table.tex')
    create_compact_latex_table(all_results, 'results_table_compact.tex')

    # Print the DataFrame summary
    print(f"\n{Colors.BOLD}DataFrame Summary:{Colors.END}")
    print(df.to_string(index=False))

    # Also print the LaTeX table content to terminal
    print(f"\n{Colors.BOLD}{Colors.CYAN}LaTeX Table (Compact):{Colors.END}")
    with open('results_table_compact.tex', 'r') as f:
        print(f.read())


if __name__ == '__main__':
    main()

