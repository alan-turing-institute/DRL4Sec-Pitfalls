#!/usr/bin/env python3
"""
Parser for train-log files to extract episode statistics.
Memory-efficient: computes statistics incrementally during parsing.
"""

import glob
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from scipy import stats as scipy_stats


try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Error: pandas is required. Install with: pip install pandas")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    class tqdm:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            pass


@dataclass
class RunningStats:
    """Welford's online algorithm for computing mean and variance incrementally."""
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0  # Sum of squared differences from mean
    
    def update(self, value: float):
        """Add a new value and update running statistics."""
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.M2 += delta * delta2
    
    def get_std(self) -> float:
        """Return population standard deviation."""
        if self.n < 1:
            return 0.0
        return math.sqrt(self.M2 / self.n)


@dataclass
class EpisodeAccumulator:
    """Accumulates statistics for a single episode during parsing."""
    reward_stats: RunningStats = field(default_factory=RunningStats)
    length: int = 0
    episode_id: int = 0  # Track which episode this is (0-indexed)
    successes: List[dict] = field(default_factory=list)  # List of {reward, vuln_type, url, payload}
    final_try: Optional[int] = None
    final_url: Optional[str] = None
    final_payload: Optional[str] = None
    
    def add_step(self, reward: float, try_value: Optional[int], url: Optional[str] = None, 
                 payload: Optional[str] = None, done: bool = False):
        """Add a step's data to the episode accumulator."""
        self.length += 1
        self.reward_stats.update(reward)
        
        if try_value is not None:
            self.final_try = try_value
        
        # Track successful vulnerabilities (reward > 0)
        if reward > 0:
            vuln_type = extract_vulnerability_type_from_url(url) if url else "unknown"
            self.successes.append({
                'reward': reward,
                'vuln_type': vuln_type,
                'url': url or '',
                'payload': payload or ''
            })
        
        if done:
            self.final_url = url
            self.final_payload = payload
    
    def finalize(self) -> dict:
        """Return episode statistics as a dictionary."""
        return {
            'EpsTaken': self.episode_id + 1,  # 1-indexed: episode 0 becomes episode 1
            'avg_reward': self.reward_stats.mean,
            'std_reward': self.reward_stats.get_std(),
            'success_count': len(self.successes),
            'final_try': self.final_try,
            'url': self.final_url or '',
            'payload': self.final_payload or ''
        }


def extract_vulnerability_type_from_url(url: str) -> str:
    """Extract vulnerability type identifier from URL."""
    if not url:
        return "unknown"
    # Extract the last meaningful part of the path
    url_parts = url.split('/')
    meaningful_parts = [p for p in url_parts if p and p not in ['http:', 'https:', '']]
    if meaningful_parts:
        return meaningful_parts[-1]
    return "unknown"


def parse_train_log(file_path: str) -> pd.DataFrame:
    """
    Parse a train-log file memory-efficiently, computing episode stats incrementally.
    
    Returns a DataFrame with one row per successful vulnerability (reward > 0):
    - reward: float (the successful reward value)
    - EpsTaken: int (number of episodes taken to find this vulnerability, 1-indexed)
    - episode_avg_reward: float  
    - episode_std_reward: float
    - vuln_type: str (extracted from URL)
    - url: str
    - payload: str
    - try_value: int
    - source_file: str (basename of file)
    """
    successes = []  # Only store successful vulnerabilities
    episode_id = 0
    episode = EpisodeAccumulator()
    episode.episode_id = episode_id
    
    in_step_log = False
    current_section_lines = []
    payload_lines = []
    in_payload = False
    
    step_log_pattern = re.compile(r'-+step log-+')
    separator_pattern = re.compile(r'-{10,}')
    
    source_file = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024,
                  desc=f"Processing {source_file}", leave=False) as pbar:
            for line in f:
                pbar.update(len(line.encode('utf-8')))
                
                if step_log_pattern.match(line.strip()):
                    in_step_log = True
                    current_section_lines = []
                    payload_lines = []
                    in_payload = False
                    continue
                
                if in_step_log:
                    if separator_pattern.match(line.strip()):
                        # Process accumulated section
                        section_text = '\n'.join(current_section_lines)
                        
                        # Extract values
                        reward = 0.0
                        reward_match = re.search(r'reward:\s*([-\d.]+)', section_text)
                        if reward_match:
                            reward = float(reward_match.group(1))
                        
                        done = False
                        done_match = re.search(r'done:\s*(True|False)', section_text)
                        if done_match:
                            done = done_match.group(1) == 'True'
                        
                        try_value = None
                        try_match = re.search(r'try:\s*(\d+)', section_text)
                        if try_match:
                            try_value = int(try_match.group(1))
                        
                        url = None
                        payload = None
                        if done:
                            url_match = re.search(r'^url:\s*(.+)$', section_text, re.MULTILINE)
                            if url_match:
                                url = url_match.group(1).strip()
                            if payload_lines:
                                payload = ''.join(payload_lines).strip().rstrip('\n\r')
                        
                        # Add step to episode accumulator
                        episode.add_step(reward, try_value, url, payload, done)
                        
                        # If episode ended, finalize and store successes
                        if done:
                            ep_stats = episode.finalize()
                            
                            # Store each success with episode context
                            for success in episode.successes:
                                successes.append({
                                    'episode_id': episode_id,
                                    'reward': success['reward'],
                                    'EpsTaken': ep_stats['EpsTaken'],
                                    'episode_avg_reward': ep_stats['avg_reward'],
                                    'episode_std_reward': ep_stats['std_reward'],
                                    'vuln_type': success['vuln_type'],
                                    'url': success['url'],
                                    'payload': success['payload'],
                                    'try_value': ep_stats['final_try'],
                                    'source_file': source_file
                                })
                            
                            # Reset for next episode
                            episode_id += 1
                            episode = EpisodeAccumulator()
                            episode.episode_id = episode_id
                        
                        in_step_log = False
                        current_section_lines = []
                        payload_lines = []
                        in_payload = False
                        continue
                    
                    current_section_lines.append(line.rstrip('\n\r'))
                    
                    if line.startswith('Payload:'):
                        in_payload = True
                        payload_match = re.match(r'Payload:\s*(.*)', line)
                        if payload_match and payload_match.group(1).strip():
                            payload_lines.append(payload_match.group(1))
                    elif in_payload:
                        payload_lines.append(line.rstrip('\n\r'))
    
    # Handle last episode if it didn't end with done=True
    if episode.length > 0:
        ep_stats = episode.finalize()
        for success in episode.successes:
            successes.append({
                'episode_id': episode_id,
                'reward': success['reward'],
                'EpsTaken': ep_stats['EpsTaken'],
                'episode_avg_reward': ep_stats['avg_reward'],
                'episode_std_reward': ep_stats['std_reward'],
                'vuln_type': success['vuln_type'],
                'url': success['url'],
                'payload': success['payload'],
                'try_value': ep_stats['final_try'],
                'source_file': source_file
            })
    
    return pd.DataFrame(successes)


def parse_all_files(file_list: List[str], category_name: str) -> Tuple[pd.DataFrame, List[str]]:
    """Parse multiple train-log files and combine results into a single DataFrame.
    
    Returns:
        Tuple of (combined DataFrame, list of all parsed file names)
    """
    dfs = []
    all_parsed_files = []  # Track all files, even those with no vulns
    
    for file_path in file_list:
        try:
            source_name = os.path.basename(file_path)
            print(f"  Parsing {source_name}...")
            df = parse_train_log(file_path)
            all_parsed_files.append(source_name)  # Track this file
            
            if not df.empty:
                dfs.append(df)
                print(f"    Found {len(df)} successful vulnerabilities")
            else:
                print(f"    No successful vulnerabilities found")
        except Exception as e:
            print(f"    Error parsing {file_path}: {e}")
            continue
    
    if dfs:
        return pd.concat(dfs, ignore_index=True), all_parsed_files
    return pd.DataFrame(), all_parsed_files


def calculate_std_dev(series: pd.Series) -> float:
    """Calculate population standard deviation."""
    if series.empty:
        return 0.0
    return float(series.std(ddof=0))


def calculate_95_ci(series: pd.Series) -> Tuple[float, float]:
    """
    Calculate 95% confidence interval for the mean.
    
    Returns (mean, ci_half_width) where the interval is [mean - ci, mean + ci]
    Uses z=1.96 for 95% confidence.
    """
    if series.empty or len(series) < 2:
        return (series.mean() if not series.empty else 0.0, 0.0)
    
    n = len(series)
    mean = series.mean()
    std = series.std(ddof=1)  # Sample std dev for CI calculation
    se = std / math.sqrt(n)   # Standard error
    ci = 1.96 * se            # 95% CI half-width
    
    return (float(mean), float(ci))


def format_with_ci(mean: float, ci: float, decimals: int = 2) -> str:
    """Format a value with its 95% CI for LaTeX output."""
    if decimals == 1:
        return f"{mean:.1f} $\\pm$ {ci:.1f}"
    return f"{mean:.{decimals}f} $\\pm$ {ci:.{decimals}f}"


def calculate_proportion_ci(successes: int, total: int) -> Tuple[float, float, float]:
    """
    Calculate 95% confidence interval for a proportion using Wilson score interval.
    
    Args:
        successes: Number of runs where vulnerability was found
        total: Total number of runs
    
    Returns:
        (proportion, ci_lower, ci_upper) - the proportion and 95% CI bounds
    """
    if total == 0:
        return (0.0, 0.0, 0.0)
    
    data = [0 for _ in range(total)]
    for i in range(successes):
        data[i] = 1
    n = len(data)
    if n < 2:
        mean = np.mean(data) if n > 0 else 0.0
        return mean, mean, mean, 0.0
    confidence=0.95
    mean = np.mean(data)
    std_err = scipy_stats.sem(data)  # Standard error of the mean
    
    # Use t-distribution for small samples
    t_value = scipy_stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_of_error = t_value * std_err
    
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    
    return (mean, ci_lower, ci_upper)


def format_discovery_rate(successes: int, total: int, include_ci: bool = True) -> str:
    """Format discovery rate as 'X/Y (p% [CI])' for LaTeX output."""
    if total == 0:
        return "N/A"
    
    p, ci_lower, ci_upper = calculate_proportion_ci(successes, total)

    
    if include_ci:
        return f"{successes}/{total} ({p*100:.1f}\\% [{ci_lower*100:.1f}\\%-{ci_upper*100:.1f}\\%])"
    return f"{successes}/{total} ({p*100:.1f}\\%)"


def calculate_discovery_stats(df: pd.DataFrame) -> dict:
    """
    Calculate discovery statistics: how many runs found each vulnerability type.
    
    Returns a dict with:
        - total_runs: int
        - vuln_discovery: {vuln_type: {'runs_found': int, 'total_runs': int}}
    """
    if df.empty:
        return {'total_runs': 0, 'vuln_discovery': {}}
    
    # Get unique runs (source files)
    all_runs = df['source_file'].unique()
    total_runs = len(all_runs)
    
    # For each vulnerability type, count how many unique runs found it
    vuln_discovery = {}
    for vuln_type in df['vuln_type'].unique():
        vuln_df = df[df['vuln_type'] == vuln_type]
        runs_with_vuln = vuln_df['source_file'].nunique()
        vuln_discovery[vuln_type] = {
            'runs_found': runs_with_vuln,
            'total_runs': total_runs
        }
    
    return {
        'total_runs': total_runs,
        'vuln_discovery': vuln_discovery
    }


def generate_vulnerability_table(df: pd.DataFrame, output_prefix: str):
    """Generate a LaTeX table with statistics per successful vulnerability with 95% CIs."""
    if df.empty:
        print("No successful vulnerabilities to analyze")
        return
    
    # Calculate 95% CI for EpsTaken across unique episodes with vulnerabilities
    unique_episodes = df.drop_duplicates(subset=['episode_id', 'source_file'])
    eps_taken_mean, eps_taken_ci = calculate_95_ci(unique_episodes['EpsTaken'])
    
    # Calculate 95% CI for rewards
    reward_mean, reward_ci = calculate_95_ci(df['reward'])
    
    # Calculate 95% CI for episode avg rewards  
    ep_reward_mean, ep_reward_ci = calculate_95_ci(df['episode_avg_reward'])
    
    # Calculate discovery stats
    discovery_stats = calculate_discovery_stats(df)
    total_runs = discovery_stats['total_runs']
    
    output_file = f"{output_prefix}_vulnerability_stats.tex"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{|l|c|}\n")
        f.write("\\hline\n")
        f.write("Metric & Value (95\\% CI) \\\\\n")
        f.write("\\hline\n")
        f.write(f"Number of Runs & {total_runs} \\\\\n")
        f.write(f"Total Successful Vulnerabilities & {len(df)} \\\\\n")
        f.write(f"Average Reward & {format_with_ci(reward_mean, reward_ci, 4)} \\\\\n")
        f.write(f"Episode Average Reward & {format_with_ci(ep_reward_mean, ep_reward_ci, 4)} \\\\\n")
        f.write(f"EpsTaken & {format_with_ci(eps_taken_mean, eps_taken_ci, 1)} \\\\\n")
        f.write("\\hline\n")
        
        # Add per-vulnerability discovery rates
        if discovery_stats['vuln_discovery']:
            f.write("\\multicolumn{2}{|c|}{\\textbf{Discovery Rate by Vulnerability Type}} \\\\\n")
            f.write("\\hline\n")
            for vuln_type, stats in sorted(discovery_stats['vuln_discovery'].items()):
                runs_found = stats['runs_found']
                p, ci_lower, ci_upper = calculate_proportion_ci(runs_found, total_runs)
                f.write(f"{vuln_type} & {runs_found}/{total_runs} ({p*100:.1f}\\% [{ci_lower*100:.1f}\\%-{ci_upper*100:.1f}\\%]) \\\\\n")
            f.write("\\hline\n")
        
        f.write("\\end{tabular}\n")
        f.write("\\caption{Vulnerability Statistics with 95\\% Confidence Intervals}\n")
        f.write("\\end{table}\n")
    
    print(f"Saved vulnerability statistics table: {output_file}")
    print(f"  Total runs: {total_runs}")
    print(f"  Found {len(df)} successful vulnerabilities (reward > 0)")
    print(f"  Average reward: {reward_mean:.4f} ± {reward_ci:.4f} (95% CI)")
    print(f"  EpsTaken: {eps_taken_mean:.1f} ± {eps_taken_ci:.1f} (95% CI)")
    
    # Print discovery rates
    if discovery_stats['vuln_discovery']:
        print(f"  Discovery rates by vulnerability type:")
        for vuln_type, stats in sorted(discovery_stats['vuln_discovery'].items()):
            runs_found = stats['runs_found']
            p = runs_found / total_runs if total_runs > 0 else 0
            print(f"    {vuln_type}: {runs_found}/{total_runs} ({p*100:.1f}%)")


def generate_combined_vulnerability_table(file_type_data: dict, output_file: str = 'combined_vulnerability_stats.tex'):
    """Generate a single combined LaTeX table with file types as rows and vulnerabilities as columns.
    
    All statistics include 95% confidence intervals.
    Includes discovery rate: proportion of runs that found each vulnerability.
    """
    # Combine all DataFrames with file_type label
    all_dfs = []
    for file_type, df in file_type_data.items():
        if df is not None and not df.empty:
            df = df.copy()
            df['file_type'] = file_type
            all_dfs.append(df)
    
    if not all_dfs:
        print("No data to generate combined table.")
        return
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Get unique file types and vulnerability types
    file_types = sorted(combined_df['file_type'].unique())
    vulnerability_types = sorted(combined_df['vuln_type'].unique())
    
    # Build rows for the output table
    rows = []
    for file_type in file_types:
        ft_df = combined_df[combined_df['file_type'] == file_type]
        
        # Get total number of runs for this file type
        total_runs = ft_df['source_file'].nunique()
        
        avg_reward_row = {'File Type': file_type, 'Metric': 'Average Reward (95\\% CI)'}
        eps_taken_row = {'File Type': file_type, 'Metric': 'EpsTaken (95\\% CI)'}
        count_row = {'File Type': file_type, 'Metric': 'Count'}
        discovery_row = {'File Type': file_type, 'Metric': 'Discovery Rate (95\\% CI)'}
        
        all_rewards = []
        all_eps_taken = []
        total_vulns_found_across_runs = 0  # Track unique vuln types found in any run
        
        for vuln_type in vulnerability_types:
            vt_df = ft_df[ft_df['vuln_type'] == vuln_type]
            
            if not vt_df.empty:
                # Calculate 95% CI for reward
                reward_mean, reward_ci = calculate_95_ci(vt_df['reward'])
                avg_reward_row[vuln_type] = format_with_ci(reward_mean, reward_ci, 2)
                all_rewards.extend(vt_df['reward'].tolist())
                
                # Calculate 95% CI for EpsTaken
                eps_mean, eps_ci = calculate_95_ci(vt_df['EpsTaken'])
                eps_taken_row[vuln_type] = format_with_ci(eps_mean, eps_ci, 1)
                all_eps_taken.extend(vt_df['EpsTaken'].tolist())
                
                count_row[vuln_type] = str(len(vt_df))
                
                # Discovery rate: how many runs found this vulnerability
                runs_found = vt_df['source_file'].nunique()
                discovery_row[vuln_type] = format_discovery_rate(runs_found, total_runs)
                total_vulns_found_across_runs += runs_found
            else:
                avg_reward_row[vuln_type] = "N/A"
                eps_taken_row[vuln_type] = "N/A"
                count_row[vuln_type] = "0"
                discovery_row[vuln_type] = format_discovery_rate(0, total_runs)
        
        # Summary column with 95% CI
        if all_rewards:
            summary_mean, summary_ci = calculate_95_ci(pd.Series(all_rewards))
            avg_reward_row['Summary'] = format_with_ci(summary_mean, summary_ci, 2)
        else:
            avg_reward_row['Summary'] = "N/A"
        
        if all_eps_taken:
            summary_eps, summary_ci_eps = calculate_95_ci(pd.Series(all_eps_taken))
            eps_taken_row['Summary'] = format_with_ci(summary_eps, summary_ci_eps, 1)
        else:
            eps_taken_row['Summary'] = "N/A"
        
        count_row['Summary'] = str(len(ft_df))
        
        # Summary discovery rate: average number of vulnerabilities found per run
        # For each run, count how many unique vulnerability types were found
        # Then average across all runs and show with CI
        num_vuln_types = len(vulnerability_types)
        if total_runs > 0 and num_vuln_types > 0:
            # Count vulns found per run
            vulns_per_run = []
            for source_file in ft_df['source_file'].unique():
                run_df = ft_df[ft_df['source_file'] == source_file]
                vulns_found_in_run = run_df['vuln_type'].nunique()
                vulns_per_run.append(vulns_found_in_run)
            
            # Calculate mean and CI
            vulns_series = pd.Series(vulns_per_run)
            avg_vulns, ci_half = calculate_95_ci(vulns_series)
            
            # Format as "avg/total (percentage [CI])"
            pct = (avg_vulns / num_vuln_types) * 100
            pct_lower = ((avg_vulns - ci_half) / num_vuln_types) * 100
            pct_upper = ((avg_vulns + ci_half) / num_vuln_types) * 100
            discovery_row['Summary'] = f"{avg_vulns:.1f}/{num_vuln_types} ({pct:.1f}\\% [{max(0,pct_lower):.1f}\\%-{min(100,pct_upper):.1f}\\%])"
        else:
            discovery_row['Summary'] = "N/A"
        
        rows.append(avg_reward_row)
        rows.append(eps_taken_row)
        rows.append(count_row)
        rows.append(discovery_row)
    
    # Create output DataFrame and generate LaTeX
    output_df = pd.DataFrame(rows)
    output_df = output_df.set_index(['File Type', 'Metric'])
    
    latex_str = output_df.to_latex(
        escape=False,
        float_format="%.2f",
        caption="Vulnerability Statistics by File Type and Vulnerability Type (with 95\\% CI)",
        label="tab:vulnerability_stats"
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex_str)
    
    print(f"Saved combined vulnerability statistics table: {output_file}")
    print(f"  File types: {', '.join(file_types)}")
    print(f"  Vulnerability types: {', '.join(vulnerability_types)}")
    print(f"  All statistics include 95% confidence intervals")
    print(f"  Discovery rates show X/Y runs that found each vulnerability")


def export_to_csv(df: pd.DataFrame, output_file: str):
    """Export DataFrame to CSV."""
    if df.empty:
        print("No data to export.")
        return
    
    df.to_csv(output_file, index=False)
    print(f"Exported {len(df)} successful vulnerabilities to {output_file}")


def generate_vulnerability_matrix(file_type_data: dict, all_files_data: dict = None, output_file: str = 'vulns_out.pkl'):
    """
    Generate a 2D vulnerability matrix DataFrame.
    
    Rows: Individual runs (source files), grouped by file type
    Columns: Vulnerability types (1 if found, 0 if not found)
    
    Args:
        file_type_data: Dict mapping file type names to DataFrames
        all_files_data: Dict mapping file type names to list of ALL parsed file names
                       (includes files with no vulnerabilities found)
        output_file: Output pickle file path
    
    Returns:
        DataFrame with the vulnerability matrix
    """
    # Combine all DataFrames with file_type label
    all_dfs = []
    for file_type, df in file_type_data.items():
        if df is not None and not df.empty:
            df = df.copy()
            df['file_type'] = file_type
            all_dfs.append(df)
    
    if not all_dfs:
        print("No data to generate vulnerability matrix.")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Get all unique vulnerability types
    all_vuln_types = sorted(combined_df['vuln_type'].unique())
    
    # Build the matrix: rows are (file_type, source_file), columns are vulnerability types
    rows = []
    
    # If all_files_data is provided, use it to include ALL files (even those with 0 vulns)
    if all_files_data:
        for file_type, file_list in all_files_data.items():
            if not file_list:
                continue
            
            ft_df = combined_df[combined_df['file_type'] == file_type] if not combined_df.empty else pd.DataFrame()
            
            for source_file in sorted(file_list):
                # Check what vulns were found in this run
                if not ft_df.empty:
                    run_df = ft_df[ft_df['source_file'] == source_file]
                    vulns_found_in_run = set(run_df['vuln_type'].unique()) if not run_df.empty else set()
                else:
                    vulns_found_in_run = set()
                
                row = {
                    'file_type': file_type,
                    'source_file': source_file
                }
                
                # Mark each vulnerability as 1 (found) or 0 (not found)
                for vuln_type in all_vuln_types:
                    row[vuln_type] = 1 if vuln_type in vulns_found_in_run else 0
                
                rows.append(row)
    else:
        # Fallback: only include files that have at least one vulnerability
        for file_type in sorted(combined_df['file_type'].unique()):
            ft_df = combined_df[combined_df['file_type'] == file_type]
            
            for source_file in sorted(ft_df['source_file'].unique()):
                run_df = ft_df[ft_df['source_file'] == source_file]
                vulns_found_in_run = set(run_df['vuln_type'].unique())
                
                row = {
                    'file_type': file_type,
                    'source_file': source_file
                }
                
                for vuln_type in all_vuln_types:
                    row[vuln_type] = 1 if vuln_type in vulns_found_in_run else 0
                
                rows.append(row)
    
    # Create DataFrame
    matrix_df = pd.DataFrame(rows)
    
    if matrix_df.empty:
        print("No runs to include in vulnerability matrix.")
        return pd.DataFrame()
    
    # Set multi-index for grouping
    matrix_df = matrix_df.set_index(['file_type', 'source_file'])
    
    # Save to pickle
    matrix_df.to_pickle(output_file)
    print(f"Saved vulnerability matrix: {output_file}")
    print(f"  Shape: {matrix_df.shape[0]} runs x {matrix_df.shape[1]} vulnerability types")
    
    # Count runs per file type
    file_type_counts = matrix_df.reset_index()['file_type'].value_counts().sort_index()
    print(f"  Runs per file type: {dict(file_type_counts)}")
    
    # Also save as CSV for easy viewing
    csv_file = output_file.replace('.pkl', '.csv')
    matrix_df.to_csv(csv_file)
    print(f"  Also saved as CSV: {csv_file}")
    
    return matrix_df


def load_from_csv(csv_file: str) -> pd.DataFrame:
    """
    Load DataFrame from CSV file and ensure it has EpsTaken column.
    
    If the CSV has episode_length instead of EpsTaken, compute EpsTaken from episode_id.
    """
    df = pd.read_csv(csv_file)
    
    # Check if EpsTaken exists, if not compute it from episode_id
    if 'EpsTaken' not in df.columns:
        if 'episode_id' in df.columns:
            df['EpsTaken'] = df['episode_id'] + 1  # Convert 0-indexed to 1-indexed
            print(f"  Computed EpsTaken from episode_id for {os.path.basename(csv_file)}")
        elif 'episode_length' in df.columns:
            # Fallback: if we only have episode_length, we can't compute EpsTaken
            # But we'll try to infer from the order (not ideal, but better than nothing)
            print(f"  Warning: {os.path.basename(csv_file)} has episode_length but no episode_id")
            print(f"  Cannot compute EpsTaken accurately. Please re-parse the original train-log files.")
            # Create a dummy EpsTaken column (this is not ideal)
            df['EpsTaken'] = 1
        else:
            print(f"  Warning: {os.path.basename(csv_file)} has neither EpsTaken nor episode_id/episode_length")
            df['EpsTaken'] = 1
    
    # Remove episode_length if it exists (we're using EpsTaken now)
    if 'episode_length' in df.columns:
        df = df.drop(columns=['episode_length'])
    
    return df


def find_train_log_files(directory: str = '.') -> Tuple[List[str], List[str], List[str], List[str]]:
    """Find all train-log-* files and categorize them into 'd', 'non-d', 'da', and 'ds' groups."""
    pattern = os.path.join(directory, 'train-log-*')
    all_files = glob.glob(pattern)
    
    d_files = []
    non_d_files = []
    da_files = []
    ds_files = []
    
    for file_path in sorted(all_files):
        if file_path.endswith('-da.txt'):
            da_files.append(file_path)
        elif file_path.endswith('ds-nda.txt'):
            ds_files.append(file_path)
        elif file_path.endswith('-d.txt') and 'non-d' not in file_path:
            d_files.append(file_path)
        elif file_path.endswith('-non-d.txt'):
            non_d_files.append(file_path)
    
    return d_files, non_d_files, da_files, ds_files


def find_combined_csv_files(directory: str = '.') -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Find combined CSV files for each category."""
    d_csv = os.path.join(directory, 'train-log-d_combined.csv')
    non_d_csv = os.path.join(directory, 'train-log-non-d_combined.csv')
    da_csv = os.path.join(directory, 'train-log-da_combined.csv')
    ds_csv = os.path.join(directory, 'train-log-ds_combined.csv')
    
    d_file = d_csv if os.path.exists(d_csv) else None
    non_d_file = non_d_csv if os.path.exists(non_d_csv) else None
    da_file = da_csv if os.path.exists(da_csv) else None
    ds_file = ds_csv if os.path.exists(ds_csv) else None
    
    return d_file, non_d_file, da_file, ds_file


def main():
    # Check if user wants to use CSV files
    use_csv = False
    if len(sys.argv) >= 2 and sys.argv[1] in ['--from-csv', '--csv', '-c']:
        use_csv = True
        args_start = 2
    else:
        args_start = 1
    
    if len(sys.argv) == args_start or (len(sys.argv) == args_start + 1 and sys.argv[args_start] in ['--all', '-a', '--batch']):
        # Batch mode
        if use_csv:
            print("Batch mode: Loading from combined CSV files...")
            
            d_csv, non_d_csv, da_csv, ds_csv = find_combined_csv_files()
            
            d_df = pd.DataFrame()
            non_d_df = pd.DataFrame()
            da_df = pd.DataFrame()
            ds_df = pd.DataFrame()
            
            # Track all parsed files (we'll extract from source_file column)
            all_files_data = {'d': [], 'non-d': [], 'da': [], 'ds': []}
            
            if d_csv:
                print(f"\nLoading 'd' data from {os.path.basename(d_csv)}:")
                d_df = load_from_csv(d_csv)
                if not d_df.empty:
                    all_files_data['d'] = d_df['source_file'].unique().tolist()
                    print(f"  Loaded {len(d_df)} successful vulnerabilities from {len(all_files_data['d'])} runs")
            
            if non_d_csv:
                print(f"\nLoading 'non-d' data from {os.path.basename(non_d_csv)}:")
                non_d_df = load_from_csv(non_d_csv)
                if not non_d_df.empty:
                    all_files_data['non-d'] = non_d_df['source_file'].unique().tolist()
                    print(f"  Loaded {len(non_d_df)} successful vulnerabilities from {len(all_files_data['non-d'])} runs")
            
            if da_csv:
                print(f"\nLoading 'da' data from {os.path.basename(da_csv)}:")
                da_df = load_from_csv(da_csv)
                if not da_df.empty:
                    all_files_data['da'] = da_df['source_file'].unique().tolist()
                    print(f"  Loaded {len(da_df)} successful vulnerabilities from {len(all_files_data['da'])} runs")
            
            if ds_csv:
                print(f"\nLoading 'ds' data from {os.path.basename(ds_csv)}:")
                ds_df = load_from_csv(ds_csv)
                if not ds_df.empty:
                    all_files_data['ds'] = ds_df['source_file'].unique().tolist()
                    print(f"  Loaded {len(ds_df)} successful vulnerabilities from {len(all_files_data['ds'])} runs")
        else:
            # Original batch mode: parse train-log files
            print("Batch mode: Processing all train-log-* files...")
            
            d_files, non_d_files, da_files, ds_files = find_train_log_files()
            
            print(f"\nFound {len(d_files)} 'd' files, {len(non_d_files)} 'non-d' files, "
                  f"{len(da_files)} 'da' files, and {len(ds_files)} 'ds' files")
            
            d_df = pd.DataFrame()
            non_d_df = pd.DataFrame()
            da_df = pd.DataFrame()
            ds_df = pd.DataFrame()
            
            # Track all parsed files (including those with no vulns)
            all_files_data = {'d': [], 'non-d': [], 'da': [], 'ds': []}
            
            if d_files:
                print(f"\nProcessing 'd' files:")
                d_df, d_all_files = parse_all_files(d_files, 'd')
                all_files_data['d'] = d_all_files
            
            if non_d_files:
                print(f"\nProcessing 'non-d' files:")
                non_d_df, non_d_all_files = parse_all_files(non_d_files, 'non-d')
                all_files_data['non-d'] = non_d_all_files
            
            if da_files:
                print(f"\nProcessing 'da' files:")
                da_df, da_all_files = parse_all_files(da_files, 'da')
                all_files_data['da'] = da_all_files
            
            if ds_files:
                print(f"\nProcessing 'ds' files:")
                ds_df, ds_all_files = parse_all_files(ds_files, 'ds')
                all_files_data['ds'] = ds_all_files
        
        # Export and generate tables (only if not using CSV mode, or if user wants to regenerate)
        if not use_csv:
            for name, df in [('d', d_df), ('non-d', non_d_df), ('da', da_df), ('ds', ds_df)]:
                if not df.empty:
                    export_to_csv(df, f'train-log-{name}_combined.csv')
                    print(f"\nSummary for '{name}' files:")
                    print(f"  Total successful vulnerabilities: {len(df)}")
                    print(f"  Average reward: {df['reward'].mean():.4f}")
                    generate_vulnerability_table(df, f'train-log-{name}_combined')
        else:
            # Just print summaries when loading from CSV
            for name, df in [('d', d_df), ('non-d', non_d_df), ('da', da_df), ('ds', ds_df)]:
                if not df.empty:
                    print(f"\nSummary for '{name}' files:")
                    print(f"  Total successful vulnerabilities: {len(df)}")
                    print(f"  Average reward: {df['reward'].mean():.4f}")
                    print(f"  Average EpsTaken: {df['EpsTaken'].mean():.1f}")
                    generate_vulnerability_table(df, f'train-log-{name}_combined')
        
        # Check if any files were parsed at all
        total_files = sum(len(files) for files in all_files_data.values())
        
        if d_df.empty and non_d_df.empty and da_df.empty and ds_df.empty:
            print("\nNo successful vulnerabilities found in any files.")
            if total_files > 0:
                print(f"  (Parsed {total_files} files total)")
        else:
            print("\n" + "=" * 80)
            print("Generating combined vulnerability statistics table...")
            file_type_data = {
                'd': d_df,
                'da': da_df,
                'ds': ds_df,
                'non-d': non_d_df
            }
            generate_combined_vulnerability_table(file_type_data, 'combined_vulnerability_stats.tex')
            
            # Generate vulnerability matrix (2D: runs x vulnerabilities)
            # Include ALL parsed files, even those with no vulns found
            print("\n" + "=" * 80)
            print("Generating vulnerability matrix...")
            generate_vulnerability_matrix(file_type_data, all_files_data, 'vulns_out.pkl')
    
    else:
        # Single file mode
        file_path = sys.argv[1]
        
        if len(sys.argv) >= 3:
            output_file = sys.argv[2]
        else:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_file = f"{base_name}_parsed.csv"
        
        try:
            print(f"Parsing {file_path}...")
            df = parse_train_log(file_path)
            
            export_to_csv(df, output_file)
            
            base_name = os.path.splitext(output_file)[0]
            generate_vulnerability_table(df, base_name)
            
            print("\n" + "=" * 80)
            print("Summary Statistics:")
            print(f"  Total successful vulnerabilities: {len(df)}")
            if not df.empty:
                print(f"  Average reward: {df['reward'].mean():.4f}")
                print(f"  Average EpsTaken: {df['EpsTaken'].mean():.1f}")
            
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error parsing file: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()
