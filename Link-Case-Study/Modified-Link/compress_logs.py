#!/usr/bin/env python3
"""
Compress existing train-log files to the new minimal format.
Removes unnecessary logging while preserving all data needed for parsing.
Processes files line by line to handle large files efficiently.
"""

import os
import sys
import glob
import re
import shutil
from typing import Optional

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


def compress_log_file(input_file: str, output_file: Optional[str] = None, backup: bool = True):
    """
    Compress a log file to the minimal format, processing line by line.
    
    Args:
        input_file: Path to the input log file
        output_file: Path to the output file (default: overwrite input)
        backup: Whether to create a backup of the original file
    """
    print(f"Compressing {os.path.basename(input_file)}...")
    
    # Get original file size for progress tracking
    input_size = os.path.getsize(input_file)
    
    # Create backup if requested
    if backup and output_file is None:
        backup_file = input_file + ".backup"
        if not os.path.exists(backup_file):
            print(f"  Creating backup: {os.path.basename(backup_file)}")
            shutil.copy2(input_file, backup_file)
    
    # Determine output file (use temp file if overwriting to avoid issues)
    use_temp = False
    if output_file is None:
        output_file = input_file
        temp_output = input_file + ".tmp"
        use_temp = True
    else:
        temp_output = output_file
    
    # Patterns for matching
    step_log_pattern = re.compile(r'-+step log-+')
    separator_pattern = re.compile(r'-{10,}')
    
    # State tracking
    in_step_log = False
    current_step_data = {}
    current_section_lines = []
    payload_lines = []
    in_payload = False
    step_count = 0
    
    # Process file line by line
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            with open(temp_output, 'w', encoding='utf-8') as f_out:
                with tqdm(total=input_size, unit='B', unit_scale=True, unit_divisor=1024,
                          desc=f"  Processing", leave=False) as pbar:
                    for line in f_in:
                        bytes_read = len(line.encode('utf-8'))
                        pbar.update(bytes_read)
                        
                        # Check if this is the start of a step log section
                        if step_log_pattern.match(line.strip()):
                            in_step_log = True
                            current_step_data = {}
                            current_section_lines = []
                            payload_lines = []
                            in_payload = False
                            continue
                        
                        # If we're in a step log section
                        if in_step_log:
                            # Check if this is the separator line (end of step log)
                            if separator_pattern.match(line.strip()):
                                # Process the accumulated section
                                section_text = '\n'.join(current_section_lines)
                                
                                # Extract reward
                                reward_match = re.search(r'reward:\s*([-\d.]+)', section_text)
                                if reward_match:
                                    current_step_data['reward'] = float(reward_match.group(1))
                                
                                # Extract done
                                done_match = re.search(r'done:\s*(True|False)', section_text)
                                if done_match:
                                    current_step_data['done'] = done_match.group(1) == 'True'
                                
                                # Extract try
                                try_match = re.search(r'try:\s*(\d+)', section_text)
                                if try_match:
                                    current_step_data['try'] = int(try_match.group(1))
                                
                                # If done is True, also extract payload and url
                                if current_step_data.get('done', False):
                                    # Extract url (line starting with "url:")
                                    url_match = re.search(r'^url:\s*(.+)$', section_text, re.MULTILINE)
                                    if url_match:
                                        current_step_data['url'] = url_match.group(1).strip()
                                    
                                    # Process accumulated payload lines
                                    if payload_lines:
                                        payload_value = ''.join(payload_lines).strip()
                                        payload_value = payload_value.rstrip('\n\r')
                                        if payload_value:
                                            current_step_data['payload'] = payload_value
                                
                                # Write compressed format immediately
                                if current_step_data:
                                    step_count += 1
                                    f_out.write("----------------step log----------------\n")
                                    
                                    # Write try
                                    if 'try' in current_step_data:
                                        f_out.write(f"try: {current_step_data['try']}\n")
                                    
                                    # Write reward
                                    if 'reward' in current_step_data:
                                        f_out.write(f"reward: {current_step_data['reward']}\n")
                                    
                                    # Write done
                                    if 'done' in current_step_data:
                                        f_out.write(f"done: {current_step_data['done']}\n")
                                    
                                    # Only write url and Payload when done=True
                                    if current_step_data.get('done', False):
                                        if 'url' in current_step_data and current_step_data['url']:
                                            f_out.write(f"url: {current_step_data['url']}\n")
                                        if 'payload' in current_step_data and current_step_data['payload']:
                                            f_out.write(f"Payload: {current_step_data['payload']}\n")
                                    
                                    # Write end separator
                                    f_out.write("----------------------------------------\n")
                                
                                # Reset for next step
                                in_step_log = False
                                current_step_data = {}
                                current_section_lines = []
                                payload_lines = []
                                in_payload = False
                                continue
                            
                            # Accumulate lines for this section
                            current_section_lines.append(line.rstrip('\n\r'))
                            
                            # Track payload lines (starts after "Payload:" and continues until separator)
                            if line.startswith('Payload:'):
                                in_payload = True
                                # Extract payload value from this line (after "Payload:")
                                payload_match = re.match(r'Payload:\s*(.*)', line)
                                if payload_match:
                                    payload_value = payload_match.group(1)
                                    if payload_value.strip():
                                        payload_lines.append(payload_value)
                            elif in_payload:
                                # Continue accumulating payload lines until we hit the separator
                                payload_lines.append(line.rstrip('\n\r'))
        
        # Replace original file with compressed version if we used temp file
        if use_temp:
            if os.path.exists(output_file):
                os.remove(output_file)
            shutil.move(temp_output, output_file)
        
        # Get file sizes
        output_size = os.path.getsize(output_file)
        reduction = ((input_size - output_size) / input_size) * 100
        
        print(f"  Parsed {step_count} steps")
        print(f"  Original size: {input_size:,} bytes ({input_size / (1024*1024):.2f} MB)")
        print(f"  Compressed size: {output_size:,} bytes ({output_size / (1024*1024):.2f} MB)")
        print(f"  Reduction: {reduction:.1f}%")
        
        return True
        
    except Exception as e:
        # Clean up temp file on error
        if use_temp and os.path.exists(temp_output):
            os.remove(temp_output)
        print(f"  Error compressing file: {e}")
        import traceback
        traceback.print_exc()
        return False


def compress_all_logs(directory: str = '.', backup: bool = True, pattern: str = 'train-log-*.txt'):
    """
    Compress all log files matching the pattern.
    
    Args:
        directory: Directory to search for log files
        backup: Whether to create backups
        pattern: Glob pattern to match log files
    """
    pattern_path = os.path.join(directory, pattern)
    log_files = glob.glob(pattern_path)
    
    # Filter out backup files
    log_files = [f for f in log_files if not f.endswith('.backup')]
    
    if not log_files:
        print(f"No log files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(log_files)} log files to compress\n")
    
    total_original_size = 0
    total_compressed_size = 0
    
    for log_file in sorted(log_files):
        original_size = os.path.getsize(log_file)
        total_original_size += original_size
        
        success = compress_log_file(log_file, backup=backup)
        
        if success:
            compressed_size = os.path.getsize(log_file)
            total_compressed_size += compressed_size
        
        print()
    
    # Print summary
    if total_original_size > 0:
        total_reduction = ((total_original_size - total_compressed_size) / total_original_size) * 100
        print("=" * 60)
        print("Summary:")
        print(f"  Files processed: {len(log_files)}")
        print(f"  Total original size: {total_original_size:,} bytes ({total_original_size / (1024*1024):.2f} MB)")
        print(f"  Total compressed size: {total_compressed_size:,} bytes ({total_compressed_size / (1024*1024):.2f} MB)")
        print(f"  Total reduction: {total_reduction:.1f}%")
        print(f"  Space saved: {(total_original_size - total_compressed_size) / (1024*1024):.2f} MB")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python compress_logs.py <file> [--no-backup]")
        print("   or: python compress_logs.py --all [--no-backup]")
        print("\nOptions:")
        print("  --all: Compress all train-log-*.txt files in current directory")
        print("  --no-backup: Don't create backup files")
        sys.exit(1)
    
    backup = '--no-backup' not in sys.argv
    
    if sys.argv[1] == '--all':
        compress_all_logs(backup=backup)
    else:
        input_file = sys.argv[1]
        if not os.path.exists(input_file):
            print(f"Error: File '{input_file}' not found.")
            sys.exit(1)
        
        compress_log_file(input_file, backup=backup)


if __name__ == '__main__':
    main()

