#!/usr/bin/env python3
"""
OpenRouter Proxy Monitor Script

Continuously monitors a directory for request_*.json files, processes them,
and writes responses to response_*.json files.
"""

import os
import json
import asyncio
import aiohttp
import time
import logging
import shutil
from pathlib import Path
from typing import Set
import errno

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MONITOR_DIR = Path("/home/jz4391/openrouter_proxy")
PROCESSED_DIR = MONITOR_DIR / "processed"
POLL_INTERVAL = 1.0  # seconds


async def prompt_openrouter(url, headers, payload, timeout) -> str:
    """Make a request to OpenRouter API and return the response content."""
    connector = aiohttp.TCPConnector(force_close=True)
    async with aiohttp.ClientSession(
        connector=connector,
        json_serialize=lambda x: json.dumps(x, ensure_ascii=False)
    ) as session:
        response = await session.post(
            url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout)
        )
        data = await response.json(encoding='utf-8')
        return data["choices"][0]["message"]["content"]


async def process_request(request_json_fpath: Path) -> str:
    """Process a request JSON file and return the result."""
    with open(request_json_fpath, 'r') as f:
        request_json = json.load(f)
    
    url = request_json['url']
    headers = request_json['headers']
    payload = request_json['payload']
    timeout = request_json['timeout']
    
    return await prompt_openrouter(url, headers, payload, timeout)


async def process_file(request_file: Path, processed_files: Set[str]) -> None:
    """Process a single request file."""
    filename = request_file.name
    
    # Skip if already processed
    if filename in processed_files:
        return
    
    # Skip if response file already exists
    response_filename = filename.replace("request_", "response_")
    response_file = request_file.parent / response_filename
    if response_file.exists():
        logger.info(f"Response file already exists for {filename}, skipping")
        processed_files.add(filename)
        return
    
    try:
        logger.info(f"Processing {filename}...")
        
        # Process the request
        result = await process_request(request_file)
        
        # Write response file
        response_data = {"result": result}
        with open(response_file, 'w') as f:
            json.dump(response_data, f, indent=2)
        
        logger.info(f"Successfully wrote response to {response_filename}")
        
        # Move request file to processed folder
        processed_path = PROCESSED_DIR / filename
        if not request_file.exists():
            logger.warning(f"Request file {filename} no longer exists, skipping move")
        else:
            source_path = str(request_file.resolve())  # Use absolute path
            dest_path = str(processed_path.resolve())   # Use absolute path
            logger.debug(f"Moving {source_path} to {dest_path}")
            
            # Try os.rename first (atomic move on same filesystem)
            try:
                os.rename(source_path, dest_path)
            except OSError as e:
                if e.errno == errno.EXDEV:
                    # Cross-filesystem move, use shutil.move
                    logger.debug(f"Cross-filesystem move detected, using shutil.move")
                    shutil.move(source_path, dest_path)
                else:
                    raise
            
            # Verify the move succeeded
            if os.path.exists(source_path):
                logger.error(f"Failed to move {filename} - file still exists at {source_path}")
                raise Exception(f"File move failed: {filename} still exists in {MONITOR_DIR}")
            if not os.path.exists(dest_path):
                logger.error(f"Failed to move {filename} - file not found at {dest_path}")
                raise Exception(f"File move failed: {filename} not found in {PROCESSED_DIR}")
            logger.info(f"Successfully moved {filename} from {MONITOR_DIR} to {PROCESSED_DIR}")
        
        # Mark as processed
        processed_files.add(filename)
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}", exc_info=True)
        # Don't add to processed_files so it can be retried


async def monitor_loop() -> None:
    """Main monitoring loop."""
    # Ensure processed directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Track processed files to avoid reprocessing
    processed_files: Set[str] = set()
    
    logger.info(f"Starting monitor for directory: {MONITOR_DIR}")
    logger.info(f"Polling interval: {POLL_INTERVAL} seconds")
    
    while True:
        try:
            # List all files in the directory
            if not MONITOR_DIR.exists():
                logger.warning(f"Monitor directory does not exist: {MONITOR_DIR}")
                await asyncio.sleep(POLL_INTERVAL)
                continue
            
            # Find request_*.json files
            request_files = [
                MONITOR_DIR / f
                for f in os.listdir(MONITOR_DIR)
                if f.startswith("request_") and f.endswith(".json")
                and (MONITOR_DIR / f).is_file()
            ]
            
            # Process each file
            for request_file in request_files:
                await process_file(request_file, processed_files)
            
            # Wait before next poll
            await asyncio.sleep(POLL_INTERVAL)
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
            break
        except Exception as e:
            logger.error(f"Error in monitor loop: {e}", exc_info=True)
            await asyncio.sleep(POLL_INTERVAL)


def main():
    """Entry point."""
    try:
        asyncio.run(monitor_loop())
    except KeyboardInterrupt:
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
