#!/usr/bin/env python3
"""
Session Manager for AI Safety Research
Helps manage Claude Code sessions and maintain context across research tasks
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

class SessionManager:
    """Manages research sessions and checkpoints"""
    
    def __init__(self, base_dir: str = ".claude/sessions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def create_checkpoint(self, name: str, metadata: Dict) -> Path:
        """
        Create a session checkpoint
        
        Args:
            name: Checkpoint name
            metadata: Session metadata (issue number, description, etc.)
            
        Returns:
            Path to checkpoint file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{timestamp}_{name}.json"
        checkpoint_path = self.base_dir / checkpoint_name
        
        checkpoint_data = {
            "name": name,
            "timestamp": timestamp,
            "metadata": metadata,
            "created_at": datetime.now().isoformat()
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
            
        return checkpoint_path
    
    def list_checkpoints(self, issue_number: Optional[int] = None) -> List[Dict]:
        """
        List all checkpoints, optionally filtered by issue number
        
        Args:
            issue_number: Optional issue number to filter by
            
        Returns:
            List of checkpoint metadata
        """
        checkpoints = []
        
        for checkpoint_file in self.base_dir.glob("*.json"):
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                
            if issue_number is None or data.get("metadata", {}).get("issue_number") == issue_number:
                checkpoints.append({
                    "file": checkpoint_file.name,
                    "name": data.get("name"),
                    "created_at": data.get("created_at"),
                    "metadata": data.get("metadata", {})
                })
                
        return sorted(checkpoints, key=lambda x: x["created_at"], reverse=True)
    
    def get_checkpoint(self, checkpoint_file: str) -> Dict:
        """Load a specific checkpoint"""
        checkpoint_path = self.base_dir / checkpoint_file
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
            
        with open(checkpoint_path, 'r') as f:
            return json.load(f)


def main():
    """CLI interface for session management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage Claude Code research sessions")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Create checkpoint command
    create_parser = subparsers.add_parser("create", help="Create a new checkpoint")
    create_parser.add_argument("name", help="Checkpoint name")
    create_parser.add_argument("--issue", type=int, help="Issue number")
    create_parser.add_argument("--description", help="Description")
    
    # List checkpoints command
    list_parser = subparsers.add_parser("list", help="List checkpoints")
    list_parser.add_argument("--issue", type=int, help="Filter by issue number")
    
    # Get checkpoint command
    get_parser = subparsers.add_parser("get", help="Get checkpoint details")
    get_parser.add_argument("file", help="Checkpoint filename")
    
    args = parser.parse_args()
    
    manager = SessionManager()
    
    if args.command == "create":
        metadata = {
            "issue_number": args.issue,
            "description": args.description or ""
        }
        path = manager.create_checkpoint(args.name, metadata)
        print(f"Created checkpoint: {path}")
        
    elif args.command == "list":
        checkpoints = manager.list_checkpoints(args.issue)
        
        if not checkpoints:
            print("No checkpoints found")
        else:
            for cp in checkpoints:
                print(f"\n{cp['file']}")
                print(f"  Name: {cp['name']}")
                print(f"  Created: {cp['created_at']}")
                if cp['metadata'].get('issue_number'):
                    print(f"  Issue: #{cp['metadata']['issue_number']}")
                if cp['metadata'].get('description'):
                    print(f"  Description: {cp['metadata']['description']}")
                    
    elif args.command == "get":
        try:
            checkpoint = manager.get_checkpoint(args.file)
            print(json.dumps(checkpoint, indent=2))
        except FileNotFoundError as e:
            print(f"Error: {e}")
            

if __name__ == "__main__":
    main()