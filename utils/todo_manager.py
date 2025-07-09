#!/usr/bin/env python3
"""
Todo Manager for AI Safety Research
Converts between Claude's TodoWrite format and markdown files
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import re

class TodoManager:
    """Manages todo items and synchronizes with todo.md files"""
    
    def __init__(self, todo_file: str = "todo.md"):
        self.todo_file = Path(todo_file)
        
    def parse_markdown_todos(self) -> List[Dict]:
        """Parse todos from markdown file"""
        if not self.todo_file.exists():
            return []
            
        with open(self.todo_file, 'r') as f:
            content = f.read()
            
        todos = []
        current_section = "pending"
        
        # Simple parser for todo.md format
        lines = content.split('\n')
        for line in lines:
            if line.startswith('## '):
                if 'Pending' in line:
                    current_section = 'pending'
                elif 'In Progress' in line:
                    current_section = 'in_progress'
                elif 'Completed' in line:
                    current_section = 'completed'
            elif line.startswith('- [ ] ') or line.startswith('- [x] '):
                checked = line.startswith('- [x] ')
                content = line[6:].strip()
                
                # Extract priority if present
                priority_match = re.search(r'\[([HML])\]', content)
                priority = 'medium'
                if priority_match:
                    p = priority_match.group(1)
                    priority = {'H': 'high', 'M': 'medium', 'L': 'low'}[p]
                    content = re.sub(r'\s*\[[HML]\]\s*', ' ', content).strip()
                
                todos.append({
                    'content': content,
                    'status': 'completed' if checked else current_section,
                    'priority': priority
                })
                
        return todos
    
    def write_markdown_todos(self, todos: List[Dict]) -> None:
        """Write todos to markdown file"""
        pending = [t for t in todos if t['status'] == 'pending']
        in_progress = [t for t in todos if t['status'] == 'in_progress']
        completed = [t for t in todos if t['status'] == 'completed']
        
        content = []
        content.append(f"# Todo List")
        content.append(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        content.append("")
        
        if in_progress:
            content.append("## 🚧 In Progress")
            for todo in in_progress:
                priority_marker = {'high': '[H]', 'medium': '[M]', 'low': '[L]'}[todo['priority']]
                content.append(f"- [ ] {priority_marker} {todo['content']}")
            content.append("")
            
        if pending:
            content.append("## 📋 Pending")
            for todo in sorted(pending, key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x['priority']]):
                priority_marker = {'high': '[H]', 'medium': '[M]', 'low': '[L]'}[todo['priority']]
                content.append(f"- [ ] {priority_marker} {todo['content']}")
            content.append("")
            
        if completed:
            content.append("## ✅ Completed")
            for todo in completed:
                content.append(f"- [x] {todo['content']}")
            content.append("")
            
        with open(self.todo_file, 'w') as f:
            f.write('\n'.join(content))
            
    def add_todo(self, content: str, priority: str = 'medium') -> None:
        """Add a new todo item"""
        todos = self.parse_markdown_todos()
        todos.append({
            'content': content,
            'status': 'pending',
            'priority': priority
        })
        self.write_markdown_todos(todos)
        
    def update_todo_status(self, content_pattern: str, new_status: str) -> bool:
        """Update the status of a todo item"""
        todos = self.parse_markdown_todos()
        
        for todo in todos:
            if content_pattern.lower() in todo['content'].lower():
                todo['status'] = new_status
                self.write_markdown_todos(todos)
                return True
                
        return False
        

def main():
    """CLI interface for todo management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage research todos")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Add todo command
    add_parser = subparsers.add_parser("add", help="Add a new todo")
    add_parser.add_argument("content", help="Todo content")
    add_parser.add_argument("--priority", choices=['high', 'medium', 'low'], default='medium')
    
    # Update status command
    update_parser = subparsers.add_parser("update", help="Update todo status")
    update_parser.add_argument("pattern", help="Content pattern to match")
    update_parser.add_argument("status", choices=['pending', 'in_progress', 'completed'])
    
    # List todos command
    list_parser = subparsers.add_parser("list", help="List all todos")
    
    args = parser.parse_args()
    
    manager = TodoManager()
    
    if args.command == "add":
        manager.add_todo(args.content, args.priority)
        print(f"Added todo: {args.content}")
        
    elif args.command == "update":
        if manager.update_todo_status(args.pattern, args.status):
            print(f"Updated todo status to: {args.status}")
        else:
            print(f"No todo found matching: {args.pattern}")
            
    elif args.command == "list":
        todos = manager.parse_markdown_todos()
        
        for status in ['in_progress', 'pending', 'completed']:
            status_todos = [t for t in todos if t['status'] == status]
            if status_todos:
                print(f"\n{status.replace('_', ' ').title()}:")
                for todo in status_todos:
                    marker = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}[todo['priority']]
                    print(f"  {marker} {todo['content']}")
                    

if __name__ == "__main__":
    main()